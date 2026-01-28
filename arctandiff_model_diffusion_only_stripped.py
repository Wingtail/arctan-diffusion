from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_func  # type: ignore
except Exception:  # pragma: no cover
    flash_attn_func = None

from arctandiff_utility import *

import torch.distributed as dist
import numpy as np

# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision.ops import MLP

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Classic fixed sinusoidal positional embeddings (Vaswani et al. 2017).

    Returns embeddings of shape:
      - (1, T, D) if batch_first=True
      - (T, 1, D) if batch_first=False
    so you can add them to your token embeddings by broadcasting.
    """
    def __init__(self, dim: int, max_len: int = 2048, batch_first: bool = True):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        self.dim = dim
        self.max_len = max_len
        self.batch_first = batch_first

        pe = self._build_table(max_len, dim)  # (max_len, dim)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_table(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)

        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )  # (ceil(D/2),)

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def _maybe_extend(self, needed_len: int) -> None:
        if needed_len <= self.pe.size(0):
            return
        # extend table on the fly
        new_len = max(needed_len, int(self.pe.size(0) * 1.5))
        self.pe = self._build_table(new_len, self.dim).to(self.pe.device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: token embeddings
          - (B, T, D) if batch_first=True
          - (T, B, D) if batch_first=False

        start_pos: offset for generation/caching (e.g., past length)
        positions: optional explicit positions (T,) or (B, T) if you need custom indexing.
        """
        if self.batch_first:
            if x.dim() != 3:
                raise ValueError("Expected x shape (B, T, D) for batch_first=True")
            B, T, D = x.shape
        else:
            if x.dim() != 3:
                raise ValueError("Expected x shape (T, B, D) for batch_first=False")
            T, B, D = x.shape

        if D != self.dim:
            raise ValueError(f"Embedding dim mismatch: x has D={D}, module dim={self.dim}")

        if positions is None:
            needed = start_pos + T
            self._maybe_extend(needed)
            pos_emb = self.pe[start_pos : start_pos + T]  # (T, D)
        else:
            # positions can be (T,) or (B, T)
            if positions.dim() == 1:
                needed = int(positions.max().item()) + 1
                self._maybe_extend(needed)
                pos_emb = self.pe[positions]  # (T, D)
            elif positions.dim() == 2:
                needed = int(positions.max().item()) + 1
                self._maybe_extend(needed)
                pos_emb = self.pe[positions]  # (B, T, D)
            else:
                raise ValueError("positions must be shape (T,) or (B, T)")

        # Match dtype/device of x
        pos_emb = pos_emb.to(dtype=x.dtype, device=x.device)

        if positions is not None and positions.dim() == 2:
            # already (B, T, D)
            return pos_emb if self.batch_first else pos_emb.transpose(0, 1)

        # (T, D) -> broadcastable
        if self.batch_first:
            return pos_emb.unsqueeze(0)  # (1, T, D)
        return pos_emb.unsqueeze(1)      # (T, 1, D)


class LearnedPositionalEmbedding(nn.Module):
    """
    Trainable absolute positional embeddings using nn.Embedding.
    """
    def __init__(self, dim: int, max_len: int = 2048, batch_first: bool = True):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        self.dim = dim
        self.max_len = max_len
        self.batch_first = batch_first
        self.emb = nn.Embedding(max_len, dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.batch_first:
            if x.dim() != 3:
                raise ValueError("Expected x shape (B, T, D) for batch_first=True")
            B, T, D = x.shape
        else:
            if x.dim() != 3:
                raise ValueError("Expected x shape (T, B, D) for batch_first=False")
            T, B, D = x.shape

        if D != self.dim:
            raise ValueError(f"Embedding dim mismatch: x has D={D}, module dim={self.dim}")

        if positions is None:
            if start_pos + T > self.max_len:
                raise ValueError(
                    f"Sequence length {start_pos+T} exceeds max_len={self.max_len}. "
                    "Increase max_len or switch to sinusoidal."
                )
            pos = torch.arange(start_pos, start_pos + T, device=x.device)  # (T,)
            pos_emb = self.emb(pos)  # (T, D)
        else:
            if int(positions.max().item()) >= self.max_len:
                raise ValueError(
                    f"positions.max()={int(positions.max().item())} exceeds max_len={self.max_len}."
                )
            pos_emb = self.emb(positions.to(device=x.device))

        if positions is not None and positions.dim() == 2:
            return pos_emb if self.batch_first else pos_emb.transpose(0, 1)

        if self.batch_first:
            return pos_emb.unsqueeze(0)  # (1, T, D)
        return pos_emb.unsqueeze(1)      # (T, 1, D)


class PositionEmbedding(nn.Module):
    """
    Drop-in module to ADD position embeddings to token embeddings.

    Example:
        x = token_emb(input_ids)            # (B, T, D)
        x = pos_emb(x, start_pos=past_len)  # adds position embedding
    """
    def __init__(
        self,
        dim: int,
        max_len: int = 2048,
        *,
        kind: Literal["sinusoidal", "learned"] = "sinusoidal",
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.batch_first = batch_first

        if kind == "sinusoidal":
            self.pos = SinusoidalPositionalEmbedding(dim, max_len=max_len, batch_first=batch_first)
        elif kind == "learned":
            self.pos = LearnedPositionalEmbedding(dim, max_len=max_len, batch_first=batch_first)
        else:
            raise ValueError("kind must be 'sinusoidal' or 'learned'")

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.dropout(x + self.pos(x, start_pos=start_pos, positions=positions))

@torch.no_grad()
def init_dit_weights_(model: nn.Module, std: float = 0.02) -> None:
    """
    DiT-style init:
      - ViT/timm init for Linear/Conv/Embedding: trunc_normal_(std=0.02), bias=0
      - Norm scales: 1 (and bias 0 if present)
      - adaLN-Zero: zero last Linear in each DiTBlock.adaLN_modulation
      - (optional) zero final output head if you pass it / name it
    """

    def _vit_timm_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)

        # If you ever swap nn.LayerNorm -> LayerNorm, or add LN elsewhere.
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Your custom nn.LayerNorm (has only .weight)
        elif m.__class__.__name__ == "nn.LayerNorm":
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)

    # 1) Apply ViT/timm-style init everywhere
    model.apply(_vit_timm_init)

    # 2) adaLN-Zero: force modulation output to start at exactly 0
    for m in model.modules():
        if m.__class__.__name__ == "DiTBlock":
            last = m.adaLN_modulation[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    # 3) OPTIONAL: if you have a final prediction head, zero it too.
    #    Example patterns (uncomment / adapt to your model):
    #
    # if hasattr(model, "final_layer") and isinstance(model.final_layer, nn.Linear):
    #     nn.init.zeros_(model.final_layer.weight)
    #     nn.init.zeros_(model.final_layer.bias)
    #
    # if hasattr(model, "out") and isinstance(model.out, nn.Linear):
    #     nn.init.zeros_(model.out.weight)
    #     nn.init.zeros_(model.out.bias)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention that prefers FlashAttention 2 kernels when available,
    with safe fallback to PyTorch's scaled dot-product attention (math backend).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        is_causal: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        rotary_embedding=None,
        attn_backend: str = "auto",
        use_flash_attn: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = dropout
        if use_flash_attn is True:
            self.attn_backend = "flash_attn"
        elif use_flash_attn is False:
            self.attn_backend = "torch"
        else:
            self.attn_backend = attn_backend
        self.rotary_embedding = rotary_embedding

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3, dim=0)  # 3 x [B, num_heads, N, head_dim]
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        dropout_p = 0.0 if not self.training else self.dropout
        dtype_ok = q.dtype in (torch.float16, torch.bfloat16)
        is_cuda = q.is_cuda

        backend = (self.attn_backend or "auto").lower()

        if backend in {"flash_attn", "flash-attn"}:
            if flash_attn_func is None:
                raise RuntimeError("attn_backend='flash_attn' requires the flash_attn package.")
            if not (is_cuda and dtype_ok):
                raise RuntimeError(
                    f"attn_backend='flash_attn' requires CUDA + fp16/bf16 (got device={q.device}, dtype={q.dtype})."
                )
            q_ = q.transpose(1, 2)  # (B, N, nheads, headdim)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)
            out = flash_attn_func(q_, k_, v_, softmax_scale=self.scale, causal=self.is_causal, dropout_p=dropout_p)
        elif backend in {"torch_flash", "flash_sdp"}:
            if not (is_cuda and dtype_ok):
                raise RuntimeError(
                    f"attn_backend='{backend}' requires CUDA + fp16/bf16 (got device={q.device}, dtype={q.dtype})."
                )
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, is_causal=self.is_causal, dropout_p=dropout_p
                )  # [B, num_heads, N, head_dim]
            out = out.transpose(1, 2)  # (B, N, num_heads, head_dim)
        elif backend in {"flash"}:
            if not (is_cuda and dtype_ok):
                raise RuntimeError(
                    f"attn_backend='flash' requires CUDA + fp16/bf16 (got device={q.device}, dtype={q.dtype})."
                )
            if flash_attn_func is not None:
                q_ = q.transpose(1, 2)  # (B, N, nheads, headdim)
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                out = flash_attn_func(q_, k_, v_, softmax_scale=self.scale, causal=self.is_causal, dropout_p=dropout_p)
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    out = F.scaled_dot_product_attention(
                        q, k, v, scale=self.scale, is_causal=self.is_causal, dropout_p=dropout_p
                    )  # [B, num_heads, N, head_dim]
                out = out.transpose(1, 2)  # (B, N, num_heads, head_dim)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, scale=self.scale, is_causal=self.is_causal, dropout_p=dropout_p
            )  # [B, num_heads, N, head_dim]
            out = out.transpose(1, 2)  # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(out)

class FeedForward(nn.Module):
    """Simple Gated-SwiGLU feedforward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x = self.act(x_fc1)
        x = self.fc2(x)
        return self.dropout(x)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """Diffusion Transformer block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        is_causal=False,
        rotary_embedding=None,
        attn_backend: str = "auto"
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            dim,
            num_heads,
            is_causal=is_causal,
            rotary_embedding=rotary_embedding,
            attn_backend=attn_backend,
            dropout=attn_dropout
        )
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + (gate_mlp * self.mlp(mlp_in))
        
        return x

def sample_logit_normal_timesteps(
    batch_size: int,
    t: int,
    *,
    mu: float = -0.8,
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample t ~ LogitNormal(mu, sigma^2), then map to (0,1) via sigmoid.
    Defaults have more noise than with JiT (mu ≈ -0.8 for ImageNet 256).
    """
    device = device or torch.device("cpu")
    normal = torch.randn(batch_size, t, device=device, dtype=torch.float32) * sigma + mu
    t = torch.sigmoid(normal)  # (B,)
    return t.to(dtype=dtype)

def sample_uniform_timesteps(
    batch_size: int,
    t: int,
    a: float = 0.0,
    b: float = 1.0,
    sample_same_timesteps: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample ~ Uniform(a, b)
    """
    if b < a:
        raise ValueError(f"Expected b >= a, got a={a}, b={b}")
    device = device or torch.device("cpu")
    if sample_same_timesteps:
        u = torch.rand(batch_size, device=device, dtype=dtype).unsqueeze(-1).expand(-1, t)  # in [0, 1)
        return u * (b - a) + a
    u = torch.rand(batch_size, t, device=device, dtype=dtype)  # in [0, 1)
    return u * (b - a) + a

def sample_same_timesteps_(
    batch_size: int,
    t: int,
    mu: float = -1.3,
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    device = device or torch.device("cpu")
    normal = torch.randn(batch_size, device=device, dtype=torch.float32) * sigma + mu
    t = torch.sigmoid(normal).unsqueeze(-1).expand(-1, t)  # (B,)
    return t.to(dtype=dtype)

class ClsEmbedding(nn.Module):
    def __init__(self, num_features, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2) 

def patch_timeseries(batch, patch_size=2, pad_mode: str = "right"):
    B, T, C = batch.shape
    if T%patch_size != 0:
        pad_len = patch_size - (T % patch_size)
        if pad_mode == "right":
            pad_left = 0
            pad_right = pad_len
        elif pad_mode == "symmetric":
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode}")

        if pad_left > 0:
            left_pad = torch.zeros(B, pad_left, C, device=batch.device, dtype=batch.dtype)
            batch = torch.cat([left_pad, batch], dim=1)
        if pad_right > 0:
            right_pad = torch.zeros(B, pad_right, C, device=batch.device, dtype=batch.dtype)
            batch = torch.cat([batch, right_pad], dim=1)
        T = T + pad_len

    return batch.reshape(B, T // patch_size, patch_size, C)

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = d_model // 2
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class JiTEncoderModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_layers, rotary_embedding=None, dropout=0.0, downstream_task="forecasting", attn_dropout=0.0, mlp_ratio=4.0, is_causal=False, patch_size=8, attn_backend="auto"):
        super().__init__()
        self.rotary_embedding = None
        self.n_layers = n_layers
        
        self.position_embedding = PositionEmbedding(dim=hidden_dim)
        
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                is_causal=is_causal,
                attn_backend=attn_backend,
                rotary_embedding=None,
            ) for _ in range(n_layers)
        ])
        
        self.patch_size = patch_size
        
        if downstream_task == 'classification':
            self.input_layer = nn.Linear(in_dim*self.patch_size, hidden_dim)
        else:
            self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.final_layernorm = nn.LayerNorm(hidden_dim)
        
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        
        init_dit_weights_(self, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        # t: [B, T]
        x = self.input_layer(x)  # [B, T, hidden_dim]
        x = self.position_embedding(x)  # [B, T, hidden_dim]

        t = self.time_embedding(t)  # [B, T, hidden_dim]

        for i, block in enumerate(self.dit_blocks):
            x = block(x, t)
        
        x = self.final_layernorm(x)
        x = self.projector(x)

        return x
    
    def extract_intermediate_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        # t: [B, T]
        x = self.input_layer(x)  # [B, T, hidden_dim]
        x = self.position_embedding(x)  # [B, T, hidden_dim]

        t = self.time_embedding(t)  # [B, T, hidden_dim]

        features = []
        for i, block in enumerate(self.dit_blocks):
            x = block(x, t)
            features.append(x)
        
        x = self.final_layernorm(x)
        x = self.projector(x)
        
        features.append(x)
        return features

class DiffusionReconstructionModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_layers, dropout=0.0, attn_dropout=0.0, mlp_ratio=2.0, is_causal=False, downstream_task='forecasting', rotary_embedding=None, patch_size=8, time_embedding=None, attn_backend="auto"):
        super().__init__()

        self.n_layers = n_layers
        
        self.block = nn.ModuleList([DiTBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            is_causal=is_causal,
            attn_backend=attn_backend,
            rotary_embedding=None,
        ) for _ in range(n_layers)])
        
        self.patch_size = patch_size
        
        self.projector = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, in_dim*self.patch_size) if downstream_task == 'classification' else nn.Linear(hidden_dim, in_dim))
        
        self.time_embedding = time_embedding

        init_dit_weights_(self, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        # x: [B, T, hidden_dim]
        # t: [B, T]
        
        t = self.time_embedding(t)  # [B, T, hidden_dim]
        if cond is not None:
            t = t + cond
        
        for i, layer in enumerate(self.block):
            x = layer(x, t)

        x = self.projector(x)
        return x

class ArcTanDiffusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_layers, model_type="half_normalized", dropout=0.0, attn_dropout=0.0, mlp_ratio=4.0, is_causal=False, downstream_task='forecasting', input_len=336, patch_size=8, recon_head_depth=2, loss_type='arctan', model_pred_type="x0", attn_backend="auto"):
        super().__init__()
        
        self.rotary_embedding = None
        self.n_layers = n_layers
        
        self.encoder = JiTEncoderModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_layers=n_layers,
            rotary_embedding=None,
            dropout=dropout,
            attn_dropout=attn_dropout,
            mlp_ratio=mlp_ratio,
            is_causal=is_causal,
            patch_size=patch_size,
            downstream_task=downstream_task,
            attn_backend=attn_backend
        )

        self.input_len = input_len
        
        self.patch_size = patch_size
        
        self.downstream_task = downstream_task

        if model_pred_type not in {"x0", "e", "v"}:
            raise ValueError(f"Unknown model_pred_type: {model_pred_type}. Expected one of ['x0', 'e', 'v'].")
        self.model_pred_type = model_pred_type
        
        print("Setting recon head depth: ", recon_head_depth)
        self.reconstruction_head = DiffusionReconstructionModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_layers=recon_head_depth,
            dropout=dropout,
            attn_dropout=attn_dropout,
            mlp_ratio=mlp_ratio,
            is_causal=is_causal,
            patch_size=patch_size,
            downstream_task=downstream_task,
            rotary_embedding=None,
            time_embedding=self.encoder.time_embedding,
            attn_backend=attn_backend
        )

        init_dit_weights_(self, std=0.02)
        
        if loss_type == 'arctan':
            print("Setting loss type to arctan")
            self.diffusion_criterion = ArcTanLoss()
        elif loss_type == 'mse':
            print("Setting loss type to mse")
            self.diffusion_criterion = nn.MSELoss()
        elif loss_type == 'huber':
            print("Setting loss type to huber")
            self.diffusion_criterion = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        # t: [B, T]
        x = self.encoder(x, t)

        return x

    def sample_timesteps(
        self,
        x,
        t_low=0.0,
        t_high=1.0,
        target='v',
        sample_same=False,
        sample_mixed=False,
        timestep_sampling: str = "uniform",
    ):
        N, B, T, C = x.shape
        
        batch_size = N * B
        
        if t_low == 1.0:
            t = torch.ones(batch_size, T, device=x.device, dtype=x.dtype)
        else:
            if sample_mixed:
                sample_same = (np.random.rand() < 0.5)
            
            if sample_same:
                t = sample_same_timesteps_(
                    batch_size,
                    T,
                    device=x.device,
                    dtype=x.dtype
                )
            else:
                if timestep_sampling == "uniform":
                    t = sample_uniform_timesteps(
                        batch_size,
                        T,
                        a=t_low,
                        b=t_high,
                        device=x.device,
                        dtype=x.dtype
                    )
                elif timestep_sampling == "logit_normal":
                    if t_high < t_low:
                        raise ValueError(f"Expected t_high >= t_low, got t_low={t_low}, t_high={t_high}")
                    t = sample_logit_normal_timesteps(
                        batch_size,
                        T,
                        mu=0.8,
                        sigma=1.0,
                        device=x.device,
                        dtype=x.dtype
                    )
                    if t_low != 0.0 or t_high != 1.0:
                        t = t * (t_high - t_low) + t_low
                else:
                    raise ValueError(f"Unknown timestep_sampling: {timestep_sampling}")

        t = t.reshape(N, B, T)
        e = torch.randn_like(x)
        z = t.unsqueeze(-1) * x + (1 - t).unsqueeze(-1) * e  # Noised input

        if target == 'v':
            # print("Using v target")
            target = (x-z) / (1.0 - t).unsqueeze(-1).clamp(min=0.05)
        elif target == 'x0':
            # print("Using x0 target")
            target = x
        elif target == 'e':
            # print("Using e target")
            target = e
        
        t_input = (t*1000.0).long()
        
        return z, target, t, t_input

    def prep_input(self, x):
        pad_mode = "symmetric" if self.downstream_task == "classification" else "right"
        x = patch_timeseries(x, patch_size=self.patch_size, pad_mode=pad_mode) #{B, T//patch_size, patch_size, C}
        B, T, P, C = x.shape
        
        if self.downstream_task == 'classification':
            x = x.reshape(x.size(0), x.size(1), -1)  # flatten patch dimension
        elif self.downstream_task == 'forecasting':
            x = x.permute(0, 3, 1, 2).reshape(B*C, T, P)
        
        if self.downstream_task == 'classification':
            x = x.reshape(B, T, P, C).reshape(B, T, P*C)
        
        return x

    def clean_forward(self, x: torch.Tensor, t=1.0, extract_features=False) -> torch.Tensor:
        pad_mode = "symmetric" if self.downstream_task == "classification" else "right"
        x = patch_timeseries(x, patch_size=self.patch_size, pad_mode=pad_mode) #{B, T//patch_size, patch_size, C}
        B, T, P, C = x.shape
        
        clean_t = torch.ones(B, T, device=x.device, dtype=x.dtype) * t
        
        if self.downstream_task == 'classification':
            x = x.reshape(x.size(0), x.size(1), -1)  # flatten patch dimension
        elif self.downstream_task == 'forecasting':
            x = x.permute(0, 3, 1, 2).reshape(B*C, T, P)
            
            clean_t = clean_t.unsqueeze(-1).expand(-1,-1,C) # [B, T, C]
            clean_t = clean_t.permute(0,2,1).reshape(B*C, T)  #[B*C, T]
        
        if self.downstream_task == 'classification':
            x = x.reshape(B, T, P, C).reshape(B, T, P*C)

        if t < 1.0:
            x = clean_t.unsqueeze(-1) * x + (1 - clean_t).unsqueeze(-1) * torch.randn_like(x)  # noised input

        t_input = (clean_t*1000.0).long() 

        if extract_features:
            features = self.encoder.extract_intermediate_features(x, t_input)
        else:
            features = self.encoder(x, t_input)

        return features

    def train_step(self, x, timestep_sampling="logit_normal", diffusion_loss_type='v', sample_same_timesteps=False, sample_mixed=False, model_pred_type=None):
        x_raw = x  # [B, T, in_dim]
        x = self.prep_input(x)  # [B_effective, T, in_dim]
        x_diffusion = x.unsqueeze(0) # [1, B_effective, T, in_dim]

        if timestep_sampling == "logit_normal":
            t_low = 0.8
            t_high = 1.0
        elif timestep_sampling == "uniform":
            t_low = 0.0
            t_high = 1.0

        z, diffusion_target, diffusion_t, diffusion_t_input = self.sample_timesteps(x_diffusion, t_low=t_low, t_high=t_high, target=diffusion_loss_type, sample_same=sample_same_timesteps, sample_mixed=sample_mixed, timestep_sampling=timestep_sampling)
        
        # print("diffusion t: ", diffusion_t)
        
        N, B_effective, T, in_dim = z.shape
        
        z = z.reshape(-1, T, in_dim)  # [(1+n_global_views)*B_effective, T, in_dim]
        t_input_diffusion_input = diffusion_t_input.reshape(-1, T)  # [(1+n_global_views)*B_effective, T]
        
        features = self.forward(z, t_input_diffusion_input)
        
        features = features.reshape(N, -1, T, features.size(-1))

        # clean_features = self.clean_forward(x_raw, t=1.0, extract_features=False)
        # #pool clean features
        # clean_features = torch.amax(clean_features, dim=1, keepdim=True)  # [1, B_effective, hidden_dim]

        ## Diffusion reconstruction loss
        pred = self.reconstruction_head(features[0], diffusion_t_input[0])  # only use local view wo CLS for diffusion loss

        if diffusion_loss_type not in {"x0", "e", "v"}:
            raise ValueError(f"Unknown diffusion_loss_type: {diffusion_loss_type}. Expected one of ['x0', 'e', 'v'].")
        pred_type = model_pred_type or self.model_pred_type
        if pred_type not in {"x0", "e", "v"}:
            raise ValueError(f"Unknown model_pred_type: {pred_type}. Expected one of ['x0', 'e', 'v'].")

        z0 = z.squeeze(0)
        t0 = diffusion_t.squeeze(0)
        denom = (1.0 - t0).clamp(min=0.05)
        t_denom = t0.clamp(min=0.05)

        if pred_type == "x0":
            x0_pred = pred
        elif pred_type == "e":
            x0_pred = (z0 - denom.unsqueeze(-1) * pred) / t_denom.unsqueeze(-1)
        else:  # pred_type == "v"
            x0_pred = z0 + denom.unsqueeze(-1) * pred

        if diffusion_loss_type == pred_type:
            pred_for_loss = pred
        elif diffusion_loss_type == "x0":
            pred_for_loss = x0_pred
        elif diffusion_loss_type == "e":
            pred_for_loss = (z0 - t0.unsqueeze(-1) * x0_pred) / denom.unsqueeze(-1)
        else:  # diffusion_loss_type == "v"
            pred_for_loss = (x0_pred - z0) / denom.unsqueeze(-1)

        diffusion_loss = self.diffusion_criterion(pred_for_loss, diffusion_target.squeeze(0))
        ##
        
        loss = diffusion_loss

        return x0_pred, loss, {"diffusion_loss": diffusion_loss.item()}

##
class TimeEmbedding(nn.Module):
    def __init__(self, model_dim=8):
        super().__init__()
        self.model_dim = model_dim
        self.time_dim = model_dim  # 8

        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def sinusoidal_embedding(self, t):
        # t: [B, T]
        B, T = t.shape
        t = t.reshape(-1,1)
        half = self.time_dim // 2  # 4
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device) * -1
        )
        args = t * freqs  # [B, 1] * [4] -> [B, 4]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 8]
        
        emb = emb.reshape(B, T, self.time_dim)
        
        return emb

    def forward(self, t):
        emb = self.sinusoidal_embedding(t)  # [B, 8]
        return self.mlp(emb)  # [B, 8]

