from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import generate_causal_mask, generate_self_only_mask, generate_partial_mask


import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Sequence

import torch.jit
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )

    _HAS_FLEX_ATTN = True
except Exception:  # pragma: no cover - optional dependency
    _flex_attention = None
    _create_block_mask = None
    _HAS_FLEX_ATTN = False

try:
    from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore

    _HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - optional dependency
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore

        _HAS_FLASH_ATTN = True
    except Exception:  # pragma: no cover - optional dependency
        _flash_attn_func = None
        _HAS_FLASH_ATTN = False


class ChannelIndependence(nn.Module):
    def __init__(
        self,
    ):
        super(ChannelIndependence, self).__init__()

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, input_len, 1)
        return x


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output

class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True, custom_mask=None):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        if custom_mask is None:
            mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        else:
            mask = custom_mask
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x

class LinearReluAttention(nn.Module):
    """
    Linear-time attention with a ReLU feature map and optional numerator-only RoPE.

    The attention computes prefix sums for causal mode, exposing (S, z) states for
    streaming / Self-Forcing rollouts. Bidirectional mode falls back to global sums.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        is_causal: bool,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-6,
        rope_numerator_only: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        norm_type: str = "stella",
        learnable_alpha: bool = True,
        learnable_beta: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert norm_type in ("softmax_like", "stella")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        self.rope_numerator_only = rope_numerator_only
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.norm_type = norm_type
        self.learnable_alpha = learnable_alpha
        self.learnable_beta = learnable_beta

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        if self.norm_type == "stella":
            self.alpha_log = nn.Parameter(
                torch.zeros(self.num_heads), requires_grad=learnable_alpha
            )
            self.beta_log = nn.Parameter(
                torch.zeros(self.num_heads), requires_grad=learnable_beta
            )
        else:
            self.register_parameter("alpha_log", None)
            self.register_parameter("beta_log", None)

    def _positive_feature(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.eps

    def _stella_alpha_beta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return positive per-head scaling factors (α', β') used in SteLLA.

        Softplus keeps the parameters strictly positive while allowing unconstrained
        optimization in log-space.
        """
        alpha_prime = F.softplus(self.alpha_log) + 1e-6
        beta_prime = F.softplus(self.beta_log) + 1e-6
        return alpha_prime, beta_prime

    def _apply_rope_if_needed(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        *,
        rope_pos: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return phi_q, phi_k

        seq_len = phi_q.size(2)
        if rope_pos is None:
            cos, sin = _rope_angles(
                seq_len,
                self.head_dim,
                device=phi_q.device,
                dtype=phi_q.dtype,
                theta=self.rope_theta,
            )
        else:
            cos, sin = rope_pos
        return _apply_rope(phi_q, cos, sin), _apply_rope(phi_k, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        rope_pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        chunk_causal: int = 0,
        prefix_len: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if attn_mask is not None:
            raise NotImplementedError("LinearReluAttention does not support custom attention masks")

        B, N, C = x.shape
        if prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        if prefix_len > 0 and not self.is_causal:
            raise ValueError("prefix_len is only supported for causal LinearReluAttention")
        if prefix_len > N:
            raise ValueError("prefix_len cannot exceed the current sequence length")
        if prefix_len > 0 and chunk_causal > 0:
            raise ValueError("chunk_causal is not supported when prefix_len > 0")
        if prefix_len > 0 and causal_state is not None:
            raise ValueError("prefix_len with cached causal_state is not supported")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        phi_q_base = self._positive_feature(q)
        phi_k_base = self._positive_feature(k)
        phi_q_rope, phi_k_rope = self._apply_rope_if_needed(phi_q_base, phi_k_base, rope_pos=rope_pos)

        if self.norm_type == "stella":
            phi_q_num = phi_q_den = phi_q_rope
            phi_k_num = phi_k_den = phi_k_rope
        elif self.use_rope and self.rope_numerator_only:
            phi_q_num, phi_q_den = phi_q_rope, phi_q_base
            phi_k_num, phi_k_den = phi_k_rope, phi_k_base
        else:
            phi_q_num = phi_q_den = phi_q_rope
            phi_k_num = phi_k_den = phi_k_rope

        state_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if not self.is_causal:
            if self.norm_type == "softmax_like":
                kv = torch.einsum("bhnd,bhne->bhde", phi_k_num, v)
                z = torch.einsum("bhnd->bhd", phi_k_den)
                out = torch.einsum("bhnd,bhde->bhne", phi_q_num, kv)
                denom = torch.einsum("bhnd,bhd->bhn", phi_q_den, z).unsqueeze(-1)
                out = out / denom.clamp_min(self.eps)
            else:
                kv = torch.einsum("bhnd,bhne->bhde", phi_k_num, v)
                out = torch.einsum("bhnd,bhde->bhne", phi_q_num, kv)

                alpha_prime, beta_prime = self._stella_alpha_beta()
                sqrt_d = float(self.head_dim) ** 0.5
                scale = (alpha_prime * beta_prime * sqrt_d * float(N)).view(1, self.num_heads, 1, 1).to(out.dtype)
                out = out / scale.clamp_min(self.eps)
        else:
            if causal_state is not None:
                S_prev, z_prev = causal_state
            else:
                S_prev = None
                z_prev = None

            def _causal_step(
                phi_q_num_chunk: torch.Tensor,
                phi_q_den_chunk: torch.Tensor,
                phi_k_num_chunk: torch.Tensor,
                phi_k_den_chunk: torch.Tensor,
                v_chunk: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]],
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                S_state, z_state = state if state is not None else (None, None)
                kv_chunk = torch.einsum("bhnd,bhne->bhnde", phi_k_num_chunk, v_chunk)
                S_chunk = kv_chunk.cumsum(dim=2)
                if S_state is not None:
                    S_chunk = S_chunk + S_state.unsqueeze(2)
                if self.norm_type == "softmax_like":
                    z_chunk = phi_k_den_chunk.cumsum(dim=2)
                    if z_state is not None:
                        z_chunk = z_chunk + z_state.unsqueeze(2)

                    out_chunk = torch.einsum("bhnd,bhnde->bhne", phi_q_num_chunk, S_chunk)
                    denom_chunk = torch.einsum("bhnd,bhnd->bhn", phi_q_den_chunk, z_chunk).unsqueeze(-1)
                    out_chunk = out_chunk / denom_chunk.clamp_min(self.eps)
                    new_state = (S_chunk[:, :, -1], z_chunk[:, :, -1])
                else:
                    B_chunk, H_chunk, L_chunk, D_chunk = phi_q_num_chunk.shape
                    if z_state is None:
                        prev_len = torch.zeros(
                            (B_chunk, H_chunk),
                            device=phi_q_num_chunk.device,
                            dtype=phi_q_num_chunk.dtype,
                        )
                    else:
                        prev_len = z_state[..., 0]

                    offsets = torch.arange(
                        1,
                        L_chunk + 1,
                        device=phi_q_num_chunk.device,
                        dtype=phi_q_num_chunk.dtype,
                    )
                    prefix_len_chunk = prev_len.unsqueeze(-1) + offsets  # (B,H,L)

                    out_chunk = torch.einsum("bhnd,bhnde->bhne", phi_q_num_chunk, S_chunk)

                    alpha_prime, beta_prime = self._stella_alpha_beta()
                    sqrt_d = float(self.head_dim) ** 0.5
                    # Add a singleton length dimension so scaling broadcasts with (B, H, L, D)
                    scale = (alpha_prime * beta_prime * sqrt_d).view(1, H_chunk, 1, 1).to(out_chunk.dtype)
                    denom_chunk = scale * prefix_len_chunk.unsqueeze(-1)
                    out_chunk = out_chunk / denom_chunk.clamp_min(self.eps)

                    new_len = prefix_len_chunk[:, :, -1]
                    ones = torch.ones(
                        (B_chunk, H_chunk, D_chunk),
                        device=phi_q_num_chunk.device,
                        dtype=phi_q_num_chunk.dtype,
                    )
                    z_new = ones * new_len.unsqueeze(-1)
                    new_state = (S_chunk[:, :, -1], z_new)

                return out_chunk, new_state

            if prefix_len > 0:
                P = int(prefix_len)
                phi_q_num_pref = phi_q_num[:, :, :P]
                phi_q_den_pref = phi_q_den[:, :, :P]
                phi_k_num_pref = phi_k_num[:, :, :P]
                phi_k_den_pref = phi_k_den[:, :, :P]
                v_pref = v[:, :, :P]

                S_pref = torch.einsum("bhnd,bhne->bhde", phi_k_num_pref, v_pref)
                if self.norm_type == "softmax_like":
                    z_pref = torch.einsum("bhnd->bhd", phi_k_den_pref)
                    out_pref = torch.einsum("bhnd,bhde->bhne", phi_q_num_pref, S_pref)
                    denom_pref = torch.einsum("bhnd,bhd->bhn", phi_q_den_pref, z_pref).unsqueeze(-1)
                    out_pref = out_pref / denom_pref.clamp_min(self.eps)
                else:
                    alpha_prime, beta_prime = self._stella_alpha_beta()
                    sqrt_d = float(self.head_dim) ** 0.5
                    out_pref = torch.einsum("bhnd,bhde->bhne", phi_q_num_pref, S_pref)
                    scale_pref = (alpha_prime * beta_prime * sqrt_d * float(P)).view(1, self.num_heads, 1, 1).to(
                        out_pref.dtype
                    )
                    out_pref = out_pref / scale_pref.clamp_min(self.eps)
                    z_pref = torch.ones(
                        (B, self.num_heads, self.head_dim),
                        device=out_pref.device,
                        dtype=out_pref.dtype,
                    ) * float(P)

                tail_len = N - P
                if tail_len == 0:
                    out = out_pref
                    state_out = (S_pref, z_pref)
                else:
                    phi_q_num_tail = phi_q_num[:, :, P:]
                    phi_q_den_tail = phi_q_den[:, :, P:]
                    phi_k_num_tail = phi_k_num[:, :, P:]
                    phi_k_den_tail = phi_k_den[:, :, P:]
                    v_tail = v[:, :, P:]

                    state_pair = (S_pref, z_pref)
                    out_tail, state_out = _causal_step(
                        phi_q_num_tail,
                        phi_q_den_tail,
                        phi_k_num_tail,
                        phi_k_den_tail,
                        v_tail,
                        state_pair,
                    )
                    out = torch.cat([out_pref, out_tail], dim=2)
            elif chunk_causal > 0 and N > chunk_causal:
                outputs = []
                state_pair = (S_prev, z_prev) if (S_prev is not None or z_prev is not None) else None
                for start in range(0, N, chunk_causal):
                    end = min(start + chunk_causal, N)
                    res, state_pair = _causal_step(
                        phi_q_num[:, :, start:end],
                        phi_q_den[:, :, start:end],
                        phi_k_num[:, :, start:end],
                        phi_k_den[:, :, start:end],
                        v[:, :, start:end],
                        state_pair,
                    )
                    outputs.append(res)
                out = torch.cat(outputs, dim=2)
                state_out = state_pair
            else:
                state_pair = (S_prev, z_prev) if (S_prev is not None or z_prev is not None) else None
                out, state_out = _causal_step(
                    phi_q_num,
                    phi_q_den,
                    phi_k_num,
                    phi_k_den,
                    v,
                    state_pair,
                )

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(self.dropout(out))

        if return_state:
            if not self.is_causal:
                raise ValueError("return_state is only supported for causal LinearReluAttention")
            if state_out is None:
                raise RuntimeError("Causal state not computed; ensure causal_state is enabled")
            return out, state_out
        return out

class TransformerEncoderBlock_proposed(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float, linear_attn=True
    ):
        super(TransformerEncoderBlock_proposed, self).__init__()

        # self.attention = nn.MultiheadAttention(
        #     embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        # )
        if linear_attn:
            self.attention = LinearReluAttention(dim=d_model, num_heads=num_heads, is_causal=True, dropout=dropout)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )

        self.linear_attn = linear_attn

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, num_features):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        if self.linear_attn:
            attn_output = self.attention(
                x
            )
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)

        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output

class CausalTransformer_proposed(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer_proposed, self).__init__()

        layers = [
                TransformerEncoderBlock_proposed(d_model, num_heads, feedforward_dim, dropout, linear_attn=True)
                for _ in range(num_layers-1)
        ]
        layers.append(TransformerEncoderBlock_proposed(d_model, num_heads, feedforward_dim, dropout, linear_attn=False))

        self.layers = nn.ModuleList(
            layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True, num_features=1):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask, num_features)

        x = self.norm(x)
        return x


class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask, src_mask):
        """
        :param query: [batch_size * num_features, seq_len, d_model]
        :param key: [batch_size * num_features, seq_len, d_model]
        :param value: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))

        # Encoder attention
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        x = self.norm3(query + self.dropout(ff_output))

        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        mask_ratio: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.mask_ratio = mask_ratio

    def forward(self, query, key, value, is_tgt_mask=True, is_src_mask=True):
        seq_len = query.size(1)
        tgt_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            query = layer(query, key, value, tgt_mask, src_mask)
        x = self.norm(query)
        return x


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(self.receptive_field - 1),  # 左填充
            dilation=dilation,
            groups=groups
        )
        
    def forward(self, x):
        out = self.conv(x)
        # 裁剪掉多余的未来时间步，确保与输入长度一致
        return out[:, :, :x.size(2)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class CausalTCN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        
        # First linear layer to map input_dims to hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        # Create a dilated causal convolutional encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        # Input x is of shape [batch_size, seq_len, input_dims]
        
        # Flatten input (batch_size, seq_len, input_dims) -> (batch_size, seq_len, hidden_dims)
        x = self.input_fc(x)
        
        # Transpose for the convolution (batch_size, seq_len, hidden_dims) -> (batch_size, hidden_dims, seq_len)
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions
        x = self.feature_extractor(x)  # [batch_size, hidden_dims, seq_len] -> [batch_size, output_dims, seq_len]
        
        # Transpose back to [batch_size, seq_len, output_dims]
        x = x.transpose(1, 2)
        
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):   
        """
        :param x: [batch_size, seq_len, input_dims]
        :return: [batch_size, seq_len, output_dims]
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(ClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * d_model, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class OldClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


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


class ClsFlattenHead(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, num_features, dropout):
        super(ClsFlattenHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # [batch_size, seq_len * d_model]
        x = self.dropout(x)  # [batch_size, seq_len * d_model]
        x = self.forecast_head(x)  # [batch_size, pred_len * num_features]
        return x.reshape(x.size(0), self.pred_len, self.num_features)


class ARFlattenHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        dropout: float,
    ):
        super(ARFlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, seq_len * patch_len, num_features]
        """
        x = self.forecast_head(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.dropout(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.flatten(x)  # (batch_size, num_features, seq_len * patch_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len * patch_len, num_features)
        return x