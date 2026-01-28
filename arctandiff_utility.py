import torch
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Dict
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

@torch.no_grad()
def channel_discard(feat: torch.Tensor, ch_idx: torch.Tensor) -> torch.Tensor:
    """
    Zero out flagged channels (DiTF 'channel discard'). :contentReference[oaicite:7]{index=7}
    """
    if ch_idx.numel() == 0:
        return feat
    feat = feat.clone()
    feat[..., ch_idx] = 0
    return feat

def arctan_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    *,
    a: float = 1.1,
    k: float = 1.3,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.05,
    reduction: str = "mean",  # "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    ArcTanLoss from:
      Zhang et al., "FreMixer ... with noise insensitive ArcTanLoss" (Scientific Reports, 2025)

    y_hat, y: any matching shape, e.g. [B, T, C] or [B, T] or [T].
    """
    if y_hat.shape != y.shape:
        raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} vs y {y.shape}")

    e = y - y_hat
    ke = k * e

    # alpha * [ -a/(2k) * ln(1 + (k e)^2) + a e * atan(k e) ]
    arctan_term = (-a / (2.0 * k)) * torch.log1p(ke * ke) + a * e * torch.atan(ke)

    # + beta * |e| + gamma * |y_hat|^2  (for real tensors, |y_hat|^2 == y_hat^2)
    loss = alpha * arctan_term + beta * torch.abs(e) + gamma * (y_hat * y_hat)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


class ArcTanLoss(nn.Module):
    def __init__(
        self,
        *,
        a: float = 1.1,
        k: float = 1.3,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.a = a
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return arctan_loss(
            y_hat,
            y,
            a=self.a,
            k=self.k,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            reduction=self.reduction,
        )
