import torch
import torch.nn as nn

from arctandiff_model_diffusion_only import ArcTanDiffusion


class ClsHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_norm = getattr(args, "use_norm", 1)
        self.input_len = getattr(args, "input_len", getattr(args, "seq_len", 336))
        self.pred_len = getattr(args, "pred_len", 96)
        self.include_cls = getattr(args, "include_cls", 0)

        patch_size = getattr(args, "patch_len", getattr(args, "patch_size", 8))
        hidden_dim = getattr(args, "d_model", getattr(args, "hidden_dim", 128))
        num_heads = getattr(args, "n_heads", getattr(args, "num_heads", 16))
        n_layers = getattr(args, "e_layers", getattr(args, "n_layers", 2))
        recon_head_depth = getattr(args, "d_layers", getattr(args, "recon_head_depth", 2))
        dropout = getattr(args, "dropout", 0.1)
        head_dropout = getattr(args, "head_dropout", 0.1)
        model_type = getattr(args, "model_type", "all_normalized")

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.backbone = ArcTanDiffusion(
            in_dim=patch_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_layers=n_layers,
            patch_size=patch_size,
            downstream_task="forecasting",
            model_type=model_type,
            is_causal=getattr(args, "is_causal", False),
            recon_head_depth=recon_head_depth,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(
                (self.input_len // self.patch_size + self.include_cls) * self.hidden_dim,
                self.pred_len,
            ),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        bsz, _, n_vars = x.shape
        if self.use_norm:
            center = x.mean(dim=1, keepdim=True).detach()
            std = x.std(dim=1, keepdim=True).detach()
            x = (x - center) / (std + 1e-5)
        else:
            center = torch.zeros(bsz, 1, n_vars, device=x.device, dtype=x.dtype)
            std = torch.ones(bsz, 1, n_vars, device=x.device, dtype=x.dtype)

        t = kwargs.pop("t", None)
        if t is None:
            t = getattr(self, "test_t", 1.0)
        features = self.backbone.clean_forward(x, t=t)
        flat = features.reshape(features.shape[0], -1)
        preds = self.head(flat)

        center = center.permute(0, 2, 1).reshape(-1, 1)
        std = std.permute(0, 2, 1).reshape(-1, 1)
        preds = preds * (std + 1e-5) + center
        preds = preds.reshape(bsz, n_vars, -1).permute(0, 2, 1)
        return preds


class ClsModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_norm = args.use_norm
        self.backbone = ArcTanDiffusion(
            in_dim=args.enc_in,
            hidden_dim=args.d_model,
            num_heads=args.n_heads,
            n_layers=args.e_layers,
            patch_size=args.patch_len,
            is_causal=getattr(args, "is_causal", False),
            downstream_task="classification",
            recon_head_depth=args.d_layers,
            dropout=args.dropout,
        )
        self.head = ClsHead(
            d_model=args.d_model,
            num_classes=args.num_classes,
            dropout=args.head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std

        t = getattr(self, "test_t", 1.0)
        features = self.backbone.clean_forward(x, t=t)
        return self.head(features)
