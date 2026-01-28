import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cps_robustness.data_module import CPSRobustnessData
from data_provider.data_factory import data_provider
from cost.cost import CoST as CoSTModel
from cost.tasks._eval_protocols import fit_ridge
from cost_train import prepare_split


def ensure_attr(args, name, default):
    if not hasattr(args, name) or getattr(args, name) is None:
        setattr(args, name, default)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("args", None)
    return ckpt, ckpt.get("args", None) if isinstance(ckpt, dict) else None


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def build_model(args):
    model_name = args.model
    task = args.downstream_task

    if model_name == "CoST":
        raise ValueError("CoST should be initialized via build_cost_wrapper().")

    if model_name in ("TimeDART", "TimeDART_proposed", "TimeDART_v2"):
        from models import TimeDART, TimeDART_proposed, TimeDART_v2
        mod = {
            "TimeDART": TimeDART,
            "TimeDART_proposed": TimeDART_proposed,
            "TimeDART_v2": TimeDART_v2,
        }[model_name]
        if task == "classification":
            model = mod.ClsModel(args).float()
        else:
            model = mod.Model(args).float()
    elif model_name == "ArcTanDiffusion":
        from models import ArcTanDiffusion
        if task == "classification":
            model = ArcTanDiffusion.ClsModel(args).float()
        else:
            model = ArcTanDiffusion.Model(args).float()
    elif model_name == "SimMTM":
        from models import SimMTM
        if task == "classification":
            raise ValueError("SimMTM classification is not supported in this repo.")
        model = SimMTM.Model(args).float()
    elif model_name == "PatchTST":
        from models import PatchTST
        if task == "classification":
            args.task_name = "classification"
        else:
            args.task_name = "long_term_forecast"
        model = PatchTST.Model(args, patch_len=args.patch_len, stride=args.stride).float()
    elif model_name == "DLinear":
        from models import DLinear
        if task == "classification":
            raise ValueError("DLinear classification is not supported.")
        model = DLinear.Model(args).float()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class CoSTForecastWrapper:
    def __init__(
        self,
        cost_model,
        ridge,
        pred_len,
        channel_independent,
        add_time_features,
        instance_norm,
        time_scaler,
        eval_batch_size,
        max_train_length,
    ):
        self.cost_model = cost_model
        self.ridge = ridge
        self.pred_len = pred_len
        self.channel_independent = channel_independent
        self.add_time_features = add_time_features
        self.instance_norm = instance_norm
        self.time_scaler = time_scaler
        self.eval_batch_size = eval_batch_size
        self.max_train_length = max_train_length

    def to(self, device):
        self.cost_model.device = device
        self.cost_model.net.to(device)
        return self

    def eval(self):
        self.cost_model.net.eval()
        return self

    def _instance_norm(self, values, eps=1e-5):
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return (values - mean) / std

    def _scale_time_features(self, stamp):
        if self.time_scaler is None:
            return stamp
        n_cov = stamp.shape[-1]
        flat = stamp.reshape(-1, n_cov)
        scaled = self.time_scaler.transform(flat).reshape(stamp.shape)
        return scaled

    def __call__(self, batch_x, batch_x_mark=None):
        x = batch_x.detach().cpu().numpy()
        stamp = batch_x_mark.detach().cpu().numpy() if batch_x_mark is not None else None

        if self.channel_independent:
            b, t, c = x.shape
            values = x.transpose(0, 2, 1).reshape(b * c, t, 1)
            if self.instance_norm:
                values = self._instance_norm(values)
            if self.add_time_features:
                if stamp is None:
                    raise ValueError("CoST expects time features but none were provided.")
                stamp = self._scale_time_features(stamp)
                stamp_ci = np.repeat(stamp, repeats=c, axis=0)
                encoder_input = np.concatenate([stamp_ci, values], axis=-1)
            else:
                encoder_input = values
            n_channels = c
            batch_size = b
        else:
            values = x
            if self.instance_norm:
                values = self._instance_norm(values)
            if self.add_time_features:
                if stamp is None:
                    raise ValueError("CoST expects time features but none were provided.")
                stamp = self._scale_time_features(stamp)
                encoder_input = np.concatenate([stamp, values], axis=-1)
            else:
                encoder_input = values
            n_channels = values.shape[2]
            batch_size = values.shape[0]

        if encoder_input.shape[1] < self.max_train_length:
            pad_len = self.max_train_length - encoder_input.shape[1]
            pad_values = np.full(
                (encoder_input.shape[0], pad_len, encoder_input.shape[2]),
                np.nan,
                dtype=encoder_input.dtype,
            )
            encoder_input = np.concatenate([pad_values, encoder_input], axis=1)

        reprs = self.cost_model.encode(
            encoder_input,
            mode="forecasting",
            casual=True,
            sliding_length=None,
            sliding_padding=0,
            batch_size=self.eval_batch_size,
        )
        last_repr = reprs[:, -1, :]
        pred = self.ridge.predict(last_repr)

        if self.channel_independent:
            pred = pred.reshape(batch_size, n_channels, self.pred_len)
            pred = np.transpose(pred, (0, 2, 1))
        else:
            pred = pred.reshape(batch_size, self.pred_len, n_channels)

        return torch.from_numpy(pred).to(batch_x.device, dtype=batch_x.dtype)


def fit_cost_ridge(args, cost_model):
    train_ds, train_loader = data_provider(args, "train")
    val_ds, val_loader = data_provider(args, "val")

    time_scaler = None
    if args.cost_add_time_features:
        _, _, time_scaler, _ = prepare_split(
            train_ds,
            add_time_features=True,
            channel_independent=False,
            instance_norm=False,
            time_scaler=None,
            fit_time_scaler=True,
        )

    eval_bs = args.cost_eval_batch_size or args.batch_size
    max_len = args.cost_max_train_length

    def instance_norm(values, eps=1e-5):
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return (values - mean) / std

    def scale_stamp(stamp):
        if time_scaler is None:
            return stamp
        n_cov = stamp.shape[-1]
        flat = stamp.reshape(-1, n_cov)
        scaled = time_scaler.transform(flat).reshape(stamp.shape)
        return scaled

    def pad_to_max_len(arr, max_len):
        if arr.shape[1] >= max_len:
            return arr
        pad_len = max_len - arr.shape[1]
        pad_vals = np.full(
            (arr.shape[0], pad_len, arr.shape[2]),
            np.nan,
            dtype=arr.dtype,
        )
        return np.concatenate([pad_vals, arr], axis=1)

    def collect(loader, max_samples=None):
        features = []
        labels = []
        total = 0
        for batch_x, batch_y, batch_x_mark, _ in loader:
            x = batch_x.detach().cpu().numpy()
            y = batch_y.detach().cpu().numpy()
            stamp = batch_x_mark.detach().cpu().numpy() if args.cost_add_time_features else None

            if args.cost_channel_independent:
                b, t, c = x.shape
                values = x.transpose(0, 2, 1).reshape(b * c, t, 1)
                if args.cost_instance_norm:
                    values = instance_norm(values)
                if args.cost_add_time_features:
                    stamp = scale_stamp(stamp)
                    stamp = np.repeat(stamp, repeats=c, axis=0)
                    encoder_input = np.concatenate([stamp, values], axis=-1)
                else:
                    encoder_input = values
                y = y[:, -args.pred_len :, :].transpose(0, 2, 1).reshape(b * c, args.pred_len)
            else:
                values = x
                if args.cost_instance_norm:
                    values = instance_norm(values)
                if args.cost_add_time_features:
                    stamp = scale_stamp(stamp)
                    encoder_input = np.concatenate([stamp, values], axis=-1)
                else:
                    encoder_input = values
                y = y[:, -args.pred_len :, :].reshape(values.shape[0], -1)

            encoder_input = pad_to_max_len(encoder_input, max_len)
            reprs = cost_model.encode(
                encoder_input,
                mode="forecasting",
                casual=True,
                sliding_length=None,
                sliding_padding=0,
                batch_size=eval_bs,
            )
            last_repr = reprs[:, -1, :]

            features.append(last_repr)
            labels.append(y)
            total += last_repr.shape[0]
            if max_samples is not None and total >= max_samples:
                break

        feats = np.concatenate(features, axis=0)
        labs = np.concatenate(labels, axis=0)
        if max_samples is not None and feats.shape[0] > max_samples:
            feats = feats[:max_samples]
            labs = labs[:max_samples]
        return feats, labs

    train_features, train_labels = collect(train_loader, args.cost_ridge_max_samples)
    val_features, val_labels = collect(val_loader, args.cost_ridge_max_samples)
    ridge = fit_ridge(train_features, train_labels, val_features, val_labels)

    return ridge, time_scaler

def forward_model(model, batch_x, batch_x_mark, batch_y, batch_y_mark, args):
    model_name = args.model
    task = args.downstream_task

    if task == "classification":
        if model_name == "PatchTST":
            return model(batch_x, batch_x_mark, batch_x, batch_x_mark)
        if model_name == "ArcTanDiffusion":
            return model(batch_x)
        return model(batch_x)

    if model_name == "CoST":
        return model(batch_x, batch_x_mark)
    if model_name in ("TimeDART", "TimeDART_proposed", "TimeDART_v2", "DLinear", "ArcTanDiffusion"):
        return model(batch_x)
    if model_name == "SimMTM":
        return model(batch_x, batch_x_mark)
    if model_name == "PatchTST":
        return model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    raise ValueError(f"Unhandled model for forward: {model_name}")


def _update_metric_sums(metric, preds, trues, total, count, eps=1e-6):
    diff = preds - trues
    if metric == "MSE":
        total += (diff ** 2).sum().item()
        count += diff.numel()
    elif metric == "MAE":
        total += diff.abs().sum().item()
        count += diff.numel()
    elif metric == "MAPE":
        total += (diff.abs() / (trues.abs() + eps)).sum().item()
        count += diff.numel()
    elif metric == "SMAPE":
        total += (2.0 * diff.abs() / (preds.abs() + trues.abs() + eps)).sum().item()
        count += diff.numel()
    else:
        raise ValueError(f"Unsupported test_metric '{metric}'.")
    return total, count


def eval_forecast(model, dataloader, device, args, target_idx=None, progress_label="eval"):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        loader = dataloader
        if getattr(args, "progress", 1):
            loader = tqdm(dataloader, desc=progress_label, leave=False)
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                batch_x, batch_y, batch_x_mark = batch
            else:
                batch_x, batch_y = batch
                batch_x_mark = None

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is None:
                batch_x_mark = torch.zeros(
                    (batch_x.shape[0], batch_x.shape[1], 1), device=device, dtype=batch_x.dtype
                )
            else:
                batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = torch.zeros(
                (batch_y.shape[0], batch_y.shape[1], 1), device=device, dtype=batch_y.dtype
            )

            outputs = forward_model(model, batch_x, batch_x_mark, batch_y, batch_y_mark, args)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            outputs = outputs[:, -args.pred_len :, :]
            batch_y = batch_y[:, -args.pred_len :, :]

            if target_idx is not None:
                outputs = outputs[:, :, target_idx : target_idx + 1]
                batch_y = batch_y[:, :, target_idx : target_idx + 1]

            total, count = _update_metric_sums(
                args.test_metric, outputs, batch_y, total, count
            )
    return total / max(1, count)


def _ensure_3d_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x


def eval_classification(model, dataloader, device, args, corrupt_fn=None, progress_label="eval"):
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        loader = dataloader
        if getattr(args, "progress", 1):
            loader = tqdm(dataloader, desc=progress_label, leave=False)
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                batch_x, batch_y, batch_x_mark, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_x, batch_y = batch
                batch_x_mark = None
            else:
                batch_x, batch_y = batch
                batch_x_mark = None

            batch_x = batch_x.float()
            batch_x = _ensure_3d_tensor(batch_x)
            if corrupt_fn is not None:
                batch_x = corrupt_fn(batch_x)

            batch_y = batch_y.long()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if batch_x_mark is None:
                batch_x_mark = torch.zeros(
                    (batch_x.shape[0], batch_x.shape[1], 1), device=device, dtype=batch_x.dtype
                )
            else:
                batch_x_mark = batch_x_mark.float().to(device)

            outputs = forward_model(model, batch_x, batch_x_mark, batch_y, None, args)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if outputs.dim() == 1:
                preds = (outputs > 0).long()
            else:
                num_classes = getattr(args, "num_classes", None)
                if outputs.dim() > 2 and num_classes is not None and outputs.shape[1] == num_classes:
                    class_dim = 1
                else:
                    class_dim = -1
                preds = outputs.argmax(dim=class_dim)

            target = batch_y
            if target.dim() > preds.dim():
                target = target.squeeze()
            target = target.view(-1)
            preds = preds.view(-1)
            if preds.numel() != target.numel():
                raise ValueError(
                    f"Predictions/targets size mismatch for accuracy: {preds.shape} vs {target.shape}"
                )

            correct += (preds == target).sum().item()
            count += target.numel()
    return correct / max(1, count)


def _select_affected_sensors(gen, n_features, prct_affected):
    if n_features <= 0:
        return []
    min_prct = 1.0 / n_features + 1e-9
    prct = max(min_prct, min(1.0, prct_affected))
    n_affected = int(n_features * prct)
    n_affected = max(1, min(n_features, n_affected))
    perm = torch.randperm(n_features, generator=gen)
    return perm[:n_affected].tolist()


def _safe_std(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return torch.where(std < eps, torch.ones_like(std), std)


def _apply_drift(x, severity, affected, gen):
    if severity <= 0 or not affected:
        return x
    x = x.clone()
    std = _safe_std(x)
    x[:, :, affected] += severity * std[:, :, affected]
    return x


def _apply_dying_signal(x, severity, affected, gen):
    if severity <= 0 or not affected:
        return x
    x = x.clone()
    factor = 1.0 - severity
    x[:, :, affected] *= factor
    return x


def _apply_noise(x, severity, affected, gen):
    if severity <= 0 or not affected:
        return x
    x = x.clone()
    std = _safe_std(x)
    noise = torch.randn(x.shape, generator=gen, dtype=x.dtype) * std * severity
    mask = torch.zeros((1, 1, x.shape[2]), dtype=x.dtype)
    mask[:, :, affected] = 1.0
    return x + noise * mask


def _apply_flat_sensor(x, severity, affected, gen):
    if not affected:
        return x
    x = x.clone()
    bsz, seq_len, _ = x.shape
    flat_duration = int(1 + severity * max(0, seq_len - 1))
    if flat_duration <= 0:
        return x
    start_low = 1
    start_high = max(start_low, seq_len - flat_duration + 1)
    starts = torch.randint(start_low, start_high + 1, (bsz, len(affected)), generator=gen)
    for b in range(bsz):
        for idx, sensor in enumerate(affected):
            start = int(starts[b, idx].item())
            end = min(seq_len, start + flat_duration)
            if start - 1 < 0:
                continue
            last_val = x[b, start - 1, sensor]
            x[b, start:end, sensor] = last_val
    return x


def _apply_missing_block(x, severity, gen):
    x = x.clone()
    bsz, seq_len, _ = x.shape
    max_duration = max(1, int(seq_len * 0.5))
    missing_duration = int(1 + severity * max(0, max_duration - 1))
    if missing_duration <= 0:
        return x
    max_start = max(0, seq_len - missing_duration)
    if max_start == 0:
        starts = torch.zeros((bsz,), dtype=torch.long)
    else:
        starts = torch.randint(0, max_start + 1, (bsz,), generator=gen)
    for b in range(bsz):
        start = int(starts[b].item())
        end = min(seq_len, start + missing_duration)
        x[b, start:end, :] = 0.0
    return x


def _apply_outlier(x, severity, affected, gen):
    if severity <= 0 or not affected:
        return x
    x = x.clone()
    bsz, seq_len, _ = x.shape
    std = _safe_std(x)
    hickup_value = 1.0 + severity * (25.0 - 1.0)
    positions = torch.randint(0, seq_len, (bsz, len(affected)), generator=gen)
    for b in range(bsz):
        for idx, sensor in enumerate(affected):
            pos = int(positions[b, idx].item())
            x[b, pos, sensor] += hickup_value * std[b, 0, sensor]
    return x


def _apply_time_warp(x, severity, affected, gen, warp_min, warp_max):
    if severity <= 0 or not affected:
        return x
    x_np = x.detach().cpu().numpy().copy()
    bsz, seq_len, _ = x_np.shape
    warp_duration = int(seq_len * 0.5)
    if warp_duration <= 0:
        return x
    warp_factor = warp_min + severity * (warp_max - warp_min)
    max_start = max(1, int(seq_len * 0.5))
    start_positions = torch.randint(0, max_start, (bsz,), generator=gen).numpy()
    orig_time = np.arange(seq_len, dtype=np.float32)
    irreg_time = np.full(warp_duration, warp_factor, dtype=np.float32)
    for b in range(bsz):
        start = int(start_positions[b])
        irreg_time_index = np.cumsum(irreg_time) + start - 1
        irreg_time_index = np.clip(irreg_time_index, 0, seq_len - 1)
        for sensor in affected:
            x_np[b, start:start + warp_duration, sensor] = np.interp(
                irreg_time_index, orig_time, x_np[b, :, sensor]
            )
    return torch.from_numpy(x_np)


def _make_classification_corrupt_fn(scenario, severity, args, n_features, seed):
    gen = torch.Generator().manual_seed(seed)
    if scenario in ("flat_sensor", "faster_sampling", "slower_sampling"):
        prct = min(args.prct_affected_sensors * 5, 1.0)
    else:
        prct = args.prct_affected_sensors
    affected = _select_affected_sensors(gen, n_features, prct) if scenario != "missing_data" else []

    def corrupt(x):
        x = _ensure_3d_tensor(x)
        if scenario == "drift":
            return _apply_drift(x, severity, affected, gen)
        if scenario == "dying_signal":
            return _apply_dying_signal(x, severity, affected, gen)
        if scenario == "noise":
            return _apply_noise(x, severity, affected, gen)
        if scenario == "flat_sensor":
            return _apply_flat_sensor(x, severity, affected, gen)
        if scenario == "missing_data":
            return _apply_missing_block(x, severity, gen)
        if scenario == "faster_sampling":
            return _apply_time_warp(x, severity, affected, gen, warp_min=1.0, warp_max=5.0)
        if scenario == "slower_sampling":
            return _apply_time_warp(x, severity, affected, gen, warp_min=1.0, warp_max=0.0)
        if scenario == "outlier":
            return _apply_outlier(x, severity, affected, gen)
        return x

    return corrupt


def _infer_classification_shape(dataset):
    sample = dataset[0][0] if isinstance(dataset[0], (list, tuple)) else dataset[0]
    if torch.is_tensor(sample):
        shape = sample.shape
    else:
        shape = np.asarray(sample).shape
    if len(shape) == 1:
        return shape[0], 1
    if len(shape) == 2:
        return shape[0], shape[1]
    return shape[-2], shape[-1]


def _build_classification_test_loader(args):
    test_ds, _ = data_provider(args, "test")
    n_samples = None if args.n_test_samples <= 0 else args.n_test_samples
    if n_samples is not None and n_samples < len(test_ds):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(test_ds), size=n_samples, replace=False)
        test_ds = Subset(test_ds, indices)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return test_ds, test_loader


def _resolve_csv_path(args):
    csv_path = args.data_path
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        csv_path = os.path.join(args.root_path, args.data_path)
    return csv_path


def _is_csv_like(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz") or lower.endswith(".tsv") or lower.endswith(".txt")


def _load_csv_dates(args):
    csv_path = _resolve_csv_path(args)
    if not os.path.exists(csv_path):
        return None
    if not _is_csv_like(csv_path):
        return None
    try:
        df_head = pd.read_csv(csv_path, nrows=1)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError):
        return None
    cols = list(df_head.columns)
    if not cols:
        return None
    date_col = cols[0]
    try:
        pd.to_datetime(df_head[date_col])
    except (ValueError, TypeError):
        return None
    df_dates = pd.read_csv(csv_path, usecols=[date_col])
    return pd.to_datetime(df_dates[date_col])


def _infer_feature_names_from_csv(args):
    csv_path = _resolve_csv_path(args)
    if not os.path.exists(csv_path):
        return None
    if not _is_csv_like(csv_path):
        return None
    try:
        df_head = pd.read_csv(csv_path, nrows=1)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError):
        return None
    cols = list(df_head.columns)
    if not cols:
        return None

    date_col = None
    if cols[0].lower() in ("date", "timestamp", "time"):
        date_col = cols[0]
    if date_col is None:
        try:
            pd.to_datetime(df_head[cols[0]])
            date_col = cols[0]
        except (ValueError, TypeError):
            date_col = None

    if date_col is not None:
        cols = [c for c in cols if c != date_col]

    if args.features == "S":
        if args.target in cols:
            return [args.target]
        return [cols[-1]]

    if args.target in cols:
        cols = [c for c in cols if c != args.target] + [args.target]
    return cols


def _build_data_provider_frames(args, include_timestamps=False):
    train_ds, _ = data_provider(args, "train")
    val_ds, _ = data_provider(args, "val")
    test_ds, _ = data_provider(args, "test")

    feature_names = _infer_feature_names_from_csv(args)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(train_ds.data_x.shape[1])]

    train_df = pd.DataFrame(train_ds.data_x, columns=feature_names)
    test_df = pd.DataFrame(test_ds.data_x, columns=feature_names)
    if include_timestamps:
        dates = _load_csv_dates(args)
        if dates is None:
            raise ValueError("Could not infer timestamp column for time features.")
        train_len = train_ds.data_x.shape[0]
        val_len = val_ds.data_x.shape[0]
        test_len = test_ds.data_x.shape[0]
        total_len = train_len + val_len + test_len
        if len(dates) < total_len:
            raise ValueError("CSV timestamp length is shorter than data_provider splits.")
        train_df.index = dates.iloc[:train_len].values
        test_start = train_len + val_len
        test_df.index = dates.iloc[test_start : test_start + test_len].values
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="CPS robustness benchmark (ported) for time series models")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--downstream_task", type=str, default="forecast", choices=["forecast", "classification"])
    parser.add_argument("--use_ckpt_args", action="store_true", help="load args from checkpoint when available")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)

    # data
    parser.add_argument("--data", type=str, default=None, help="dataset name for logging")
    parser.add_argument("--root_path", type=str, default="./datasets")
    parser.add_argument("--data_path", type=str, required=True, help="path to csv/parquet file")
    parser.add_argument("--features", type=str, default="M", choices=["M", "MS", "S"])
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    # sequence
    parser.add_argument("--input_len", type=int, default=336)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)

    # model hyperparams (fallbacks when ckpt args are absent)
    parser.add_argument(
        "--is_causal", type=bool, default=False, help="whether the model is causal"
    )
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument("--model_type", type=str, default="all_normalized")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--include_cls", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument(
        "--auto_use_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "auto-adjust use_norm for ArcTanDiffusion classification checkpoints that "
            "don't store a use_norm setting"
        ),
    )

    # CoST options
    parser.add_argument(
        "--cost_add_time_features",
        action="store_true",
        help="Append time covariates to CoST inputs (matches cost_train --add_time_features)",
    )
    parser.add_argument(
        "--cost_channel_independent",
        action="store_true",
        help="Treat each channel as an independent series for CoST",
    )
    parser.add_argument(
        "--cost_instance_norm",
        action="store_true",
        help="Apply per-series, per-channel normalization over time for CoST inputs",
    )
    parser.add_argument(
        "--cost_repr_dims",
        type=int,
        default=320,
        help="CoST representation dimension",
    )
    parser.add_argument(
        "--cost_hidden_dims",
        type=int,
        default=64,
        help="CoST hidden dimension",
    )
    parser.add_argument(
        "--cost_depth",
        type=int,
        default=10,
        help="CoST TCN depth",
    )
    parser.add_argument(
        "--cost_kernels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="CoST kernel sizes for temporal decomposition",
    )
    parser.add_argument(
        "--cost_alpha",
        type=float,
        default=0.0005,
        help="CoST seasonal loss weight",
    )
    parser.add_argument(
        "--cost_max_train_length",
        type=int,
        default=3000,
        help="CoST max training length (affects encoding padding)",
    )
    parser.add_argument(
        "--cost_eval_batch_size",
        type=int,
        default=None,
        help="Batch size for CoST encoding during ridge fit/inference (defaults to batch_size)",
    )
    parser.add_argument(
        "--cost_ridge_max_samples",
        type=int,
        default=100000,
        help="Maximum number of ridge samples collected from train/val windows",
    )

    # benchmark config
    parser.add_argument("--n_train_samples", type=int, default=None)
    parser.add_argument("--n_val_samples", type=int, default=None)
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=0,
        help="number of test samples per scenario (0 = all samples from test split)",
    )
    parser.add_argument("--n_severity_levels", type=int, default=101)
    parser.add_argument("--prct_affected_sensors", type=float, default=0.1)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--purged_fraction", type=float, default=0.01)
    parser.add_argument(
        "--test_metric",
        type=str,
        default="MSE",
        choices=["MSE", "MAE", "MAPE", "SMAPE", "CE", "ACC"],
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=1,
        help="show tqdm progress bars for each scenario (1=on, 0=off)",
    )
    parser.add_argument(
        "--eval_all_variates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "evaluate robustness on all variates for multivariate features "
            "(default: True for M/MS, False for S)"
        ),
    )

    parser.add_argument("--out_dir", type=str, default="./outputs/robustness_cps")
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="",
        help="prefix for output files (default: <model>_<dataset>_)",
    )
    parser.add_argument(
        "--use_data_provider",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use data_provider splits/scaling to avoid train/test leakage",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_state = None
    ckpt_args = None
    if args.model != "CoST":
        ckpt_state, ckpt_args = load_checkpoint(args.checkpoint, device=device)
        ckpt_state = _strip_module_prefix(ckpt_state)
        if args.use_ckpt_args and ckpt_args is not None:
            for k, v in ckpt_args.items():
                if hasattr(args, k) and getattr(args, k) is None:
                    setattr(args, k, v)
    else:
        ckpt_state, _ = load_checkpoint(args.checkpoint, device=device)
        ckpt_state = _strip_module_prefix(ckpt_state)

    ensure_attr(args, "seq_len", args.input_len)
    ensure_attr(args, "task_name", "finetune")
    ensure_attr(args, "downstream_task", args.downstream_task)
    args.use_gpu = torch.cuda.is_available()
    args.device = device

    if args.downstream_task == "classification" and args.model == "ArcTanDiffusion":
        if args.auto_use_norm and (ckpt_args is None or "use_norm" not in ckpt_args):
            if args.use_norm != 0:
                print(
                    "Auto-setting use_norm=0 to match ArcTanDiffusion linear-probe training "
                    "(inputs were not normalized)."
                )
            args.use_norm = 0

    if args.model == "CoST":
        if not isinstance(ckpt_state, dict):
            raise ValueError("CoST checkpoint must be a state_dict.")
        inferred_input_dims = None
        if "input_fc.weight" in ckpt_state:
            inferred_input_dims = ckpt_state["input_fc.weight"].shape[1]
            args.cost_hidden_dims = ckpt_state["input_fc.weight"].shape[0]
        for key, value in ckpt_state.items():
            if key.endswith("projector.weight") and value.ndim >= 2:
                args.cost_repr_dims = value.shape[0]
                break
        net_indices = []
        for key in ckpt_state.keys():
            if key.startswith("feature_extractor.net."):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    net_indices.append(int(parts[2]))
        if net_indices:
            args.cost_depth = max(net_indices)

        input_dims = inferred_input_dims
        if input_dims is None:
            input_dims = args.enc_in

        cost_model = CoSTModel(
            input_dims=input_dims,
            kernels=args.cost_kernels,
            alpha=args.cost_alpha,
            max_train_length=args.cost_max_train_length,
            output_dims=args.cost_repr_dims,
            hidden_dims=args.cost_hidden_dims,
            depth=args.cost_depth,
            device=device,
            lr=0.001,
            batch_size=args.cost_eval_batch_size or args.batch_size,
        )
        cost_model.net.load_state_dict(ckpt_state, strict=True)

        ridge, time_scaler = fit_cost_ridge(args, cost_model)
        model = CoSTForecastWrapper(
            cost_model=cost_model,
            ridge=ridge,
            pred_len=args.pred_len,
            channel_independent=args.cost_channel_independent,
            add_time_features=args.cost_add_time_features,
            instance_norm=args.cost_instance_norm,
            time_scaler=time_scaler,
            eval_batch_size=args.cost_eval_batch_size or args.batch_size,
            max_train_length=args.cost_max_train_length,
        ).to(device)
    else:
        model = build_model(args).to(device)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
        if missing_keys:
            print(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        if args.downstream_task == "classification" and args.model == "ArcTanDiffusion":
            missing_head = [k for k in missing_keys if k.startswith("head.")]
            missing_backbone = [k for k in missing_keys if k.startswith("backbone.")]
            if missing_head or missing_backbone:
                raise ValueError(
                    "Checkpoint is missing ArcTanDiffusion classification head/backbone weights. "
                    "Use a combined checkpoint that includes both (see linear-probe save_model_dir)."
                )

    dataset_name = args.data
    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    results = []
    if args.downstream_task == "classification":
        if args.test_metric != "ACC":
            print("Overriding test_metric to ACC for classification tasks.")
            args.test_metric = "ACC"
        if args.model == "CoST":
            raise ValueError("CoST classification is not supported in this benchmark runner.")

        test_ds, test_loader = _build_classification_test_loader(args)
        _, n_features = _infer_classification_shape(test_ds)

        classification_scenarios = [
            "drift",
            "dying_signal",
            "noise",
            "flat_sensor",
            "missing_data",
            "faster_sampling",
            "slower_sampling",
            "outlier",
        ]

        results.append({"scenario": "normal", "severity": np.nan, "loss": eval_classification(
            model, test_loader, device, args, corrupt_fn=None, progress_label="normal"
        )})

        for scenario_idx, scenario in enumerate(classification_scenarios, start=1):
            for severity in np.linspace(0, 1, args.n_severity_levels):
                seed = int(args.seed + scenario_idx * 100000 + round(float(severity) * 1e6))
                corrupt_fn = _make_classification_corrupt_fn(
                    scenario, float(severity), args, n_features, seed
                )
                label = f"{scenario} sev={severity:.3f}"
                loss = eval_classification(
                    model, test_loader, device, args, corrupt_fn=corrupt_fn, progress_label=label
                )
                results.append(
                    {
                        "scenario": scenario,
                        "severity": float(severity),
                        "loss": loss,
                    }
                )
    else:
        n_test_samples = None if args.n_test_samples <= 0 else args.n_test_samples
        use_time_features = args.model == "CoST" and args.cost_add_time_features
        time_features_mode = "timeF" if args.embed == "timeF" else "raw"
        if args.use_data_provider:
            train_df, test_df = _build_data_provider_frames(
                args, include_timestamps=use_time_features
            )
            data_module = CPSRobustnessData(
                file_path=args.data_path,
                input_len=args.input_len,
                target_len=args.pred_len,
                stride=1,
                n_train_samples=args.n_train_samples,
                n_val_samples=args.n_val_samples,
                n_test_samples=n_test_samples,
                n_severity_levels=args.n_severity_levels,
                prct_affected_sensors=args.prct_affected_sensors,
                train_split=args.train_split,
                val_split=args.val_split,
                purged_fraction=args.purged_fraction,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                features=args.features,
                target=args.target,
                train_df=train_df,
                val_df=None,
                test_df=test_df,
                already_scaled=True,
                return_time_features=use_time_features,
                time_features_freq=args.freq,
                time_features_mode=time_features_mode,
            )
        else:
            data_module = CPSRobustnessData(
                file_path=args.data_path,
                input_len=args.input_len,
                target_len=args.pred_len,
                stride=1,
                n_train_samples=args.n_train_samples,
                n_val_samples=args.n_val_samples,
                n_test_samples=n_test_samples,
                n_severity_levels=args.n_severity_levels,
                prct_affected_sensors=args.prct_affected_sensors,
                train_split=args.train_split,
                val_split=args.val_split,
                purged_fraction=args.purged_fraction,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                features=args.features,
                target=args.target,
                return_time_features=use_time_features,
                time_features_freq=args.freq,
                time_features_mode=time_features_mode,
            )
        data_module.setup()
        test_loaders, mapping = data_module.build_test_loaders()
        target_idx = data_module.get_target_index()
        if args.eval_all_variates is None:
            args.eval_all_variates = args.features in ("M", "MS")
        if args.eval_all_variates and args.features in ("M", "MS"):
            target_idx = None

        for dl_idx, loader in enumerate(test_loaders):
            scenario = mapping.loc[dl_idx, "scenario"]
            severity = mapping.loc[dl_idx, "severity"]
            label = f"{scenario}" if pd.isna(severity) else f"{scenario} sev={severity:.3f}"
            loss = eval_forecast(
                model, loader, device, args, target_idx=target_idx, progress_label=label
            )
            results.append(
                {
                    "scenario": scenario,
                    "severity": np.nan if pd.isna(severity) else float(severity),
                    "loss": loss,
                }
            )

    result_df = pd.DataFrame(results)
    normal_rows = result_df[result_df.scenario == "normal"]
    if normal_rows.empty:
        raise RuntimeError("No 'normal' scenario found in results.")
    normal_performance = normal_rows["loss"].values[0]
    if args.test_metric == "ACC":
        result_df["rel_perf"] = (1e-6 + result_df["loss"]) / (1e-6 + normal_performance)
    else:
        result_df["rel_perf"] = (1e-6 + normal_performance) / (1e-6 + result_df["loss"])

    rel_perf_df = (
        result_df[result_df.scenario != "normal"][["scenario", "rel_perf"]]
        .groupby("scenario")
        .agg(["mean", "std"])
    )
    rel_perf_df.columns = ["mean", "std"]

    robustness_scores = {
        "normal_performance": float(normal_performance),
        "robustness_score_min": float(rel_perf_df["mean"].min()),
        "robustness_score_mean": float(rel_perf_df["mean"].mean()),
        "robustness_score_prod": float(rel_perf_df["mean"].prod()),
    }

    out_dir = os.path.join(args.out_dir, f"{args.model}_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)
    file_prefix = args.file_prefix.strip() or f"{args.model}_{dataset_name}_"
    if not file_prefix.endswith("_"):
        file_prefix += "_"

    scenario_csv = os.path.join(out_dir, f"{file_prefix}scenario_results_{args.test_metric}.csv")
    result_df.to_csv(scenario_csv, index=False)

    rel_perf_csv = os.path.join(out_dir, f"{file_prefix}scenario_rel_perf_{args.test_metric}.csv")
    rel_perf_df.reset_index().to_csv(rel_perf_csv, index=False)

    scores_json = os.path.join(out_dir, f"{file_prefix}robustness_scores_{args.test_metric}.json")
    with open(scores_json, "w") as f:
        json.dump(robustness_scores, f, indent=2)

    print(f"Normal performance ({args.test_metric}): {normal_performance:.6f}")
    print(
        "Robustness scores:",
        json.dumps(robustness_scores, indent=2),
    )
    print(f"Saved scenario results to {scenario_csv}")


if __name__ == "__main__":
    main()
