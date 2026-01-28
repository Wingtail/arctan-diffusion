import argparse
import datetime
import os
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data_provider.data_factory import data_provider
from cost.cost import CoST
from cost.tasks._eval_protocols import fit_ridge
from cost.tasks.forecasting import generate_pred_samples, cal_metrics
from cost.utils import init_dl_program, name_with_datetime, pkl_save


def save_checkpoint_callback(save_every=1, unit="epoch"):
    assert unit in ("epoch", "iter")

    def callback(model, loss):
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0:
            model.save(f"{run_dir}/model_{n}.pkl")

    return callback


def apply_patchtst_preset(args):
    if args.patchtst_preset is None:
        return
    if args.patchtst_preset == "paper_base":
        args.encoder_type = "transformer"
        args.depth = 3
        args.transformer_heads = 16
        args.hidden_dims = 128
        args.transformer_ffn_dim = 256
        args.transformer_dropout = 0.2
    elif args.patchtst_preset == "paper_small":
        args.encoder_type = "transformer"
        args.depth = 3
        args.transformer_heads = 4
        args.hidden_dims = 16
        args.transformer_ffn_dim = 128
        args.transformer_dropout = 0.3
    else:
        raise ValueError(f"Unknown PatchTST preset: {args.patchtst_preset}")


DATASET_PATHS = {
    "etth1": ("./datasets/ETT-small", "ETTh1.csv"),
    "etth2": ("./datasets/ETT-small", "ETTh2.csv"),
    "ettm1": ("./datasets/ETT-small", "ETTm1.csv"),
    "ettm2": ("./datasets/ETT-small", "ETTm2.csv"),
    "electricity": ("./datasets/electricity", "electricity.csv"),
    "traffic": ("./datasets/traffic", "traffic.csv"),
    "exchange": ("./datasets/exchange", "exchange.csv"),
    "weather": ("./datasets/weather", "weather.csv"),
    "wth": ("./datasets/weather", "weather.csv"),
}


def resolve_dataset_paths(args):
    if args.root_path and args.data_path:
        return

    key = args.dataset.lower()
    if key in DATASET_PATHS and not args.root_path:
        args.root_path, args.data_path = DATASET_PATHS[key]
        return

    if not args.root_path:
        args.root_path = "./datasets"
    if not args.data_path:
        args.data_path = f"{args.dataset}.csv"


def dataset_to_array(dataset, include_time=False):
    data = getattr(dataset, "data_x", None)
    if data is None:
        raise ValueError("Dataset does not expose data_x for CoST training.")
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if hasattr(data, "values"):
        data = data.values
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim == 2:
        data = data[None, :, :]
    elif data.ndim != 3:
        raise ValueError(f"Expected data_x to be 1D/2D/3D, got shape {data.shape}")
    data = data.astype(np.float32)

    if not include_time:
        return data, None

    stamp = getattr(dataset, "data_stamp", None)
    if stamp is None:
        raise ValueError("Dataset does not expose data_stamp for time features.")
    if torch.is_tensor(stamp):
        stamp = stamp.detach().cpu().numpy()
    if hasattr(stamp, "values"):
        stamp = stamp.values
    stamp = np.asarray(stamp)
    if stamp.ndim == 1:
        stamp = stamp[:, None]
    if stamp.ndim == 2:
        stamp = stamp[None, :, :]
    elif stamp.ndim != 3:
        raise ValueError(f"Expected data_stamp to be 1D/2D/3D, got shape {stamp.shape}")
    stamp = stamp.astype(np.float32)

    return data, stamp


def reshape_channel_independent(values, stamp=None):
    b, t, c = values.shape
    values_ci = values.transpose(0, 2, 1).reshape(b * c, t, 1)
    stamp_ci = None
    if stamp is not None:
        base_stamp = stamp[0] if stamp.shape[0] > 0 else stamp
        stamp_ci = np.repeat(base_stamp[None, :, :], b * c, axis=0)
    return values_ci, stamp_ci


def instance_normalize(values, eps=1e-5):
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (values - mean) / std


def prepare_split(
    dataset,
    add_time_features,
    channel_independent,
    instance_norm,
    time_scaler=None,
    fit_time_scaler=False,
):
    values, stamp = dataset_to_array(dataset, include_time=add_time_features)

    if channel_independent:
        values, stamp = reshape_channel_independent(values, stamp)

    label_values = values

    if add_time_features:
        if stamp is None:
            raise ValueError("Time features requested but data_stamp is missing.")
        n_covariate_cols = stamp.shape[2]
        if fit_time_scaler:
            time_scaler = StandardScaler().fit(stamp.reshape(-1, n_covariate_cols))
        if time_scaler is None:
            raise ValueError("Time scaler is not initialized.")
        stamp = time_scaler.transform(stamp.reshape(-1, n_covariate_cols)).reshape(stamp.shape)
    else:
        n_covariate_cols = 0

    encoder_values = instance_normalize(values) if instance_norm else values

    if add_time_features:
        encoder_input = np.concatenate([stamp, encoder_values], axis=-1)
    else:
        encoder_input = encoder_values

    return encoder_input, label_values, time_scaler, n_covariate_cols


def inverse_transform(scaler, arr):
    if scaler is None:
        return arr
    n_features = getattr(scaler, "n_features_in_", None)
    if (
        arr.ndim == 4
        and arr.shape[-1] == 1
        and arr.shape[0] > 1
        and n_features == arr.shape[0]
    ):
        tmp = arr.swapaxes(0, 3)
        flat = tmp.reshape(-1, tmp.shape[-1])
        inv = scaler.inverse_transform(flat).reshape(tmp.shape)
        return inv.swapaxes(0, 3)
    flat = arr.reshape(-1, arr.shape[-1])
    inv = scaler.inverse_transform(flat)
    return inv.reshape(arr.shape)


def encode_split(model, data, padding, batch_size=256):
    pad = max(0, min(padding, data.shape[1] - 1))
    return model.encode(
        data,
        mode="forecasting",
        casual=True,
        sliding_length=1,
        sliding_padding=pad,
        batch_size=batch_size,
    ), pad


def build_full_series(train_data, valid_data, test_data, overlap):
    overlap = max(0, int(overlap))

    def trim_front(arr):
        if overlap <= 0:
            return arr
        if arr.shape[1] <= overlap:
            return arr[:, 0:0]
        return arr[:, overlap:]

    valid_trim = trim_front(valid_data)
    test_trim = trim_front(test_data)

    full = np.concatenate([train_data, valid_trim, test_trim], axis=1)

    train_end = train_data.shape[1]
    valid_start = train_end
    valid_end = valid_start + valid_trim.shape[1]
    test_start = valid_end
    test_end = test_start + test_trim.shape[1]

    return full, slice(0, train_end), slice(valid_start, valid_end), slice(test_start, test_end)


def build_full_series_pair(
    train_enc,
    valid_enc,
    test_enc,
    train_lbl,
    valid_lbl,
    test_lbl,
    overlap,
):
    full_enc, train_slice, valid_slice, test_slice = build_full_series(
        train_enc, valid_enc, test_enc, overlap
    )
    full_lbl, train_slice_l, valid_slice_l, test_slice_l = build_full_series(
        train_lbl, valid_lbl, test_lbl, overlap
    )
    if (
        train_slice != train_slice_l
        or valid_slice != valid_slice_l
        or test_slice != test_slice_l
    ):
        raise ValueError("Encoder/label splits diverged after overlap trimming.")
    return full_enc, full_lbl, train_slice, valid_slice, test_slice


def eval_linear_probe(
    model,
    train_enc,
    valid_enc,
    test_enc,
    train_lbl,
    valid_lbl,
    test_lbl,
    scaler,
    pred_lens,
    padding,
    eval_batch_size,
):
    t = time.time()
    train_repr, train_pad = encode_split(model, train_enc, padding, batch_size=eval_batch_size)
    valid_repr, _ = encode_split(model, valid_enc, padding, batch_size=eval_batch_size)
    test_repr, _ = encode_split(model, test_enc, padding, batch_size=eval_batch_size)
    encoder_infer_time = time.time() - t

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}

    for pred_len in pred_lens:
        if train_repr.shape[1] <= pred_len + train_pad:
            raise ValueError(
                f"Train split too short for pred_len={pred_len} with padding={train_pad}."
            )

        train_features, train_labels = generate_pred_samples(
            train_repr, train_lbl, pred_len, drop=train_pad
        )
        valid_features, valid_labels = generate_pred_samples(
            valid_repr, valid_lbl, pred_len
        )
        test_features, test_labels = generate_pred_samples(test_repr, test_lbl, pred_len)

        t = time.time()
        lr = fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_lbl.shape[0], -1, pred_len, test_lbl.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        test_pred_inv = inverse_transform(scaler, test_pred)
        test_labels_inv = inverse_transform(scaler, test_labels)

        out_log[pred_len] = {
            "norm": test_pred,
            "raw": test_pred_inv,
            "norm_gt": test_labels,
            "raw_gt": test_labels_inv,
        }
        ours_result[pred_len] = {
            "norm": cal_metrics(test_pred, test_labels),
            "raw": cal_metrics(test_pred_inv, test_labels_inv),
        }

    eval_res = {
        "ours": ours_result,
        "encoder_infer_time": encoder_infer_time,
        "lr_train_time": lr_train_time,
        "lr_infer_time": lr_infer_time,
    }
    return out_log, eval_res


def eval_linear_probe_full(
    model,
    full_enc,
    full_lbl,
    train_slice,
    valid_slice,
    test_slice,
    scaler,
    pred_lens,
    padding,
    eval_batch_size,
):
    t = time.time()
    all_repr, used_pad = encode_split(
        model, full_enc, padding, batch_size=eval_batch_size
    )
    encoder_infer_time = time.time() - t

    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    train_data = full_lbl[:, train_slice]
    valid_data = full_lbl[:, valid_slice]
    test_data = full_lbl[:, test_slice]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}

    for pred_len in pred_lens:
        if train_repr.shape[1] <= pred_len + used_pad:
            raise ValueError(
                f"Train split too short for pred_len={pred_len} with padding={used_pad}."
            )

        train_features, train_labels = generate_pred_samples(
            train_repr, train_data, pred_len, drop=used_pad
        )
        valid_features, valid_labels = generate_pred_samples(
            valid_repr, valid_data, pred_len
        )
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        test_pred_inv = inverse_transform(scaler, test_pred)
        test_labels_inv = inverse_transform(scaler, test_labels)

        out_log[pred_len] = {
            "norm": test_pred,
            "raw": test_pred_inv,
            "norm_gt": test_labels,
            "raw_gt": test_labels_inv,
        }
        ours_result[pred_len] = {
            "norm": cal_metrics(test_pred, test_labels),
            "raw": cal_metrics(test_pred_inv, test_labels_inv),
        }

    eval_res = {
        "ours": ours_result,
        "encoder_infer_time": encoder_infer_time,
        "lr_train_time": lr_train_time,
        "lr_infer_time": lr_infer_time,
    }
    return out_log, eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The dataset name (also used for data_provider)")
    parser.add_argument(
        "run_name",
        help="The folder name used to save model, output and evaluation metrics.",
    )
    parser.add_argument("--data", type=str, default=None, help="dataset key for data_provider")
    parser.add_argument(
        "--downstream_task",
        type=str,
        default="forecast",
        help="downstream task, options:[forecasting, classification]",
    )
    parser.add_argument("--root_path", type=str, default=None, help="Dataset root path")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset file name")
    parser.add_argument("--features", type=str, default="M", help="M, S, or MS")
    parser.add_argument("--target", type=str, default="OT", help="Target feature name")
    parser.add_argument("--freq", type=str, default="h", help="Frequency string")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--seasonal_patterns",
        type=str,
        default="Monthly",
        help="subset for M4",
    )
    parser.add_argument("--seq_len", type=int, default=None, help="alias for input_len")
    parser.add_argument("--input_len", type=int, default=336, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="label length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
    parser.add_argument("--num_workers", type=int, default=5, help="Data loader workers")

    parser.add_argument("--gpu", type=int, default=0, help="GPU index (defaults to 0)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--repr-dims", type=int, default=320, help="Representation dim")
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="Max training sequence length",
    )
    parser.add_argument("--iters", type=int, default=None, help="Number of iterations")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N iterations/epochs",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Max threads for torch/numpy",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run linear-probe evaluation after training",
    )
    parser.add_argument(
        "--add_time_features",
        action="store_true",
        help="Append time covariates (from data_stamp) to inputs, CoST-style",
    )
    parser.add_argument(
        "--instance_norm",
        action="store_true",
        help="Apply per-series, per-channel normalization over time (inputs only)",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="full",
        choices=["full", "split"],
        help="full: encode full series then slice; split: encode each split separately",
    )
    parser.add_argument(
        "--channel_independent",
        action="store_true",
        help="Treat each channel as an independent series (CoST-style for ECL/ETT)",
    )
    parser.add_argument(
        "--pred_lens",
        type=int,
        nargs="+",
        default=None,
        help="Prediction lengths for linear probe (defaults to pred_len)",
    )
    parser.add_argument(
        "--kernels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Kernel sizes for mixture of AR expert layers",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0005,
        help="Weight for seasonal loss",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        default=64,
        help="Hidden dimension in CoST encoder",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Number of dilated conv blocks",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="tcn",
        choices=["tcn", "transformer"],
        help="Encoder backbone type",
    )
    parser.add_argument(
        "--transformer_heads",
        type=int,
        default=4,
        help="Number of attention heads for transformer encoder",
    )
    parser.add_argument(
        "--transformer_ffn_dim",
        type=int,
        default=None,
        help="Transformer feedforward dimension (defaults to 4x hidden)",
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.1,
        help="Transformer dropout probability",
    )
    parser.add_argument(
        "--patchtst_preset",
        type=str,
        default=None,
        choices=["paper_base", "paper_small"],
        help="Apply PatchTST paper hyperparameter sizes (overrides transformer settings)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation encoding (defaults to batch_size)",
    )

    args = parser.parse_args()

    if args.seq_len is not None and args.input_len == 336:
        args.input_len = args.seq_len

    if args.data is None:
        args.data = args.dataset

    apply_patchtst_preset(args)

    resolve_dataset_paths(args)

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    train_dataset, _ = data_provider(args, "train")
    train_enc, train_lbl, time_scaler, _ = prepare_split(
        train_dataset,
        add_time_features=args.add_time_features,
        channel_independent=args.channel_independent,
        instance_norm=args.instance_norm,
        time_scaler=None,
        fit_time_scaler=True,
    )
    scaler = getattr(train_dataset, "scaler", None)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
    )

    if args.save_every is not None:
        unit = "epoch" if args.epochs is not None else "iter"
        config[f"after_{unit}_callback"] = save_checkpoint_callback(args.save_every, unit)

    run_dir = f"training/{args.dataset}/{name_with_datetime(args.run_name)}"
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = CoST(
        input_dims=train_enc.shape[-1],
        kernels=args.kernels,
        alpha=args.alpha,
        max_train_length=args.max_train_length,
        device=device,
        encoder_type=args.encoder_type,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        transformer_dropout=args.transformer_dropout,
        **config,
    )

    model.fit(train_enc, n_epochs=args.epochs, n_iters=args.iters, verbose=True)
    model.save(f"{run_dir}/model.pkl")

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        valid_dataset, _ = data_provider(args, "val")
        test_dataset, _ = data_provider(args, "test")
        valid_enc, valid_lbl, _, _ = prepare_split(
            valid_dataset,
            add_time_features=args.add_time_features,
            channel_independent=args.channel_independent,
            instance_norm=args.instance_norm,
            time_scaler=time_scaler,
            fit_time_scaler=False,
        )
        test_enc, test_lbl, _, _ = prepare_split(
            test_dataset,
            add_time_features=args.add_time_features,
            channel_independent=args.channel_independent,
            instance_norm=args.instance_norm,
            time_scaler=time_scaler,
            fit_time_scaler=False,
        )

        pred_lens = args.pred_lens if args.pred_lens is not None else [args.pred_len]
        padding = max(0, args.max_train_length - 1)

        eval_batch_size = args.eval_batch_size or args.batch_size
        if args.eval_mode == "full":
            full_enc, full_lbl, train_slice, valid_slice, test_slice = build_full_series_pair(
                train_enc,
                valid_enc,
                test_enc,
                train_lbl,
                valid_lbl,
                test_lbl,
                overlap=args.input_len,
            )
            if valid_slice.start == valid_slice.stop or test_slice.start == test_slice.stop:
                raise ValueError("Validation/test splits are empty after overlap trimming.")
            out, eval_res = eval_linear_probe_full(
                model,
                full_enc,
                full_lbl,
                train_slice,
                valid_slice,
                test_slice,
                scaler,
                pred_lens,
                padding,
                eval_batch_size,
            )
        else:
            out, eval_res = eval_linear_probe(
                model,
                train_enc,
                valid_enc,
                test_enc,
                train_lbl,
                valid_lbl,
                test_lbl,
                scaler,
                pred_lens,
                padding,
                eval_batch_size,
            )
        print("Evaluation result:", eval_res)
        pkl_save(f"{run_dir}/eval_res.pkl", eval_res)
        pkl_save(f"{run_dir}/out.pkl", out)

    print("Finished.")
