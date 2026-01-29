from data_provider.data_factory import data_provider
import argparse
import copy
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from arctandiff_model_diffusion_only import ArcTanDiffusion, extract_semantic_features_jit

from tqdm import tqdm
import torch
import os
from aim import Run

from arctandiff_utility import ArcTanLoss

parser = argparse.ArgumentParser(description="ArcTanDiffusion")

# basic config
parser.add_argument(
    "--task_name",
    type=str,
    default="finetune",
    help="task name, options:[pretrain, finetune]",
)
parser.add_argument("--downstream_task", type=str, default="forecast", help="downstream task, options:[forecasting, classification]")
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument(
    "--model_id", type=str, default="ArcTanDiffusion", help="model id"
)
parser.add_argument(
    "--model", type=str, default="ArcTanDiffusion", help="model name"
)
# data loader
parser.add_argument(
    "--data", type=str, default="ETTh1", help="dataset type"
)
parser.add_argument(
    "--root_path", type=str, default="./datasets", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="ETT-small/ETTh1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./outputs/checkpoints/",
    help="location of model fine-tuning checkpoints",
)
parser.add_argument(
    "--pretrain_checkpoints",
    type=str,
    default="./outputs/pretrain_checkpoints/",
    help="location of model pre-training checkpoints",
)
parser.add_argument(
    "--transfer_checkpoints",
    type=str,
    default="ckpt_best.pth",
    help="checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]",
)
parser.add_argument(
    "--load_checkpoints", type=str, default=None, help="location of model checkpoints"
)
parser.add_argument(
    "--select_channels",
    type=float,
    default=1,
    help="select the rate of channels to train",
)
parser.add_argument(
    "--use_norm",
    type=int,
    default=1,
    help="use normalization",
)
parser.add_argument(
    "--accumulation_steps",
    type=int,
    default=4,
    help="number of accumulation steps",
)
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)

# forecasting task
parser.add_argument("--seq_len", type=int, default=336, help="input sequence length")
parser.add_argument("--input_len", type=int, default=336, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=720, help="prediction sequence length"
)
parser.add_argument(
    "--test_pred_len", type=int, default=720, help="test prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)

# optimization
parser.add_argument(
    "--num_workers", type=int, default=5, help="data loader num workers"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=1,
    help="batch size for validation/test (forecasting only)",
)
parser.add_argument(
    "--train_epochs", type=int, default=10, help="number of training epochs"
)
parser.add_argument(
    "--learning_rate", "--start_lr", dest="learning_rate", type=float, default=1e-4,
    help="starting learning rate for the optimizer"
)
parser.add_argument(
    "--min_lr", type=float, default=1e-5,
    help="minimum learning rate for cosine decay"
)
parser.add_argument(
    "--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"],
    help="learning rate schedule"
)
parser.add_argument(
    "--early_stop", action="store_true",
    help="enable validation-based early stopping"
)
parser.add_argument(
    "--early_stop_patience", type=int, default=10,
    help="epochs without validation improvement before stopping"
)
parser.add_argument(
    "--early_stop_min_delta", type=float, default=0.0,
    help="minimum validation MSE improvement to reset patience"
)
parser.add_argument(
    "--save_test_metrics_csv",
    action="store_true",
    help="append final test metrics to <data>_test_metrics_<suffix>.csv",
)

# fine-tuning schedule / regularization
parser.add_argument(
    "--head_warmup_epochs",
    type=int,
    default=40,
    help="train only the linear head for this many epochs before unfreezing the backbone",
)
parser.add_argument(
    "--model_learning_rate",
    type=float,
    default=None,
    help="learning rate for backbone params after unfreezing (defaults to --learning_rate / 20)",
)
parser.add_argument(
    "--head_weight_decay",
    type=float,
    default=0.0,
    help="weight decay for the linear head",
)
parser.add_argument(
    "--model_weight_decay",
    type=float,
    default=0.0,
    help="weight decay for backbone params",
)
parser.add_argument(
    "--model_min_lr",
    type=float,
    default=None,
    help="minimum learning rate for cosine decay on backbone params (defaults to --min_lr scaled by backbone LR ratio)",
)
parser.add_argument(
    "--drift_penalty_lambda",
    type=float,
    default=0,
    help="L2 penalty weight to keep backbone params close to their initial (pretrained) values; 0 disables",
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=1.0,
    help="clip gradients to this global norm; <=0 disables",
)
parser.add_argument(
    "--load_dir", type=str, default=None, help="location of model checkpoints to finetune from"
)
parser.add_argument(
    "--patch_size", type=int, default=12, help="patch size for time series"
)
parser.add_argument(
    "--in_dim", type=int, default=12, help="input dimension for time series"
)
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="hidden dimension for time series"
)
parser.add_argument(
    "--num_heads", type=int, default=16, help="number of attention heads for time series"
)
parser.add_argument(
    "--n_layers", type=int, default=4, help="number of attention heads for time series"
)
parser.add_argument(
    "--model_type", type=str, default="all_normalized", help="model type, options: [all_normalized, half_normalized, all_dit]"
)
parser.add_argument(
    "--epoch_to_load", type=int, default=50, help="which epoch to load for finetuning"
)
parser.add_argument(
    "--csv_suffix", type=str, default="", help="suffix for saving csv metrics"
)
parser.add_argument(
    "--base_model_lr_scale", type=float, default=1.0, help="scale of the base model lr compared to the head lr"
)
parser.add_argument(
    "--recon_head_depth", type=int, default=1, help="depth of the reconstruction head"
)
parser.add_argument(
    "--random_init", type=int, default=0, help="whether to randomly initialize the model"
)
parser.add_argument(
    "--dropout", type=float, default=0.1, help="dropout rate for the model"
)
parser.add_argument(
    "--head_dropout", type=float, default=0.1, help="dropout rate for the head"
)
parser.add_argument(
    "--linear_probe",
    action="store_true",
    help="train only the finetune layer (freeze backbone) and save best head by val MSE",
)
parser.add_argument(
    "--linear_probe_save_dir",
    type=str,
    default="./outputs/linear_probe_checkpoints",
    help="directory to save best linear-probe head weights",
)
parser.add_argument(
    "--save_model_dir",
    type=str,
    default=None,
    help="directory to save combined backbone+head checkpoint for downstream evaluation",
)
parser.add_argument(
    "--is_causal",
    action="store_true",
    help="whether the model is causal",
)
parser.add_argument(
    "--instance_norm", type=int, default=1, help="whether to use instance normalization"
)
parser.add_argument(
    "--metric_save_dir", type=str, default="./outputs_canary/metrics/", help="directory to save metrics"
)
parser.add_argument(
    "--mlp_ratio", type=int, default=4, help="mlp ratio for transformer"
)
parser.add_argument(
    "--few_shot_ratio",
    type=float,
    default=1.0,
    help="fraction of training data to use (e.g., 0.05 or 0.1). Values >1 are treated as percentages.",
)
parser.add_argument(
    "--few_shot_seed",
    type=int,
    default=2025,
    help="random seed for selecting the few-shot training subset",
)
parser.add_argument(
    "--clean_forward_noise_level",
    type=float,
    default=1.0,
    help="noise level to use for clean_forward; 1.0 means no noise",
)

args = parser.parse_args()

os.makedirs(args.metric_save_dir, exist_ok=True)

if args.model_learning_rate is None:
    args.model_learning_rate = args.learning_rate * args.base_model_lr_scale
if args.model_min_lr is None:
    if args.learning_rate > 0:
        args.model_min_lr = args.min_lr * (args.model_learning_rate / args.learning_rate)
    else:
        args.model_min_lr = args.min_lr

if args.base_model_lr_scale == 0.0:
    print("Setting linear probe to True since base model LR scale is 0.0")
    args.linear_probe = True

print("linear probe is set to: ", args.linear_probe)
print("instance norm is set to: ", args.instance_norm)

def patch_timeseries(batch, patch_size=2):
    B, T = batch.shape
    return batch.reshape(B, T // patch_size, patch_size)

def process_chunk_eval(chunk_x, chunk_y, df_model, model, finetune_layer, device):
    #Channel separation
    series = chunk_x.float().to(device)
    B, T, C = series.shape
    # series = series.permute(0,2,1).reshape(B*C, T)  # Assuming 2 channels
    
    # Instance normalization
    if args.instance_norm == 1:
        std_dev = series.std(dim=1, keepdim=True)
        center = series.mean(dim=1, keepdim=True)
        series = (series - center) / (std_dev + 1e-5)
    
    # series = series * 0.5 # scale the series
    
    # x = df_model.get_features(series.to(device), t=0.95)[2]
    x = df_model.clean_forward(series.to(device), t=args.clean_forward_noise_level)
    # x = F.normalize(x, dim=-1)
    
    x = x.reshape(x.shape[0], -1)

    if args.instance_norm == 1:
        center = center.permute(0,2,1).reshape(-1, 1)
        std_dev = std_dev.permute(0,2,1).reshape(-1, 1)
        finetune_output = (finetune_layer(x)) * (std_dev + 1e-5) + center
    else:
        finetune_output = finetune_layer(x)
    
    target = chunk_y.float().to(device).permute(0,2,1).reshape(B*C, -1)[:, -args.pred_len:] #(B*C, pred_len)
    
    finetune_output = finetune_output.reshape(B, C, -1).permute(0,2,1) # (B, pred_len, C)
    target = target.reshape(B, C, -1).permute(0,2,1) # (B, pred_len, C)
    
    return finetune_output, target

def maybe_build_few_shot_loader(dataset, dataloader, ratio, seed):
    if ratio is None:
        return dataset, dataloader
    if ratio <= 0:
        raise ValueError("--few_shot_ratio must be > 0.")
    if ratio > 1.0:
        if ratio <= 100.0:
            ratio = ratio / 100.0
        else:
            raise ValueError("--few_shot_ratio > 100 is invalid.")
    if ratio >= 1.0:
        return dataset, dataloader
    total = len(dataset)
    if total == 0:
        return dataset, dataloader

    num_samples = max(1, int(total * ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:num_samples].tolist()
    subset = Subset(dataset, indices)

    drop_last = dataloader.drop_last
    if len(subset) < dataloader.batch_size:
        drop_last = False
        print(
            f"Few-shot subset size ({len(subset)}) is smaller than batch size "
            f"({dataloader.batch_size}); disabling drop_last."
        )

    few_shot_loader = DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        drop_last=drop_last,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
    )
    print(
        f"Using few-shot training subset: {len(subset)}/{total} "
        f"({ratio * 100:.2f}% of training data)."
    )
    return subset, few_shot_loader

dataset, dataloader = data_provider(args, "train")
val_dataset, val_dataloader = data_provider(args, "val")
test_dataset, test_dataloader = data_provider(args, "test")

dataset, dataloader = maybe_build_few_shot_loader(
    dataset, dataloader, args.few_shot_ratio, args.few_shot_seed
)

device = "cuda"

df_model = ArcTanDiffusion(in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads, n_layers=args.n_layers, patch_size=args.patch_size, downstream_task="forecasting", model_type=args.model_type, recon_head_depth=args.recon_head_depth, dropout=args.dropout, mlp_ratio=args.mlp_ratio, is_causal=args.is_causal).to(device)

if args.random_init == 0:
    print("Loading model from: ", f"{args.load_dir}/arctandiff_model_epoch_{args.epoch_to_load}.pt")
    df_model.load_state_dict(torch.load(f"{args.load_dir}/arctandiff_model_epoch_{args.epoch_to_load}.pt", map_location=device), strict=False)
else:
    print("Randomly initializing model.")

criterion = nn.MSELoss()

finetune_layer = nn.Sequential(
    nn.Dropout(args.head_dropout),
    nn.Linear((args.input_len // args.patch_size) * args.hidden_dim, args.pred_len),
).to(device)

param_groups = [
    {
        "params": list(finetune_layer.parameters()),
        "lr": args.learning_rate,
        "weight_decay": args.head_weight_decay,
    }
]
if not args.linear_probe:
    param_groups.append(
        {
            "params": list(df_model.parameters()),
            "lr": args.model_learning_rate,
            "weight_decay": args.model_weight_decay,
        }
    )

optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

scheduler = None
if args.lr_scheduler == "cosine":
    total_steps = max(1, len(dataloader) * args.train_epochs)
    head_start_lr = optimizer.param_groups[0]["lr"]
    head_min_lr = args.min_lr
    base_start_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
    base_min_lr = args.model_min_lr if len(optimizer.param_groups) > 1 else None

    def cosine_lambda(step: int, start_lr: float, min_lr: float, t_max: int) -> float:
        if t_max <= 0:
            return 1.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * step / t_max))
        return (min_lr + (start_lr - min_lr) * cosine) / start_lr if start_lr > 0 else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: cosine_lambda(step, head_start_lr, head_min_lr, total_steps)
        ]
        + (
            [lambda step: cosine_lambda(step, base_start_lr, base_min_lr, total_steps)]
            if base_start_lr is not None
            else []
        ),
    )

chunk_size = 4
best_val_mse = float("inf")
best_state = None
best_val_mse_epoch = -1
epochs_without_improve = 0

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)

def cpu_state_dict(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def evaluate(df_model, finetune_layer, dataloader, device, pred_len, chunk_size):
    tot_val_loss = 0.0
    df_model.eval()
    finetune_layer.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        batch_chunk_x = []
        batch_chunk_y = []
        
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(dataloader):
            batch_chunk_x.append(batch_x)
            batch_chunk_y.append(batch_y)
            
            if len(batch_chunk_x) * batch_x.shape[0] >= chunk_size:
                chunk_x = torch.cat(batch_chunk_x, dim=0)
                chunk_y = torch.cat(batch_chunk_y, dim=0)
                
                finetune_output, target = process_chunk_eval(chunk_x, chunk_y, df_model, df_model, finetune_layer, device)
                
                # finetune_output: (B, pred_len, C)
                # target: (B, pred_len, C)
                
                all_targets.append(target.cpu().numpy())
                all_preds.append(finetune_output.cpu().numpy())
                
                val_loss = F.mse_loss(finetune_output, target)
                tot_val_loss += val_loss.item()
                
                batch_chunk_x = []
                batch_chunk_y = []

        if len(batch_chunk_x) > 0:
            chunk_x = torch.cat(batch_chunk_x, dim=0)
            chunk_y = torch.cat(batch_chunk_y, dim=0)
            
            finetune_output, target = process_chunk_eval(chunk_x, chunk_y, df_model, df_model, finetune_layer, device)
            
            all_targets.append(target.cpu().numpy())
            all_preds.append(finetune_output.cpu().numpy())
            
            val_loss = F.mse_loss(finetune_output, target)
            tot_val_loss += val_loss.item()
        
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        mae_val = np.abs(all_targets - all_preds).mean()
        mse_val = ((all_targets - all_preds)**2).mean()
    
    return mse_val, mae_val
    
    


# Freeze the pretrained backbone at the start (head warmup).
# set_requires_grad(model, False)

log_counter = 0
for epoch in range(args.train_epochs):
    df_model.train()
    finetune_layer.train()
    head_only = args.linear_probe
    set_requires_grad(df_model, not head_only)
    set_requires_grad(finetune_layer, True)

    print(f"Epoch {epoch+1}")
    if len(optimizer.param_groups) > 1:
        print(
            f"LR(head)={optimizer.param_groups[0]['lr']:.6e} | "
            f"LR(base)={optimizer.param_groups[1]['lr']:.6e} | "
        )
    else:
        print(f"LR(head)={optimizer.param_groups[0]['lr']:.6e} | ")
    with tqdm(dataloader) as pbar:
        track_loss = 0.0

        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
            series = batch_x.float().to(device)
            
            #Channel separation
            B, T, C = series.shape

            # Instance normalization
            if args.instance_norm == 1:
                std_dev = series.std(dim=1, keepdim=True)
                center = series.mean(dim=1, keepdim=True)
                series = (series - center) / (std_dev + 1e-5)
                
                center = center.permute(0,2,1).reshape(-1, 1)
                std_dev = std_dev.permute(0,2,1).reshape(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            if head_only:
                with torch.no_grad():
                    embeddings = df_model.clean_forward(series.to(device), t=args.clean_forward_noise_level)
            else:
                embeddings = df_model.clean_forward(series.to(device), t=args.clean_forward_noise_level)

            x = embeddings.reshape(embeddings.shape[0], -1)
            
            if args.instance_norm == 1:
                finetune_output = (finetune_layer(x)) * (std_dev + 1e-5) + center
            else:
                finetune_output = finetune_layer(x)
            
            task_loss = criterion(
                finetune_output,
                batch_y.float().to(device).permute(0, 2, 1).reshape(B * C, -1)[:, -args.pred_len:],
            )

            loss = task_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            track_loss = task_loss.item()
            pbar.set_description(f"Loss: {track_loss:.4f}")

            log_counter += 1

    mse_val, mae_val = evaluate(df_model, finetune_layer, val_dataloader, device, args.pred_len, chunk_size)
    
    print("Mse val: ", mse_val)
    print("Mae val: ", mae_val)
    
    mse_test, mae_test = evaluate(df_model, finetune_layer, test_dataloader, device, args.pred_len, chunk_size)
    
    print("Mse test: ", mse_test)
    print("Mae test: ", mae_test)
    
    break_early = False
    improved_val_mse = mse_val < best_val_mse - args.early_stop_min_delta
    if improved_val_mse:
        best_val_mse = mse_val
        best_state = {
            "finetune_layer": cpu_state_dict(finetune_layer),
        }
        if not args.linear_probe:
            best_state["df_model"] = cpu_state_dict(df_model)
        best_val_mse_epoch = epoch + 1
        epochs_without_improve = 0
        print(f"New best validation MSE: {best_val_mse:.6f}")
        if args.linear_probe:
            os.makedirs(args.linear_probe_save_dir, exist_ok=True)
            head_path = os.path.join(
                args.linear_probe_save_dir,
                f"{args.model_id}_{args.data}_pl{args.pred_len}_best_head.pth",
            )
            torch.save(
                {
                    "epoch": best_val_mse_epoch,
                    "val_mse": float(best_val_mse),
                    "finetune_layer_state_dict": finetune_layer.state_dict(),
                },
                head_path,
            )
            print(f"Saved best linear-probe head to: {head_path}")
    else:
        if args.early_stop:
            epochs_without_improve += 1
            print(f"No validation improvement. Patience {epochs_without_improve}/{args.early_stop_patience}")
            if epochs_without_improve >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break_early = True

    if break_early:
        break

def append_metrics_csv(metrics_csv: str, pred_len: int, mse: float, mae: float) -> None:
    header_needed = (not os.path.exists(metrics_csv)) or os.path.getsize(metrics_csv) == 0
    with open(metrics_csv, "a") as f:
        if header_needed:
            f.write("pred_len,mse,mae\n")
        f.write(f"{pred_len},{float(mse):.6f},{float(mae):.6f}\n")
    print(f"Appended test metrics to {metrics_csv}")

def evaluate_with_state(state, label: str):
    if state is None:
        print(f"Skipping test evaluation for {label} (no saved state).")
        return None, None
    finetune_layer.load_state_dict(state["finetune_layer"])
    if "df_model" in state:
        df_model.load_state_dict(state["df_model"])
    print(f"Loaded fine-tuning weights for {label}.")
    mse_eval, mae_eval = evaluate(df_model, finetune_layer, test_dataloader, device, args.pred_len, chunk_size)
    print(f"Final MSE test ({label}): ", mse_eval)
    print(f"Final MAE test ({label}): ", mae_eval)
    return mse_eval, mae_eval

mse_val_best, mae_val_best = evaluate_with_state(
    best_state, f"best val MSE @ epoch {best_val_mse_epoch}"
)

def save_combined_checkpoint(state, label: str):
    if args.save_model_dir is None:
        return
    if state is None:
        print(f"Skipping combined checkpoint for {label} (no saved state).")
        return
    backbone_state = state.get("df_model") or cpu_state_dict(df_model)
    head_state = state.get("finetune_layer")
    if head_state is None:
        print(f"Skipping combined checkpoint for {label} (missing head state).")
        return
    combined_state = {f"backbone.{k}": v for k, v in backbone_state.items()}
    combined_state.update({f"head.{k}": v for k, v in head_state.items()})
    os.makedirs(args.save_model_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.save_model_dir,
        f"{args.model_id}_{args.data}_pl{args.pred_len}_{label}.pth",
    )
    torch.save({"model_state_dict": combined_state, "args": vars(args)}, checkpoint_path)
    print(f"Saved combined checkpoint to: {checkpoint_path}")

save_combined_checkpoint(best_state, "best_val_mse")

if args.save_test_metrics_csv:
    if mse_val_best is not None and mae_val_best is not None:
        metrics_csv = f"{args.metric_save_dir}{args.data}_test_metrics_{args.csv_suffix}.csv"
        append_metrics_csv(metrics_csv, args.pred_len, mse_val_best, mae_val_best)
