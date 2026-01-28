from data_provider.data_factory import data_provider
import argparse
import copy
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from arctandiff_model_diffusion_only_stripped import ArcTanDiffusion, extract_semantic_features_jit

from tqdm import tqdm
import torch
import os
from aim import Run

parser = argparse.ArgumentParser(description="ArcTanDiffusion")

# basic config
parser.add_argument(
    "--task_name",
    type=str,
    default="finetune",
    help="task name, options:[pretrain, finetune]",
)
parser.add_argument("--downstream_task", type=str, default="classification", help="downstream task, options:[forecasting, classification]")
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument(
    "--model_id", type=str, default="Epilepsy", help="model id"
)
parser.add_argument(
    "--model", type=str, default="ArcTanDiffusion", help="model name"
)
# data loader
parser.add_argument(
    "--data", type=str, default="Epilepsy", help="dataset type"
)
parser.add_argument(
    "--root_path", type=str, default="datasets/Epilepsy/", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="datasets/", help="data file")
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
parser.add_argument("--seq_len", type=int, default=206, help="input sequence length")
parser.add_argument("--input_len", type=int, default=206, help="input sequence length")
parser.add_argument("--label_len", type=int, default=0, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)
parser.add_argument(
    "--test_pred_len", type=int, default=96, help="test prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)

# optimization
parser.add_argument(
    "--num_workers", type=int, default=5, help="data loader num workers"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size of train input data"
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=1,
    help="batch size for validation/test (forecasting only)",
)
parser.add_argument(
    "--train_epochs", type=int, default=100, help="number of training epochs"
)
parser.add_argument(
    "--learning_rate", "--start_lr", dest="learning_rate", type=float, default=1e-3,
    help="starting learning rate for the optimizer"
)
parser.add_argument(
    "--min_lr", type=float, default=1e-5,
    help="minimum learning rate for cosine decay"
)
parser.add_argument(
    "--lr_scheduler", type=str, default="none", choices=["none", "cosine"],
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
    help="append final test metrics to <data>_test_metrics.csv",
)
parser.add_argument("--num_classes", type=int, default=None, help="number of classes (defaults to --pred_len)")

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
    help="learning rate for backbone params after unfreezing (defaults to 0.1 * --learning_rate)",
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
    "--load_dir", type=str, default="arctandiffusion_epilepsy_base_distill", help="location of model checkpoints to finetune from"
)
parser.add_argument(
    "--patch_size", type=int, default=6, help="patch size for time series"
)
parser.add_argument(
    "--in_dim", type=int, default=3, help="input dimension for time series"
)
parser.add_argument(
    "--hidden_dim", type=int, default=128, help="hidden dimension for time series"
)
parser.add_argument(
    "--num_heads", type=int, default=16, help="number of attention heads for time series"
)
parser.add_argument(
    "--n_layers", type=int, default=2, help="number of attention layers"
)
parser.add_argument(
    "--model_type", type=str, default="all_normalized", help="model type, options: [all_normalized, half_normalized, all_dit]"
)
parser.add_argument(
    "--include_cls", type=int, default=0, help="whether to include cls token"
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
    "--linear_probe_save_dir",
    type=str,
    default="./outputs/linear_probe_checkpoints",
    help="directory to save best linear-probe head weights",
)
parser.add_argument(
    "--save_model_dir",
    type=str,
    default=None,
    help="directory to save combined backbone+head checkpoint",
)
parser.add_argument(
    "--recon_head_depth", type=int, default=1, help="depth of the reconstruction head"
)
parser.add_argument(
    "--random_init", type=int, default=0, help="whether to randomly initialize the model"
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

args = parser.parse_args()
if args.model_learning_rate is None:
    args.model_learning_rate = args.learning_rate * args.base_model_lr_scale
if args.num_classes is None:
    args.num_classes = args.pred_len
linear_probe = args.base_model_lr_scale == 0.0
if linear_probe:
    print("Linear probe mode enabled (base_model_lr_scale=0.0). Backbone frozen; using no_grad for features.")

def patch_timeseries(batch, patch_size=2):
    B, T = batch.shape
    return batch.reshape(B, T // patch_size, patch_size)

def process_chunk_eval(chunk_x, chunk_y, df_model, finetune_layer, device):
    #Channel separation
    series = chunk_x.float().to(device)
    # B, T, C = series.shape
    # series = series.permute(0,2,1).reshape(B*C, T)  # Assuming 2 channels
    
    # Instance normalization
    # std_dev = series.std(dim=1, keepdim=True)
    # center = series.mean(dim=1, keepdim=True)
    # series = (series - center) / (std_dev + 1e-5)

    # x = df_model.get_features(patch.to(device), t=torch.zeros(patch.shape[0], patch.shape[1], device=device) + 0.95)[5]
    x = df_model.clean_forward(series.to(device))
    # x = F.normalize(x, dim=-1)
    # x = x.reshape(x.shape[0], -1)
    
    finetune_output = finetune_layer(x)
    target = chunk_y.long()
    
    return finetune_output, target

class OldClsHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes, bias=False)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])

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

print("test dataset length: ", len(test_dataset))

device = "cuda"

df_model = ArcTanDiffusion(
    in_dim=args.in_dim,
    hidden_dim=args.hidden_dim,
    num_heads=args.num_heads,
    n_layers=args.n_layers,
    patch_size=args.patch_size,
    downstream_task=args.downstream_task,
    recon_head_depth=args.recon_head_depth,
    dropout=0.2,
    input_len=args.input_len,
    model_type=args.model_type,
).to(device)
checkpoint_path = os.path.join(args.load_dir, f"arctandiff_model_epoch_{args.epoch_to_load}.pt")
if args.random_init == 0:
    print("Loading model from: ", checkpoint_path)
    df_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print("Randomly initializing model.")

if linear_probe:
    for p in df_model.parameters():
        p.requires_grad_(False)

# finetune_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(208//8*128, 4)).to(device)

finetune_layer = OldClsHead(args.hidden_dim, args.num_classes, dropout=0.1).to(device)

param_groups = [
    {"params": list(finetune_layer.parameters()), "lr": args.learning_rate, "weight_decay": args.head_weight_decay},
]
if not linear_probe:
    param_groups.append(
        {"params": list(df_model.parameters()), "lr": args.model_learning_rate, "weight_decay": args.model_weight_decay}
    )

optimizer = torch.optim.AdamW(
    param_groups,
    betas=(0.9, 0.999)
)

scheduler = None
if args.lr_scheduler == "cosine":
    total_steps = max(1, len(dataloader) * args.train_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )

chunk_size = 256
best_val_mse = float("inf")
epochs_without_improve = 0
best_acc = -float("inf")
best_epoch = None
best_state = None

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)

def cpu_state_dict(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def save_linear_probe_head(head_state: dict, epoch: int, val_acc: float, test_acc):
    if not linear_probe or args.linear_probe_save_dir is None:
        return None
    os.makedirs(args.linear_probe_save_dir, exist_ok=True)
    head_path = os.path.join(
        args.linear_probe_save_dir,
        f"{args.model_id}_{args.data}_pl{args.pred_len}_best_head.pth",
    )
    payload = {
        "epoch": int(epoch),
        "val_acc": float(val_acc),
        "test_acc": None if test_acc is None else float(test_acc),
        "finetune_layer_state_dict": head_state,
    }
    torch.save(payload, head_path)
    print(f"Saved best linear-probe head to: {head_path}")
    return head_path

def save_combined_checkpoint(backbone_state: dict, head_state: dict, label: str):
    if args.save_model_dir is None:
        return None
    os.makedirs(args.save_model_dir, exist_ok=True)
    combined = {f"backbone.{k}": v for k, v in backbone_state.items()}
    combined.update({f"head.{k}": v for k, v in head_state.items()})
    if "head.fc.bias" not in combined and "head.fc.weight" in combined:
        combined["head.fc.bias"] = torch.zeros(combined["head.fc.weight"].shape[0])
    ckpt_path = os.path.join(
        args.save_model_dir,
        f"{args.model_id}_{args.data}_pl{args.pred_len}_{label}.pth",
    )
    torch.save({"model_state_dict": combined, "args": vars(args)}, ckpt_path)
    print(f"Saved combined checkpoint to: {ckpt_path}")
    return ckpt_path

def evaluate(dataloader, df_model, finetune_layer, device, chunk_size):
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

                finetune_output, target = process_chunk_eval(chunk_x, chunk_y, df_model, finetune_layer, device)

                all_targets.append(target.cpu().numpy())
                all_preds.append(finetune_output.cpu().numpy())

                batch_chunk_x = []
                batch_chunk_y = []

        if len(batch_chunk_x) > 0:
            chunk_x = torch.cat(batch_chunk_x, dim=0)
            chunk_y = torch.cat(batch_chunk_y, dim=0)

            finetune_output, target = process_chunk_eval(chunk_x, chunk_y, df_model, finetune_layer, device)

            all_targets.append(target.cpu().numpy())
            all_preds.append(finetune_output.cpu().numpy())

    if len(all_targets) == 0:
        return None, None

    all_targets = np.concatenate(all_targets, axis=0).reshape(-1)
    all_preds = np.concatenate(all_preds, axis=0)
    predictions = np.argmax(all_preds, axis=1)
    accuracy = (predictions == all_targets).mean()
    f1 = f1_score(all_targets, predictions, average="macro")
    return accuracy, f1

# Freeze the pretrained backbone at the start (head warmup).
# set_requires_grad(model, False)

log_counter = 0
for epoch in range(args.train_epochs):
    if linear_probe:
        df_model.eval()
    else:
        df_model.train()
    finetune_layer.train()

    print(f"Epoch {epoch+1}")
    print(
        f"LR(head)={optimizer.param_groups[0]['lr']:.6e} | "
        # f"LR(body)={optimizer.param_groups[1]['lr']:.6e} | "
        # f"stage={'head-only' if head_only else 'full'}"
    )
    with tqdm(dataloader) as pbar:
        track_loss = 0.0

        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
            series = batch_x.float().to(device)
            
            #Channel separation
            # B, T, C = series.shape
            # series = series.permute(0,2,1).reshape(B*C, T)  # Assuming 2 channels
            
            # Instance normalization
            # std_dev = series.std(dim=1, keepdim=True)
            # center = series.mean(dim=1, keepdim=True)
            # series = (series - center) / (std_dev + 1e-5)

            optimizer.zero_grad(set_to_none=True)

            # with torch.no_grad():
                # embeddings = df_model.clean_forward(patch.to(device), t=torch.zeros(patch.shape[0], patch.shape[1], device=device) + 0.95)[5]
            if linear_probe:
                with torch.no_grad():
                    embeddings = df_model.clean_forward(series)
            else:
                embeddings = df_model.clean_forward(series)
            # embeddings = F.normalize(embeddings, dim=-1)
            
            # x = embeddings.reshape(embeddings.shape[0], -1)
            finetune_output = finetune_layer(embeddings)
            
            task_loss = F.cross_entropy(
                finetune_output,
                batch_y.long().to(device),
            )

            loss = task_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            track_loss = task_loss.item()
            pbar.set_description(f"Loss: {track_loss:.4f}")

            log_counter += 1

    val_accuracy, val_f1 = evaluate(val_dataloader, df_model, finetune_layer, device, chunk_size)
    if val_accuracy is None:
        print("Skipping validation metrics (empty validation set).")
    else:
        print("Accuracy val: ", val_accuracy)
        print("F1 val: ", val_f1)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_epoch = epoch + 1
            best_state = {
                "df_model": cpu_state_dict(df_model),
                "finetune_layer": cpu_state_dict(finetune_layer),
            }
            print(f"New best validation accuracy: {best_acc:.6f}")

def append_best_metrics_csv(metrics_csv: str, accuracy: float, f1: float, epoch: int) -> None:
    header_needed = (not os.path.exists(metrics_csv)) or os.path.getsize(metrics_csv) == 0
    with open(metrics_csv, "a") as f:
        if header_needed:
            f.write("epoch,accuracy,f1\n")
        f.write(f"{int(epoch)},{float(accuracy):.6f},{float(f1):.6f}\n")
    print(f"Appended best-accuracy metrics to {metrics_csv}")

if best_state is not None:
    df_model.load_state_dict(best_state["df_model"])
    finetune_layer.load_state_dict(best_state["finetune_layer"])
    print(f"Loaded best validation model from epoch {best_epoch}.")
    test_accuracy, test_f1 = evaluate(test_dataloader, df_model, finetune_layer, device, chunk_size)
    if test_accuracy is None:
        print("Skipping test metrics (empty test set).")
    else:
        print(f"Best-val model test accuracy (epoch {best_epoch}): {test_accuracy:.6f}")
        print(f"Best-val model test F1 (epoch {best_epoch}): {test_f1:.6f}")
        if args.save_test_metrics_csv:
            metrics_csv = f"{args.data}_test_metrics_{args.csv_suffix}_best_acc.csv"
            append_best_metrics_csv(metrics_csv, test_accuracy, test_f1, best_epoch)
    if linear_probe:
        save_linear_probe_head(best_state["finetune_layer"], best_epoch, best_acc, test_accuracy)
    save_combined_checkpoint(best_state["df_model"], best_state["finetune_layer"], "best_val_acc")
