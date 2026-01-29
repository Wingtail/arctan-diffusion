from data_provider.data_factory import data_provider
import argparse
import copy

parser = argparse.ArgumentParser(description="ArcTanDiffusion")

# basic config
parser.add_argument(
    "--task_name",
    type=str,
    default="pretrain",
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
parser.add_argument("--label_len", type=int, default=0, help="start token length")
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
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
parser.add_argument(
    "--lr_schedule",
    type=str,
    default="cosine",
    choices=["constant", "cosine"],
    help="learning rate schedule",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=1e-6,
    help="minimum learning rate for cosine schedule",
)
parser.add_argument(
    "--save_directory", type=str, default="./arctandiffusion_etth1_base", help="directory to save model checkpoints"
)
parser.add_argument(
    "--experiment_name", type=str, default="ETTh1_pretrain", help="directory to save model checkpoints"
)
parser.add_argument(
    "--train_epochs", type=int, default=50, help="number of training epochs"
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
    "--n_global", type=int, default=1, help="number of global views for canary pretraining"
)
parser.add_argument(
    "--loss_type", type=str, default="arctan", help="loss type, options: [arctan, mse]"
)
parser.add_argument(
    "--recon_head_depth", type=int, default=1, help="depth of the reconstruction head"
)
parser.add_argument(
    "--timestep_sampling", type=str, default="logit_normal", help="timestep sampling strategy, options: [logit_normal, uniform]"
)
parser.add_argument(
    "--is_causal",
    action="store_true",
    help="whether the model is causal",
)
parser.add_argument(
    "--sample_same_timesteps",
    action="store_true",
    help="whether to sample the same timesteps for all views",
)
parser.add_argument(
    "--diffusion_loss_type", type=str, default="v", help="diffusion loss target type, options: [x0, e, v]"
)
parser.add_argument(
    "--model_pred_type", type=str, default="x0", help="model prediction type, options: [x0, e, v]"
)
parser.add_argument(
    "--instance_norm", type=int, default=1, help="whether to use instance normalization"
)
parser.add_argument(
    "--feature_injection",
    action="store_true",
    help="whether to use feature injection in diffusion reconstruction",
)
parser.add_argument(
    "--mlp_ratio", type=float, default=4.0, help="mlp ratio in transformer blocks"
)
parser.add_argument(
    "--sample_mixed",
    action="store_true",
    help="whether to sample mixed timesteps for diffusion",
)

args = parser.parse_args()

from arctandiff_model_diffusion_only import ArcTanDiffusion

from tqdm import tqdm
import torch
import os
from aim import Run

from torch.amp import GradScaler

def patch_timeseries(batch, patch_size=2):
    B, T = batch.shape
    return batch.reshape(B, T // patch_size, patch_size)

@torch.no_grad()
def update_ema(ema_model, model, alpha=0.9999):
    # EMA over parameters
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
    # Keep buffers (e.g., running stats) in sync
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)

dataset, dataloader = data_provider(args, "train")
val_dataset, val_dataloader = data_provider(args, "val")

device = "cuda"
amp_dtype = torch.bfloat16
autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)

save_dir = args.save_directory

print("sample same timesteps: ", args.sample_same_timesteps)
print("is causal: ", args.is_causal)
print("instance norm: ", args.instance_norm)
print("training epochs: ", args.train_epochs)

model = ArcTanDiffusion(
    in_dim=args.in_dim,
    hidden_dim=args.hidden_dim,
    num_heads=args.num_heads,
    n_layers=args.n_layers,
    patch_size=args.patch_size,
    downstream_task=args.downstream_task,
    model_type=args.model_type,
    loss_type=args.loss_type,
    model_pred_type=args.model_pred_type,
    is_causal=args.is_causal,
    mlp_ratio=args.mlp_ratio,
    recon_head_depth=args.recon_head_depth,
    dropout=0.1,
    attn_dropout=0.1,
    attn_backend="flash",
).to(device)

aim_run = Run(experiment=args.experiment_name)
os.makedirs(save_dir, exist_ok=True)

encoder_params = list(model.encoder.parameters())
encoder_param_ids = {id(p) for p in encoder_params}
non_encoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
encoder_weight_decay = 0.0

norm_modules = (
    torch.nn.LayerNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.SyncBatchNorm,
)
encoder_no_decay_names = set()
for module_name, module in model.encoder.named_modules():
    if isinstance(module, norm_modules):
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            encoder_no_decay_names.add(full_name)
for name, _ in model.encoder.named_parameters():
    if name.endswith(".bias"):
        encoder_no_decay_names.add(name)

encoder_decay_params = []
encoder_no_decay_params = []
for name, param in model.encoder.named_parameters():
    if not param.requires_grad:
        continue
    if name in encoder_no_decay_names:
        encoder_no_decay_params.append(param)
    else:
        encoder_decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": encoder_decay_params, "weight_decay": encoder_weight_decay},
        {"params": encoder_no_decay_params, "weight_decay": 0.0},
        {"params": non_encoder_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
)
lr_scheduler = None
if args.lr_schedule == "cosine":
    total_steps = args.train_epochs * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )
model.train()

# --- EMA init (shadow copy of weights) ---
ema_model = copy.deepcopy(model).to(device)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

val_chunk_size = 32
best_val_loss = float("inf")
patience = 5
epochs_no_improve = 0

log_counter = 0

scaler = GradScaler(enabled="cuda" == "cuda")

print("Training with diffusion loss type: ", args.diffusion_loss_type)
print("Training with model pred type: ", args.model_pred_type)

for epoch in range(args.train_epochs):
    print(f"Epoch {epoch+1}")
    with tqdm(dataloader) as pbar:
        track_loss = 0.0
        # pbar.set_description(f"Loss: {track_loss:.4f}")
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
            optimizer.zero_grad(set_to_none=True)
            series = batch_x.to(device).float()
            with autocast_ctx:
                # Instance normalization
                if args.downstream_task == "forecasting":
                    if args.instance_norm == 1:
                        # print("performing instance norm...")
                        series = (series - series.mean(dim=1, keepdim=True)) / (series.std(dim=1, keepdim=True) + 1e-5)
                # series = series.to(dtype=amp_dtype)
                
                x_pred, loss, losses = model.train_step(
                    series,
                    timestep_sampling=args.timestep_sampling,
                    sample_same_timesteps=args.sample_same_timesteps,
                    diffusion_loss_type=args.diffusion_loss_type,
                    sample_mixed=args.sample_mixed,
                    model_pred_type=args.model_pred_type,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- EMA update (after optimizer step) ---
            update_ema(ema_model, model, alpha=0.999)

            track_loss = loss.detach().float().item()
            pbar.set_description(f"Loss: {track_loss:.4f}")
            
            for key, value in losses.items():
                aim_run.track(
                    value,
                    name=key,
                    step=log_counter
                )

            log_counter += 1
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"{save_dir}/arctandiff_model_epoch_{epoch+1}.pt")
    torch.save(ema_model.state_dict(), f"{save_dir}/arctandiff_ema_model_epoch_{epoch+1}.pt")
