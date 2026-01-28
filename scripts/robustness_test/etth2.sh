#!/bin/sh
set -eu

DATA=ETTh2
ROOT_PATH=./datasets/ETT-small
DATA_PATH=${DATA}.csv

# Mirrors scripts/ablations/etth2.sh (same-timestep loss ablation)
epoch_to_load=50
finetune_train_epochs=10
pred_len=96
input_len=336
label_len=48

patch_size=8
hidden_size=128
num_heads=16
n_layers=2
recon_head_depth=2

timestep_sampling="uniform"
diffusion_loss_type="v"
lr=0.0001

finetune_batch_size=16
finetune_start_lr=0.0001
finetune_base_model_lr_scale=0.0
finetune_dropout=0.0
finetune_head_dropout=0.0

root_path="./datasets/ETT-small/"
src_dataset="ETTh2.csv"
model_id="ETTh2"
data="ETTh2"

linear_probe_save_root="./outputs/linear_probe_checkpoints"
linear_probe_model_root="./outputs/linear_probe_models"

out_dir="./outputs/robustness_cps"
summary_csv="${out_dir}/ETTh2_loss_robustness_summary.csv"

loss_types="arctan huber mse"

mkdir -p "$out_dir"
if [ ! -f "$summary_csv" ] || [ ! -s "$summary_csv" ]; then
  echo "loss_type,checkpoint,output_dir,file_prefix" > "$summary_csv"
fi

for loss_type in $loss_types; do
  echo "=== Robustness test for loss_type: ${loss_type} ==="

  case "$loss_type" in
    arctan)
      pretrain_override="${PRETRAIN_DIR_ARCTAN:-}"
      ckpt_override="${CKPT_ARCTAN:-}"
      ;;
    huber)
      pretrain_override="${PRETRAIN_DIR_HUBER:-}"
      ckpt_override="${CKPT_HUBER:-}"
      ;;
    mse)
      pretrain_override="${PRETRAIN_DIR_MSE:-}"
      ckpt_override="${CKPT_MSE:-}"
      ;;
    *)
      echo "Unknown loss type: ${loss_type}"
      exit 1
      ;;
  esac

  linear_probe_run_tag="lp_${src_dataset}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}_${timestep_sampling}"
  linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
  model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

  pretrain_dir="arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps"
  if [ -n "$pretrain_override" ]; then
    pretrain_dir="$pretrain_override"
  elif [ ! -d "$pretrain_dir" ]; then
    pretrain_dir="$(find . -maxdepth 2 -type d \
      -name "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_*" \
      2>/dev/null | head -n 1)"
  fi

  if [ -z "${pretrain_dir:-}" ] || [ ! -d "$pretrain_dir" ]; then
    echo "Pretrain directory not found for ${loss_type}."
    echo "Expected: ${pretrain_dir}"
    echo "Tip: set PRETRAIN_DIR_${loss_type}=/path/to/pretrain_dir"
    exit 1
  fi

  combined_ckpt_default="${model_save_dir}/${model_id}_${data}_pl${pred_len}_best_val_mse.pth"
  if [ -n "$ckpt_override" ]; then
    combined_ckpt="$ckpt_override"
    if [ ! -f "$combined_ckpt" ]; then
      echo "Override checkpoint not found for ${loss_type}: ${combined_ckpt}"
      exit 1
    fi
  else
    combined_ckpt="$combined_ckpt_default"
  fi

  if [ -z "$ckpt_override" ] && { [ "${FORCE_FINETUNE:-0}" -eq 1 ] || [ ! -f "$combined_ckpt" ]; }; then
    echo "Running linear-probe finetune for ${loss_type}."
    python -u arctandiff_finetune_forecast_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "${src_dataset}" \
        --model_id "$model_id" \
        --model ArcTanDiffusion \
        --data "$data" \
        --load_dir "$pretrain_dir" \
        --features M \
        --input_len $input_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --train_epochs $finetune_train_epochs \
        --batch_size $finetune_batch_size \
        --in_dim $patch_size \
        --hidden_dim $hidden_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --linear_probe \
        --linear_probe_save_dir "$linear_probe_save_dir" \
        --save_model_dir "$model_save_dir" \
        --dropout $finetune_dropout \
        --head_dropout $finetune_head_dropout \
        --recon_head_depth $recon_head_depth \
        --model_type all_normalized \
        --early_stop \
        --include_cls 0 \
        --epoch_to_load $epoch_to_load \
        --start_lr $finetune_start_lr \
        --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps \
        --base_model_lr_scale $finetune_base_model_lr_scale \
        --save_test_metrics_csv
  fi

  if [ ! -f "$combined_ckpt" ]; then
    echo "Could not find ArcTanDiffusion combined checkpoint for ${loss_type}."
    echo "Expected: ${combined_ckpt_default}"
    echo "Tip: set CKPT_${loss_type}=/path/to/checkpoint"
    exit 1
  fi

  file_prefix="ArcTanDiffusion_${DATA}_${loss_type}_"

  python -u cps_robustness_benchmark.py \
    --model ArcTanDiffusion \
    --checkpoint "$combined_ckpt" \
    --data "$DATA" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --target OT \
    --input_len $input_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model $hidden_size \
    --n_heads $num_heads \
    --e_layers $n_layers \
    --d_layers $recon_head_depth \
    --patch_len $patch_size \
    --dropout 0.0 \
    --head_dropout 0.0 \
    --model_type all_normalized \
    --batch_size 16 \
    --n_test_samples 0 \
    --file_prefix "$file_prefix" \
    --out_dir "$out_dir"

  echo "${loss_type},${combined_ckpt},${out_dir},${file_prefix}" >> "$summary_csv"
done
