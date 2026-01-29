#!/bin/sh
set -eu

DATA=HAR
ROOT_PATH=./datasets/HAR
DATA_PATH=HAR.csv

epoch_to_load=50
finetune_train_epochs=100
pred_len=6
input_len=128

patch_size=8
hidden_size=128
num_heads=16
n_layers=2
recon_head_depth=2

diffusion_loss_type="v"
lr=0.0001
enc_in=9
num_classes=$pred_len

linear_probe_save_root="./outputs/linear_probe_checkpoints"
linear_probe_model_root="./outputs/linear_probe_models"

out_dir="./outputs/robustness_cps"
summary_csv="${out_dir}/HAR_classification_cps_robustness_summary.csv"

loss_types="arctan huber mse"

mkdir -p "$out_dir"
if [ ! -f "$summary_csv" ] || [ ! -s "$summary_csv" ]; then
  echo "loss_type,checkpoint,output_dir,file_prefix" > "$summary_csv"
fi

for loss_type in $loss_types; do
  echo "=== Robustness test for HAR loss_type: ${loss_type} ==="

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

  linear_probe_run_tag="lp_${DATA}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}"
  linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
  model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

  pretrain_dir_default="arctandiffusion_har_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}"
  pretrain_dir="$pretrain_dir_default"
  if [ -n "$pretrain_override" ]; then
    pretrain_dir="$pretrain_override"
  elif [ ! -d "$pretrain_dir" ]; then
    pretrain_dir="$(find . -maxdepth 2 -type d \
      -name "arctandiffusion_har_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}*" \
      2>/dev/null | head -n 1)"
  fi

  if [ -z "${pretrain_dir:-}" ] || [ ! -d "$pretrain_dir" ]; then
    echo "Pretrain directory not found for ${DATA} (${loss_type})."
    echo "Expected: ${pretrain_dir_default}"
    echo "Tip: set PRETRAIN_DIR_${loss_type}=/path/to/pretrain_dir"
    exit 1
  fi

  combined_ckpt_default="${model_save_dir}/${DATA}_${DATA}_pl${pred_len}_best_val_acc.pth"
  if [ -n "$ckpt_override" ]; then
    combined_ckpt="$ckpt_override"
    if [ ! -f "$combined_ckpt" ]; then
      echo "Override checkpoint not found for ${loss_type}: ${combined_ckpt}"
      exit 1
    fi
  else
    combined_ckpt="$combined_ckpt_default"
  fi

  if [ -z "$ckpt_override" ] && [ ! -f "$combined_ckpt" ]; then
    if [ -d "$model_save_dir" ]; then
      found_ckpt="$(find "$model_save_dir" -maxdepth 1 -type f -name "${DATA}_${DATA}_pl${pred_len}_*.pth" 2>/dev/null | head -n 1)"
      if [ -n "$found_ckpt" ]; then
        combined_ckpt="$found_ckpt"
      fi
    fi
  fi

  if [ -z "$ckpt_override" ] && { [ "${FORCE_FINETUNE:-0}" -eq 1 ] || [ ! -f "$combined_ckpt" ]; }; then
    echo "Running linear-probe finetune for ${DATA} (${loss_type})."
    python -u arctandiff_finetune_classification.py \
      --task_name finetune \
      --is_training 1 \
      --root_path "$ROOT_PATH" \
      --data_path "$DATA_PATH" \
      --model_id "$DATA" \
      --model ArcTanDiffusion \
      --data "$DATA" \
      --load_dir "$pretrain_dir" \
      --features M \
      --input_len $input_len \
      --label_len 0 \
      --pred_len $pred_len \
      --train_epochs $finetune_train_epochs \
      --in_dim $enc_in \
      --hidden_dim $hidden_size \
      --patch_size $patch_size \
      --downstream_task classification \
      --num_heads $num_heads \
      --lr_scheduler none \
      --n_layers $n_layers \
      --recon_head_depth $recon_head_depth \
      --model_type all_normalized \
      --include_cls 0 \
      --epoch_to_load $epoch_to_load \
      --start_lr $lr \
      --base_model_lr_scale 0.0 \
      --linear_probe_save_dir "$linear_probe_save_dir" \
      --save_model_dir "$model_save_dir" \
      --save_test_metrics_csv \
      --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}_linear_probe
    combined_ckpt="$combined_ckpt_default"
  fi

  if [ ! -f "$combined_ckpt" ]; then
    echo "Combined HAR checkpoint not found for ${loss_type}."
    echo "Expected: ${combined_ckpt_default}"
    echo "Tip: set CKPT_${loss_type}=/path/to/combined_checkpoint"
    exit 1
  fi

  file_prefix="ArcTanDiffusion_${DATA}_${loss_type}_"

  python -u cps_robustness_benchmark_classification.py \
    --model ArcTanDiffusion \
    --checkpoint "$combined_ckpt" \
    --downstream_task classification \
    --data "$DATA" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --d_model $hidden_size \
    --n_heads $num_heads \
    --e_layers $n_layers \
    --d_layers $recon_head_depth \
    --patch_len $patch_size \
    --dropout 0.0 \
    --head_dropout 0.0 \
    --use_norm 0 \
    --num_classes $num_classes \
    --model_type all_normalized \
    --batch_size 32 \
    --n_test_samples 0 \
    --test_metric CE \
    --file_prefix "$file_prefix" \
    --out_dir "$out_dir"

  echo "${loss_type},${combined_ckpt},${out_dir},${file_prefix}" >> "$summary_csv"
done
