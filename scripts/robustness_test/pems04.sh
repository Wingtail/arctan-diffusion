#!/bin/sh
set -eu

DATA=PEMS04
ROOT_PATH=./datasets/PEMS
DATA_PATH=${DATA}.npz

epoch_to_load=20
finetune_train_epochs=10
pred_lens=${PRED_LENS:-"12 24 36 48"}
input_len=96
label_len=48

patch_size=8
hidden_size=256
num_heads=8
n_layers=2
recon_head_depth=2

timestep_sampling="uniform"
loss_type="${LOSS_TYPE:-arctan}"
diffusion_loss_type="v"
lr=0.0002

instance_norm=0

finetune_batch_size=16
finetune_start_lr=0.0001
finetune_base_model_lr_scale=0.0
finetune_lr_scheduler="cosine"

# PEMS04 has 307 sensors by default; override if your data differs.
enc_in="${ENC_IN:-307}"

linear_probe_save_root="./outputs/linear_probe_checkpoints"
linear_probe_model_root="./outputs/linear_probe_models"

out_dir="./outputs/robustness_cps"
summary_csv="${out_dir}/PEMS04_loss_robustness_summary.csv"

mkdir -p "$out_dir"
if [ ! -f "$summary_csv" ] || [ ! -s "$summary_csv" ]; then
  echo "pred_len,loss_type,checkpoint,output_dir,file_prefix" > "$summary_csv"
fi

linear_probe_run_tag="lp_${DATA}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}_${timestep_sampling}"
linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

pretrain_dir_default="arctandiffusion_${DATA}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps"
pretrain_dir="${PRETRAIN_DIR:-$pretrain_dir_default}"
if [ ! -d "$pretrain_dir" ]; then
  pretrain_dir="$(find . -maxdepth 2 -type d \
    -name "arctandiffusion_${DATA}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps*" \
    2>/dev/null | head -n 1)"
fi

if [ -z "${pretrain_dir:-}" ] || [ ! -d "$pretrain_dir" ]; then
  echo "Pretrain directory not found for ${DATA}."
  echo "Expected: ${pretrain_dir_default}"
  echo "Tip: set PRETRAIN_DIR=/path/to/pretrain_dir"
  exit 1
fi

for pred_len in $pred_lens; do
  ckpt_env_var="CKPT_PEMS04_PL${pred_len}"
  ckpt_override="$(eval "printf '%s' \"\${$ckpt_env_var:-}\"")"

  combined_ckpt_default="${model_save_dir}/${DATA}_${DATA}_pl${pred_len}_best_val_mse.pth"
  if [ -n "$ckpt_override" ]; then
    combined_ckpt="$ckpt_override"
    if [ ! -f "$combined_ckpt" ]; then
      echo "Override checkpoint not found for pred_len ${pred_len}: ${combined_ckpt}"
      exit 1
    fi
  else
    combined_ckpt="$combined_ckpt_default"
  fi

  if [ -z "$ckpt_override" ] && { [ "${FORCE_FINETUNE:-0}" -eq 1 ] || [ ! -f "$combined_ckpt" ]; }; then
    echo "Running linear-probe finetune for pred_len ${pred_len}."
    python -u arctandiff_finetune_forecast_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path "$ROOT_PATH" \
        --data_path "${DATA_PATH}" \
        --model_id "$DATA" \
        --model ArcTanDiffusion \
        --data "$DATA" \
        --load_dir "$pretrain_dir" \
        --features M \
        --input_len $input_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --train_epochs $finetune_train_epochs \
        --in_dim $patch_size \
        --hidden_dim $hidden_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --instance_norm $instance_norm \
        --dropout 0.0 \
        --head_dropout 0.0 \
        --recon_head_depth $recon_head_depth \
        --model_type all_normalized \
        --batch_size $finetune_batch_size \
        --include_cls 0 \
        --epoch_to_load $epoch_to_load \
        --start_lr $finetune_start_lr \
        --lr_scheduler $finetune_lr_scheduler \
        --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps_linear_probe \
        --base_model_lr_scale $finetune_base_model_lr_scale \
        --linear_probe \
        --linear_probe_save_dir "$linear_probe_save_dir" \
        --save_model_dir "$model_save_dir" \
        --save_test_metrics_csv
  fi

  if [ ! -f "$combined_ckpt" ]; then
    echo "Could not find ArcTanDiffusion combined checkpoint for pred_len ${pred_len}."
    echo "Expected: ${combined_ckpt_default}"
    echo "Tip: set ${ckpt_env_var}=/path/to/checkpoint"
    exit 1
  fi

  file_prefix="ArcTanDiffusion_${DATA}_${loss_type}_pl${pred_len}_"

  python -u cps_robustness_benchmark.py \
    --model ArcTanDiffusion \
    --checkpoint "$combined_ckpt" \
    --data "$DATA" \
    --root_path "$ROOT_PATH" \
    --data_path "$DATA_PATH" \
    --features M \
    --input_len $input_len \
    --label_len $label_len \
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
    --model_type all_normalized \
    --batch_size 128 \
    --n_test_samples 0 \
    --file_prefix "$file_prefix" \
    --out_dir "$out_dir"

  echo "${pred_len},${loss_type},${combined_ckpt},${out_dir},${file_prefix}" >> "$summary_csv"
done
