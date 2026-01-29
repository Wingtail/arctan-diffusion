#!/bin/sh

epoch_to_load=50
finetune_train_epochs=10
pred_lens="36 48"
linear_probe_save_root="./outputs/linear_probe_checkpoints"
linear_probe_model_root="./outputs/linear_probe_models"
robustness_out_root="./outputs/robustness"


timestep_sampling="uniform"
recon_head_depth=2
num_heads=8
loss_type="arctan"
hidden_size=256
batch_size=8
lr=0.0002
lr_schedule="cosine"
patch_size=8
diffusion_loss_type="v"

finetune_batch_size=16
finetune_start_lr=0.0001
finetune_base_model_lr_scale=0.0
finetune_lradj="step"
finetune_lr_decay=0.5
finetune_pct_start=0.3
finetune_lr_scheduler="cosine"

input_len=96
n_layers=2

# Sample same timesteps

pems="PEMS04"

echo "Running for dataset: $pems"
echo "Running with patch size: $patch_size, hidden size: $hidden_size"
linear_probe_run_tag="lp_${pems}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}_${timestep_sampling}"
linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

echo "Running for dataset: $pems with same timesteps"

python -u arctandiff_train_diffusion_only.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path "${pems}.npz" \
        --model_id "${pems}" \
        --model ArcTanDiffusion \
        --data "${pems}" \
        --save_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
        --features M \
        --input_len 96 \
        --label_len 0 \
        --pred_len 0 \
        --in_dim $patch_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --hidden_dim $hidden_size \
        --num_heads $num_heads \
        --n_layers 2 \
        --lr $lr \
        --sample_same_timesteps \
        --instance_norm 0 \
        --diffusion_loss_type $diffusion_loss_type \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps"

for pred_len in $pred_lens; do
    python -u arctandiff_finetune_forecast.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path "${pems}.npz" \
        --model_id "${pems}" \
        --model ArcTanDiffusion \
        --data "${pems}" \
        --load_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
        --features M \
        --input_len $input_len \
        --label_len 48 \
        --pred_len $pred_len \
        --train_epochs $finetune_train_epochs \
        --in_dim $patch_size \
        --hidden_dim $hidden_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --instance_norm 0 \
        --dropout 0.0 \
        --clean_forward_noise_level 1.0 \
        --head_dropout 0.0 \
        --random_init 1 \
        --min_lr 1e-6 \
        --recon_head_depth $recon_head_depth \
        --model_type all_normalized \
        --batch_size $finetune_batch_size \
        --include_cls 0 \
        --epoch_to_load $epoch_to_load \
        --start_lr $finetune_start_lr \
        --lr_scheduler $finetune_lr_scheduler \
        --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps_linear_probe \
        --base_model_lr_scale $finetune_base_model_lr_scale \
        --save_test_metrics_csv
done
