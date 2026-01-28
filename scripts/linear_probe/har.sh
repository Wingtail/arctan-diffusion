#!/bin/sh

epoch_to_load=50
finetune_train_epochs=100

timestep_sampling="uniform"
recon_head_depth=2
num_heads=16
loss_type="arctan"
hidden_size=128
lr=0.0001
patch_size=8
diffusion_loss_type="v"
base_model_lr_scale=0.0

input_len=128
n_layers=2

root_path="./datasets/HAR/"
src_dataset="HAR.csv"
model_id="HAR"
data="HAR"

python -u arctandiff_finetune_classification_stripped.py \
    --task_name finetune \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --load_dir "arctandiffusion_har_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 6 \
    --train_epochs $finetune_train_epochs \
    --in_dim 9 \
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
    --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps_linear_probe \
    --base_model_lr_scale $base_model_lr_scale \
    --save_test_metrics_csv
