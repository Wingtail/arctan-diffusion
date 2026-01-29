#!/bin/sh

diffusion_weight=1.0

epoch_to_load=50
finetune_train_epochs=100

timestep_sampling="uniform"
recon_head_depth=2
num_heads=16
loss_type="arctan"
hidden_size=128
batch_size=16
lr=0.0001
lr_schedule="cosine"
patch_size=8
diffusion_loss_type="v"

echo "Running with patch size: $patch_size, hidden size: $hidden_size"

python -u arctandiff_train_diffusion_only.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path ./datasets/HAR/ \
    --data_path "HAR.csv" \
    --model_id "HAR" \
    --model ArcTanDiffusion \
    --data "HAR" \
    --save_dir "arctandiffusion_har_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
    --features M \
    --input_len 128 \
    --label_len 0 \
    --pred_len 0 \
    --in_dim 9 \
    --patch_size $patch_size \
    --downstream_task classification \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers 2 \
    --lr $lr \
    --instance_norm 0 \
    --downstream_task classification \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps"

python -u arctandiff_finetune_classification.py \
    --task_name finetune \
    --is_training 1 \
    --root_path ./datasets/HAR/ \
    --data_path "HAR.csv" \
    --model_id "HAR" \
    --model ArcTanDiffusion \
    --data "HAR" \
    --load_dir "arctandiffusion_har_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
    --features M \
    --input_len 128 \
    --label_len 0 \
    --pred_len 6 \
    --train_epochs $finetune_train_epochs \
    --in_dim 9 \
    --hidden_dim $hidden_size \
    --patch_size $patch_size \
    --downstream_task classification \
    --num_heads $num_heads \
    --lr_scheduler none \
    --n_layers 2 \
    --recon_head_depth $recon_head_depth \
    --model_type all_normalized \
    --include_cls 0 \
    --epoch_to_load $epoch_to_load \
    --start_lr 0.001 \
    --save_test_metrics_csv \
    --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps \
    --base_model_lr_scale 1.0 \
    --save_test_metrics_csv

