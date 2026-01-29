#!/bin/sh

epoch_to_load=50
finetune_train_epochs=10
pred_lens="96 192 336 720"

timestep_sampling="uniform"
recon_head_depth=2
num_heads=16
loss_type="arctan"
hidden_size=128
lr=0.0001
patch_size=8
diffusion_loss_type="v"
finetune_batch_size=16
finetune_start_lr=0.0001
finetune_base_model_lr_scale=0.0

finetune_dropout=0.0
finetune_head_dropout=0.0

input_len=336
n_layers=2

root_path="./datasets/ETT-small/"
src_dataset="ETTh2.csv"
model_id="ETTh2"
data="ETTh2"

few_shot_ratios="5 10"
few_shot_seed=2025

for few_shot_ratio in $few_shot_ratios; do
    if [ "$few_shot_ratio" = "5" ]; then
        few_shot_tag="fs5"
    else
        few_shot_tag="fs10"
    fi

    echo "Few-shot ratio: ${few_shot_ratio}%"

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast.py \
            --task_name finetune \
            --is_training 1 \
            --root_path $root_path \
            --data_path "${src_dataset}" \
            --model_id $model_id \
            --model ArcTanDiffusion \
            --data $data \
            --load_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
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
            --recon_head_depth $recon_head_depth \
            --model_type all_normalized \
            --early_stop \
            --batch_size $finetune_batch_size \
            --include_cls 0 \
            --dropout $finetune_dropout \
            --head_dropout $finetune_head_dropout \
            --epoch_to_load $epoch_to_load \
            --start_lr $finetune_start_lr \
            --few_shot_ratio $few_shot_ratio \
            --few_shot_seed $few_shot_seed \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps_${few_shot_tag} \
            --base_model_lr_scale $finetune_base_model_lr_scale \
            --save_test_metrics_csv
    done
done
