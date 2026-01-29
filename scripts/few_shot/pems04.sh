#!/bin/sh

epoch_to_load=50
finetune_train_epochs=10
pred_lens="12 24 36 48"

timestep_sampling="uniform"
recon_head_depth=2
num_heads=8
loss_type="arctan"
hidden_size=256
lr=0.0002
patch_size=8
diffusion_loss_type="v"
finetune_batch_size=16
finetune_start_lr=0.0005
finetune_base_model_lr_scale=1.0

input_len=96
n_layers=2

root_path="./datasets/PEMS/"
pems="PEMS04"

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
            --recon_head_depth $recon_head_depth \
            --model_type all_normalized \
            --early_stop \
            --batch_size $finetune_batch_size \
            --include_cls 0 \
            --epoch_to_load $epoch_to_load \
            --start_lr $finetune_start_lr \
            --few_shot_ratio $few_shot_ratio \
            --few_shot_seed $few_shot_seed \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_sample_same_timesteps_${few_shot_tag} \
            --base_model_lr_scale $finetune_base_model_lr_scale \
            --save_test_metrics_csv
    done
done
