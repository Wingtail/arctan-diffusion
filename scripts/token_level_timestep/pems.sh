#!/bin/sh

epoch_to_load=20
finetune_train_epochs=10
pred_lens="12 24 36 48"
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

pems_data="PEMS03"
for pems in $pems_data; do
    echo "Running for dataset: $pems"
    

    echo "Running with patch size: $patch_size, hidden size: $hidden_size"

    python -u arctandiff_train_diffusion_only_stripped.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path "${pems}.npz" \
        --model_id "${pems}" \
        --model ArcTanDiffusion \
        --data "${pems}" \
        --save_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
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
        --diffusion_loss_type $diffusion_loss_type \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}"

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast_stripped.py \
            --task_name finetune \
            --is_training 1 \
            --root_path ./datasets/PEMS/ \
            --data_path "${pems}.npz" \
            --model_id "${pems}" \
            --model ArcTanDiffusion \
            --data "${pems}" \
            --load_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
            --features M \
            --input_len 96 \
            --label_len 48 \
            --pred_len $pred_len \
            --train_epochs $finetune_train_epochs \
            --in_dim $patch_size \
            --hidden_dim $hidden_size \
            --patch_size $patch_size \
            --downstream_task forecasting \
            --num_heads $num_heads \
            --n_layers 2 \
            --recon_head_depth $recon_head_depth \
            --model_type all_normalized \
            --early_stop \
            --batch_size 16 \
            --include_cls 0 \
            --epoch_to_load $epoch_to_load \
            --start_lr 0.0002 \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type} \
            --base_model_lr_scale 1.0 \
            --save_test_metrics_csv
    done
done

# ----------------------------------------

# PEMS without instance normalization

pems_data="PEMS04 PEMS07 PEMS08"
for pems in $pems_data; do
    echo "Running for dataset: $pems"
    

    echo "Running with patch size: $patch_size, hidden size: $hidden_size"
    linear_probe_run_tag="lp_${pems}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}_${timestep_sampling}"
    linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
    model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

    python -u arctandiff_train_diffusion_only_stripped.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path "${pems}.npz" \
        --model_id "${pems}" \
        --model ArcTanDiffusion \
        --data "${pems}" \
        --save_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
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
        --instance_norm 0 \
        --diffusion_loss_type $diffusion_loss_type \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}"

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast_stripped.py \
            --task_name finetune \
            --is_training 1 \
            --root_path ./datasets/PEMS/ \
            --data_path "${pems}.npz" \
            --model_id "${pems}" \
            --model ArcTanDiffusion \
            --data "${pems}" \
            --load_dir "arctandiffusion_${pems}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
            --features M \
            --input_len 96 \
            --label_len 48 \
            --pred_len $pred_len \
            --train_epochs $finetune_train_epochs \
            --in_dim $patch_size \
            --hidden_dim $hidden_size \
            --patch_size $patch_size \
            --downstream_task forecasting \
            --num_heads $num_heads \
            --n_layers 2 \
            --instance_norm 0 \
            --recon_head_depth $recon_head_depth \
            --model_type all_normalized \
            --early_stop \
            --batch_size 16 \
            --include_cls 0 \
            --epoch_to_load $epoch_to_load \
            --start_lr 0.0002 \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type} \
            --base_model_lr_scale 1.0 \
            --save_test_metrics_csv
    done
done

