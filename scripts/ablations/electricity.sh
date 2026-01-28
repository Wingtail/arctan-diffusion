#!/bin/sh

epoch_to_load=50
finetune_train_epochs=10
pred_lens="96 192 336 720"
linear_probe_save_root="./outputs/linear_probe_checkpoints"
linear_probe_model_root="./outputs/linear_probe_models"
robustness_out_root="./outputs/robustness"


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

finetune_batch_size=16
finetune_start_lr=0.0004
finetune_base_model_lr_scale=1.0

finetune_dropout=0.1
finetune_head_dropout=0.1

input_len=336
n_layers=2

# Sample same timesteps

root_path="./datasets/electricity/"
src_dataset="electricity.csv"
model_id="Electricity"
data="Electricity"

echo "Running for dataset: $src_dataset"
echo "Running with patch size: $patch_size, hidden size: $hidden_size"
linear_probe_run_tag="lp_${src_dataset}_${loss_type}_patch_${patch_size}_hidden_${hidden_size}_recon_${recon_head_depth}_${timestep_sampling}"
linear_probe_save_dir="${linear_probe_save_root}/${linear_probe_run_tag}"
model_save_dir="${linear_probe_model_root}/${linear_probe_run_tag}"

echo "Running for dataset: $src_dataset with same timesteps"
python -u arctandiff_train_diffusion_only_stripped.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --save_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 0 \
    --in_dim $patch_size \
    --patch_size $patch_size \
    --downstream_task forecasting \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers $n_layers \
    --lr $lr \
    --diffusion_loss_type $diffusion_loss_type \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}"

for pred_len in $pred_lens; do
    python -u arctandiff_finetune_forecast_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --load_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}" \
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
        --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type} \
        --base_model_lr_scale $finetune_base_model_lr_scale \
        --save_test_metrics_csv
done

# Different loss types
diffusion_loss_type="v"
echo "Running for dataset: $src_dataset with different loss types"
for loss_type in "huber" "mse"; do
    echo "Loss type: $loss_type"

    python -u arctandiff_train_diffusion_only_stripped.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --save_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len 0 \
        --in_dim $patch_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --hidden_dim $hidden_size \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --lr $lr \
        --sample_same_timesteps \
        --diffusion_loss_type $diffusion_loss_type \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps"

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast_stripped.py \
            --task_name finetune \
            --is_training 1 \
            --root_path $root_path \
            --data_path "${src_dataset}" \
            --model_id $model_id \
            --model ArcTanDiffusion \
            --data $data \
            --load_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
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
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps \
            --base_model_lr_scale $finetune_base_model_lr_scale \
            --save_test_metrics_csv
    done
done

# Different Diffusion loss types
loss_type="arctan"
echo "Running for dataset: $src_dataset with different loss types"
for diffusion_loss_type in "x0" "e"; do
    echo "Diffusion loss type: $diffusion_loss_type"

    python -u arctandiff_train_diffusion_only_stripped.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --save_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len 0 \
        --in_dim $patch_size \
        --patch_size $patch_size \
        --downstream_task forecasting \
        --hidden_dim $hidden_size \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --lr $lr \
        --sample_same_timesteps \
        --diffusion_loss_type $diffusion_loss_type \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps"

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast_stripped.py \
            --task_name finetune \
            --is_training 1 \
            --root_path $root_path \
            --data_path "${src_dataset}" \
            --model_id $model_id \
            --model ArcTanDiffusion \
            --data $data \
            --load_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
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
            --dropout $finetune_dropout \
            --head_dropout $finetune_head_dropout \
            --batch_size $finetune_batch_size \
            --include_cls 0 \
            --epoch_to_load $epoch_to_load \
            --start_lr $finetune_start_lr \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps \
            --base_model_lr_scale $finetune_base_model_lr_scale \
            --save_test_metrics_csv
    done
done

# 1 Head depth ablation

recon_head_depth=1
loss_type="arctan"
diffusion_loss_type="v"
echo "1 Head depth ablation for dataset: $src_dataset"

python -u arctandiff_train_diffusion_only_stripped.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --save_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 0 \
    --in_dim $patch_size \
    --patch_size $patch_size \
    --downstream_task forecasting \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers $n_layers \
    --lr $lr \
    --sample_same_timesteps \
    --diffusion_loss_type $diffusion_loss_type \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps"

for pred_len in $pred_lens; do
    python -u arctandiff_finetune_forecast_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --load_dir "arctandiffusion_${src_dataset}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps" \
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
        --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}_same_timesteps \
        --base_model_lr_scale $finetune_base_model_lr_scale \
        --save_test_metrics_csv
done