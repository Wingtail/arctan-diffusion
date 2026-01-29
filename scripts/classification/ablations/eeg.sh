#!/bin/sh


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
patch_size=75
diffusion_loss_type="v"

root_path="./datasets/eeg_no_big/"
src_dataset="eeg_no_big.csv"
model_id="EEG"
data="EEG"
input_len=3000
pred_len=8
in_dim=2
n_layers=2

echo "Running for dataset: $src_dataset"
echo "Running with patch size: $patch_size, hidden size: $hidden_size"

echo "Running for dataset: $src_dataset without same timesteps"
python -u arctandiff_train_diffusion_only_stripped.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --save_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 0 \
    --in_dim $in_dim \
    --patch_size $patch_size \
    --downstream_task classification \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers $n_layers \
    --lr $lr \
    --instance_norm 0 \
    --downstream_task classification \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --diffusion_loss_type $diffusion_loss_type \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}"

python -u arctandiff_finetune_classification_stripped.py \
    --task_name finetune \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --load_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs $finetune_train_epochs \
    --in_dim $in_dim \
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
    --start_lr 0.001 \
    --save_test_metrics_csv \
    --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type} \
    --base_model_lr_scale 1.0 \
    --save_test_metrics_csv

# Same timestep ablation

echo "Running for dataset: $src_dataset with same timesteps (ablation)"
python -u arctandiff_train_diffusion_only_stripped.py \
    --task_name pretrain \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --save_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 0 \
    --in_dim $in_dim \
    --patch_size $patch_size \
    --downstream_task classification \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers $n_layers \
    --lr $lr \
    --instance_norm 0 \
    --downstream_task classification \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --diffusion_loss_type $diffusion_loss_type \
    --sample_same_timesteps \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps"

python -u arctandiff_finetune_classification_stripped.py \
    --task_name finetune \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --load_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs $finetune_train_epochs \
    --in_dim $in_dim \
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
    --start_lr 0.001 \
    --save_test_metrics_csv \
    --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}_sample_same_timesteps \
    --base_model_lr_scale 1.0 \
    --save_test_metrics_csv

# Different loss types (no same-timestep sampling)
diffusion_loss_type="v"
echo "Running for dataset: $src_dataset with different loss types"
for loss_type in "arctan" "huber" "mse"; do
    echo "Loss type: $loss_type"

    python -u arctandiff_train_diffusion_only_stripped.py \
        --task_name pretrain \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --save_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len 0 \
        --in_dim $in_dim \
        --patch_size $patch_size \
        --downstream_task classification \
        --hidden_dim $hidden_size \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --lr $lr \
        --instance_norm 0 \
        --downstream_task classification \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --diffusion_loss_type $diffusion_loss_type \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}"

    python -u arctandiff_finetune_classification_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --load_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len $pred_len \
        --train_epochs $finetune_train_epochs \
        --in_dim $in_dim \
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
        --start_lr 0.001 \
        --save_test_metrics_csv \
        --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type} \
        --base_model_lr_scale 1.0 \
        --save_test_metrics_csv
done

# Different Diffusion loss types (no same-timestep sampling)
loss_type="arctan"
echo "Running for dataset: $src_dataset with different diffusion loss types"
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
        --save_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len 0 \
        --in_dim $in_dim \
        --patch_size $patch_size \
        --downstream_task classification \
        --hidden_dim $hidden_size \
        --num_heads $num_heads \
        --n_layers $n_layers \
        --lr $lr \
        --instance_norm 0 \
        --downstream_task classification \
        --timestep_sampling $timestep_sampling \
        --recon_head_depth $recon_head_depth \
        --loss_type $loss_type \
        --batch_size $batch_size \
        --lr_schedule $lr_schedule \
        --diffusion_loss_type $diffusion_loss_type \
        --model_type all_dit \
        --train_epochs $epoch_to_load \
        --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}"

    python -u arctandiff_finetune_classification_stripped.py \
        --task_name finetune \
        --is_training 1 \
        --root_path $root_path \
        --data_path "${src_dataset}" \
        --model_id $model_id \
        --model ArcTanDiffusion \
        --data $data \
        --load_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
        --features M \
        --input_len $input_len \
        --label_len 0 \
        --pred_len $pred_len \
        --train_epochs $finetune_train_epochs \
        --in_dim $in_dim \
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
        --start_lr 0.001 \
        --save_test_metrics_csv \
        --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type} \
        --base_model_lr_scale 1.0 \
        --save_test_metrics_csv
done

# 1 Head depth ablation (no same-timestep sampling)
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
    --save_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len 0 \
    --in_dim $in_dim \
    --patch_size $patch_size \
    --downstream_task classification \
    --hidden_dim $hidden_size \
    --num_heads $num_heads \
    --n_layers $n_layers \
    --lr $lr \
    --instance_norm 0 \
    --downstream_task classification \
    --timestep_sampling $timestep_sampling \
    --recon_head_depth $recon_head_depth \
    --loss_type $loss_type \
    --batch_size $batch_size \
    --lr_schedule $lr_schedule \
    --diffusion_loss_type $diffusion_loss_type \
    --model_type all_dit \
    --train_epochs $epoch_to_load \
    --experiment_name "EEG_pretrain_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}"

python -u arctandiff_finetune_classification_stripped.py \
    --task_name finetune \
    --is_training 1 \
    --root_path $root_path \
    --data_path "${src_dataset}" \
    --model_id $model_id \
    --model ArcTanDiffusion \
    --data $data \
    --load_dir "arctandiffusion_eeg_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type}" \
    --features M \
    --input_len $input_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs $finetune_train_epochs \
    --in_dim $in_dim \
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
    --start_lr 0.001 \
    --save_test_metrics_csv \
    --csv_suffix ablation_${lr_schedule}_${loss_type}_lr_${lr}_epoch_to_load_${epoch_to_load}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_stripped_lr${lr}_${diffusion_loss_type} \
    --base_model_lr_scale 1.0 \
    --save_test_metrics_csv
