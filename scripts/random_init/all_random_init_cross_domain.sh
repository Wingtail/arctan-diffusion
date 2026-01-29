#!/bin/sh

# Random-init fine-tuning baselines for forecasting + classification datasets.

# Shared settings
finetune_base_model_lr_scale=1.0
finetune_train_epochs=10

timestep_sampling="uniform"
recon_head_depth=2
loss_type="arctan"
diffusion_loss_type="v"

run_forecast_random_init() {
    echo "Running random-init finetune for dataset: ${data} (${src_dataset})"
    echo "Running with patch size: $patch_size, hidden size: $hidden_size"

    load_tag=${load_tag:-$src_dataset}
    suffix_extra=${suffix_extra:-""}

    for pred_len in $pred_lens; do
        python -u arctandiff_finetune_forecast.py \
            --task_name finetune \
            --is_training 1 \
            --root_path "$root_path" \
            --data_path "$src_dataset" \
            --model_id "$model_id" \
            --model ArcTanDiffusion \
            --data "$data" \
            --load_dir "arctandiffusion_${load_tag}_${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}${suffix_extra}" \
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
            ${min_lr:+--min_lr $min_lr} ${mlp_ratio:+--mlp_ratio $mlp_ratio} ${instance_norm:+--instance_norm $instance_norm} \
            --csv_suffix ${loss_type}_patch_size_${patch_size}_hidden_size_${hidden_size}_recon_head_${recon_head_depth}_${timestep_sampling}_lr${lr}_${diffusion_loss_type}${suffix_extra}_random_init_cross_domain \
            --base_model_lr_scale $finetune_base_model_lr_scale \
            --random_init 1 \
            --save_test_metrics_csv
    done
}

# ----------------------------
# Forecasting datasets
# ----------------------------

# ETTm2
epoch_to_load=50
pred_lens="96 192 336 720"
num_heads=16
hidden_size=128
batch_size=16
lr=0.0001
patch_size=8
finetune_batch_size=16
finetune_start_lr=0.0001
finetune_dropout=0.1
finetune_head_dropout=0.1
input_len=336
n_layers=3
root_path="./datasets/ETT-small/"
src_dataset="ETTm2.csv"
model_id="ETTm2"
data="ETTm2"
min_lr=""
mlp_ratio=""
instance_norm=""
load_tag=""
suffix_extra=""
run_forecast_random_init

# Weather
epoch_to_load=50
pred_lens="96 192 336 720"
num_heads=16
hidden_size=128
batch_size=16
lr=0.0001
patch_size=8
finetune_batch_size=16
finetune_start_lr=0.0001
finetune_dropout=0.2
finetune_head_dropout=0.1
input_len=336
n_layers=3
root_path="./datasets/weather/"
src_dataset="weather.csv"
model_id="Weather"
data="Weather"
min_lr="1e-6"
mlp_ratio="2"
instance_norm=""
load_tag=""
suffix_extra=""
run_forecast_random_init

# Exchange (same-timestep naming)
epoch_to_load=50
pred_lens="96 192 336 720"
num_heads=16
hidden_size=128
batch_size=16
lr=0.0001
patch_size=8
finetune_batch_size=16
finetune_start_lr=0.0001
finetune_dropout=0.2
finetune_head_dropout=0.1
input_len=336
n_layers=3
root_path="./datasets/exchange/"
src_dataset="exchange.csv"
model_id="Exchange"
data="Exchange"
min_lr="1e-7"
mlp_ratio=""
instance_norm=""
load_tag=""
suffix_extra="_sample_same_timesteps"
run_forecast_random_init

# PEMS03/04/07/08
epoch_to_load=50
pred_lens="12 24 36 48"
num_heads=8
hidden_size=128
batch_size=8
lr=0.0002
patch_size=8
finetune_batch_size=16
finetune_start_lr=0.0002
finetune_dropout=0.1
finetune_head_dropout=0.1
input_len=96
n_layers=3
root_path="./datasets/PEMS/"
mlp_ratio=""
min_lr=""

run_pems_random_init() {
    pems="$1"
    instance_norm="$2"
    src_dataset="${pems}.npz"
    model_id="$pems"
    data="$pems"
    load_tag="$pems"
    suffix_extra=""
    run_forecast_random_init
}

run_pems_random_init "PEMS04" "0"
