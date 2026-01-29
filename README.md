# ArcTan Diffusion

Code release for the paper "ArcTan Diffusion: Simplifying Time Series Pretraining by Reparameterizing the Denoising Objective" (ICML 2026 submission). The repository includes a DiT-based diffusion pretraining model and shell scripts that reproduce the forecasting, classification, ablations, few-shot, linear-probe, random-init, and robustness experiments in the paper.

ArcTan Diffusion pretrains a diffusion Transformer to predict clean patch tokens from a linear-interpolation corruption and optimizes in v-loss space with ArcTanLoss. The decoder is used only during pretraining; downstream tasks use the encoder representations.

## Setup

1) Create a Python environment (tested with Python 3.10+).
2) Install dependencies:

```bash
pip install -r requirements.txt
pip install aim
```

Notes:
- The training scripts assume CUDA is available and set `device = "cuda"` in code. If you need CPU-only, you will need to modify the scripts.
- `aim` is used for experiment logging in the training and fine-tuning scripts.

## Data

The scripts assume datasets are located under `datasets/` using the following layout. If you already have the data, place it in the same structure.

```
datasets/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  exchange/exchange.csv
  PEMS/PEMS03.npz
  PEMS/PEMS04.npz
  PEMS/PEMS07.npz
  PEMS/PEMS08.npz
  HAR/train.pt
  HAR/val.pt
  HAR/test.pt
  Epilepsy/train_d.npy
  Epilepsy/train_l.npy
  Epilepsy/test_d.npy
  Epilepsy/test_l.npy
  eeg_no_big/samples_train.pkl
  eeg_no_big/samples_test.pkl
```

Notes:
- HAR and EEG loaders ignore `--data_path` and read from `root_path` using `train/val/test` files.
- EEG validation will fall back to the test split if `samples_val.pkl` is missing.

## Shared pretraining configuration (paper Appendix A)

Default shared recipe used for most datasets:
- DiT encoder layers: 2, decoder layers: 2
- Attention heads: 16, hidden size: 128, FFN expansion: 4.0
- Patch size: 8, dropout: 0.1, attention dropout: 0.1
- Learning rate: 1e-4 with cosine schedule
- Pretraining epochs: 50, batch size: 16
- Timestep sampling: sequence-level for forecasting and token-wise independent for classification

Dataset-specific deviations in the paper are reflected in the scripts (for example: PEMS uses hidden size 256, 8 heads, lr 2e-4; EEG uses patch size 75).

## Running experiments (scripts/)

All experiments are driven by shell scripts under `scripts/`. Run them from the repo root:

```bash
bash scripts/same_timestep/ettm1.sh
```

Each script usually runs:
1) Pretraining with `arctandiff_train_diffusion_only.py`
2) Fine-tuning with `arctandiff_finetune_forecast.py` or `arctandiff_finetune_classification.py`

### Script groups

Forecasting:
- `scripts/token_level_timestep/` uses independent per-token timesteps
- `scripts/same_timestep/` uses `--sample_same_timesteps`
- `scripts/ablations/` runs loss and objective ablations

Classification:
- `scripts/classification/` for HAR, Epilepsy, EEG
- `scripts/classification/ablations/` for loss/objective ablations

Additional studies:
- `scripts/linear_probe/` for linear-probe evaluation
- `scripts/few_shot/` for few-shot fine-tuning
- `scripts/random_init/` for random-init baselines
- `scripts/robustness_test/` for CPS robustness benchmarks

### Example runs

Forecasting (ETTm1):
```bash
bash scripts/token_level_timestep/ettm1.sh
```

Classification (HAR):
```bash
bash scripts/classification/har.sh
```

Robustness (ETTh1):
```bash
bash scripts/robustness_test/etth1.sh
```

If you want to change hyperparameters, edit the variables at the top of each script (for example: `epoch_to_load`, `patch_size`, `hidden_size`, `lr`, `loss_type`).

## Outputs and checkpoints

- Pretraining checkpoints: `arctandiffusion_*` directories in the repo root (files like `arctandiff_model_epoch_*.pt`).
- Forecasting metrics: `outputs_canary/metrics/<DATA>_test_metrics_<suffix>.csv` when `--save_test_metrics_csv` is set.
- Classification metrics: `<DATA>_test_metrics_<suffix>_best_acc.csv` in the repo root.
- Linear-probe heads: `outputs/linear_probe_checkpoints/`
- Combined backbone+head checkpoints (when enabled): `outputs/linear_probe_models/` or a custom `--save_model_dir`.
- Robustness outputs: `outputs/robustness_cps/`

## Repro tips

- Run from the repo root so relative dataset paths resolve correctly.
- If you pretrain fewer epochs, update `epoch_to_load` in the fine-tuning scripts to match the checkpoint you want.
