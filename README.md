# IPEO: Hurricane Damage Detection

Binary classifier for post-hurricane satellite tiles (`damage` vs `no_damage`) with emphasis on calibrated probabilities and dataset analysis.

## Table of Contents
- [1. Quickstart (Inference Notebook)](#1-quickstart-inference-notebook)
- [2. Environment (Training/Development)](#2-environment-trainingdevelopment)
- [3. Datasets](#3-datasets)
- [4. Training Code](#4-training-code)
- [5. Training Details](#5-training-details)
- [6. Calibration and Metrics](#6-calibration-and-metrics)
- [7. Analysis Playbook (Report Notebooks)](#7-analysis-playbook-report-notebooks)
- [8. Repo Map](#8-repo-map)

## 1. Quickstart (Inference Notebook)
1) Create the inference environment: `conda env create -f inference/environment.yml && conda activate hurricane-inference`.
2) Launch VS Code/Jupyter from the repo root and open `inference/inference.ipynb`.
3) Run cells in order. The notebook will download checkpoints and zipped datasets into the repo-root `artifacts/`, then unpack `data/` and `data_resampled/` at the repo root. No manual paths needed.
4) Outputs: validation/test metrics for time-detectors and CNN baselines, plus Grad-CAM overlays that mirror the report. Use the provided sample filenames or choose others under `data_resampled/test`.

## 2. Environment (Training/Development)
- Python 3.10 recommended.
- Install PyTorch/torchvision first (GPU example: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`; CPU: `pip install torch torchvision`).
- Then install extras: `pip install -r requirements.txt`.
- Add `src` to your `PYTHONPATH` (`export PYTHONPATH=$(pwd)/src`).

## 3. Datasets
Two datasets are available (the inference notebook pulls both automatically):
- `data.zip` → `data/`: Original Hurricane Harvey tiles with the provided train/val/test split.
- `data_resampled.zip` → `data_resampled/`: Leak-reduced version using block-based coordinate grouping (keep-one-per-location).

Manual download (if you skip the notebook step):
1) Grab both ZIPs from the shared [Drive folder](https://drive.google.com/drive/folders/1fFiyF_eVUCPRggc8Ti4RAl3zCMj8vAhL?usp=drive_link) 
2) Unzip them at the repo root so you end up with:
```
data/
  train/{damage,no_damage}
  validation/{damage,no_damage}
  test/{damage,no_damage}
data_resampled/
  train/{damage,no_damage}
  validation/{damage,no_damage}
  test/{damage,no_damage}
```
Regenerate the resampled set (only if you want different splits or precision): `python src/data/resample_dataset.py --data-root data --output-root data_resampled --val-split 0.15 --test-split 0.15 --precision 1`
Point `TrainConfig.data_root` to `data/` (original) or `data_resampled/` (cleaned); relative paths resolve from the repo root.

## 4. Training Code
```python
from config import TrainConfig
from training import Trainer

cfg = TrainConfig(
  data_root="data_resampled",  # or data
  model_name="resnet34",
  epochs=20,
  data_integrity_check=True,
  wandb_mode="disabled",
)
trainer = Trainer(cfg)
trainer.fit()
print(trainer.test())
```
Set `data_integrity_check=True` to log coordinate overlap and class counts by split during datamodule setup.

## 5. Training Details
- Config: `src/config.py` (`TrainConfig`, `build_config_from_dict`).
  - Data/augs: resize, flip, rotation, color jitter; dataset mean/std.
  - Imbalance: `balance_strategy` = `weighted_sampler` (sampler) or `class_weights` (loss).
  - Optimization: AdamW/SGD, cosine LR, label smoothing, grad clip, AMP, early stopping on `macro_f1`.
  - Logging/checkpoints: `checkpoints_dir` (best.pt) resolved from repo root, optional TensorBoard/W&B.
- Data loaders: `src/data/transforms.py`, `src/data/datamodule.py`.
  - ImageFolder, train/eval transforms, optional integrity summary of coord overlaps/conflicts when `data_integrity_check=True`.
- Models: `src/models/builder.py`, `src/models/custom.py`.
  - Backbones: resnet18/34/50, efficientnet_b0/b1/b2, convnext_tiny/small, or lightweight `custom_cnn`.
  - Head replacement to `num_classes=2` with optional dropout.
  - `_adapt_first_conv` copies pretrained conv1 weights into a new conv when `in_channels` differs (e.g., >3), avoiding random reinit.
  - `freeze_backbone=True` trains only the head (compare frozen vs fine-tune).
- Training loop: `src/training/trainer.py`.
  - Loss = cross-entropy + label smoothing (+ optional class weights).
  - Scheduler = cosine annealing; AMP + GradScaler; gradient clipping.
  - Early stopping on validation `checkpoint_metric`; stores best checkpoint; AMP enabled only when CUDA is available.

## 6. Calibration and Metrics
- Metrics (`src/validation/metrics.py`): accuracy, macro precision/recall/F1, Brier score (probability quality), ECE via reliability bins (calibration gap), plus reliability bin details.
- Temperature scaling (`src/validation/calibration.py`): fits scalar T on validation logits using LBFGS; saves `temperature.pt`. Applied as `logits / T` before softmax to improve calibration.
- Evaluation loop (`src/validation/evaluate.py`): collects logits/labels, computes metrics, supports optional temperature scaling at eval time.

## 7. Analysis Playbook (Report Notebooks)
- `00_dataset_analysis.ipynb`: split/class counts, coordinate overlap/leakage checks, duplicates, spatial maps, augmentation previews.
- `01_cnn_activation_maps.ipynb`: Grad-CAMs/activation maps for interpretability.
- `02_train_networks.ipynb`: training sweeps across backbones; writes configs/plots/summaries under `training_runs/{session}_{model}`.
- `03_evaluate_networks.ipynb`: validation/test evaluation with optional temperature scaling; saves `eval_metrics.json` and reliability/confusion/score plots.
- `04_compare_networks.ipynb`: aggregates evaluation outputs to compare accuracy/F1/calibration across models and sessions.
- `05_final_training.ipynb`: final targeted run; exports best checkpoint and temperature for deployment.

All generated figures for the report are stored under `plots/` (organized by notebook/section).

## 8. Repo Map
- `src/config.py`: configuration objects and helpers.
- `src/data/`: transforms and datamodule (ImageFolder + sampler); leak-free resampling script (`resample_dataset.py`).
- `src/models/`: backbone builder + custom CNN.
- `src/training/`: training loop, checkpointing, early stopping.
- `src/validation/`: evaluation, metrics (Brier/ECE), temperature scaling.
- `src/utils/`: logging, reproducibility, checkpoint loader.
- `training_runs/`: outputs from notebook runs (configs, summaries, checkpoints, plots).
- `plots/`: saved figures from EDA and experiment notebooks.
