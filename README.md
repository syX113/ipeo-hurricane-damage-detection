# IPEO: Hurricane Damage Detection
Binary classification for post-hurricane satellite tiles via deep learning (`damage` vs `no_damage`) with calibrated probabilities and extensive dataset analysis.
The project is based on the paper [*Building Damage Annotation on Post-Hurricane Satellite Imagery Based on Convolutional Neural Networks*](https://arxiv.org/abs/1807.01688): 

## Quick Links
- Project Report (PDF): [docs/Report - Hurricane Damage Detection.pdf](docs/Report%20-%20Hurricane%20Damage%20Detection.pdf)
- Inference Notebook: [inference/inference.ipynb](inference/inference.ipynb)
- Detailed Notebooks: [notebooks/](notebooks/) (EDA, Grad CAMS and leakage experiments, model training, evaluation, comparisons, hyperparameter tuning)

## Table of Contents
1. [Overview](#1-overview)
2. [Report Snapshot](#2-report-snapshot)
3. [Run Inference](#3-run-inference)
4. [Development Setup](#4-development-setup)
5. [Datasets](#5-datasets)
6. [Train a Model](#6-train-a-model)
7. [Training and Calibration Details](#7-training-and-calibration-details)
8. [Analysis Playbook](#8-analysis-playbook)
9. [Repo Map](#9-repo-map)

## 1. Overview
Detect building damage after hurricanes while auditing leakage and calibration. The repo includes a resampled dataset (via Google Drive), various notebooks, and calibration-aware evaluation.

## 2. Report Snapshot
The report shows that the original Harvey dataset leaks pre/post-event cues and reuses coordinates; a block-based resample collapses duplicates to 14'223 images (Train 12'600 | Val 495 | Test 1'128) to limit overlap. On the original dataset, near-perfect scores are likely shortcut-driven: ConvNeXt-Tiny reaches Macro-F1/Acc. 0.990/0.990, and even pixel-shuffled tiles (geometry destroyed, colors intact) reach ~0.76 Macro-F1. On the resampled dataset, performance drops to more realistic levels (ResNet-50 Macro-F1 0.891, Acc. 0.922); validation-fitted temperature scaling often worsens ECE, and shifting color statistics toward the no_damage domain flips ~35% of damage predictions.

## 3. Quickstart (Inference Notebook)
1. Create the inference env: `conda env create -f inference/environment.yml && conda activate hurricane-inference`.
2. Open `inference/inference.ipynb` in VS Code/Jupyter from the repo root.
3. Run cells sequentially. The notebook downloads checkpoints and zipped datasets to `artifacts/` and unpacks `data/` + `data_resampled/` automatically.
4. Outputs: validation/test metrics for time-detectors and CNN baselines, plus Grad-CAM overlays that mirror the report (use provided sample filenames or pick any from `data_resampled/test`).

## 4. Development Setup 
- Python 3.10 recommended.
- Install PyTorch/torchvision first (GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`; CPU: `pip install torch torchvision`).
- Then install project deps: `pip install -r requirements.txt`.
- Add `src` to `PYTHONPATH`: `export PYTHONPATH=$(pwd)/src`.

## 5. Datasets
Two datasets (pulled automatically by the inference notebook):
- `data.zip` -> `data/`: Original Hurricane Harvey tiles with the published train/val/test split.
- `data_resampled.zip` -> `data_resampled/`: Leak-reduced via block-based coordinate grouping (keep-one-per-location).

Manual download (if skipping the notebook):
1. Download both ZIPs from the shared [Drive folder](https://drive.google.com/drive/folders/1fFiyF_eVUCPRggc8Ti4RAl3zCMj8vAhL?usp=drive_link).
2. Unzip at the repo root to obtain:
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
Regenerate the resampled set (for different splits/precision): `python src/data/resample_dataset.py --data-root data --output-root data_resampled --val-split 0.15 --test-split 0.15 --precision 1`

Set `TrainConfig.data_root` to `data/` (original) or `data_resampled/` (cleaned); relative paths resolve from the repo root.

## 6. Train a Model
```python
from config import TrainConfig
from training import Trainer

cfg = TrainConfig(
    data_root="data_resampled",  # or "data"
    model_name="resnet34",
    epochs=20,
    data_integrity_check=True,
    wandb_mode="disabled",
)
trainer = Trainer(cfg)
trainer.fit()
print(trainer.test())
```
Set `data_integrity_check=True` to log coordinate overlap and class counts during datamodule setup.

## 7. Training and Calibration Details
- Config: `src/config.py` (`TrainConfig`, `build_config_from_dict`); data augs (resize/flip/rotation/color jitter), dataset mean/std.
- Imbalance handling: `balance_strategy` = `weighted_sampler` (sampler) or `class_weights` (loss).
- Optimization: AdamW/SGD, cosine LR, label smoothing, grad clip, AMP, early stopping on `macro_f1`; checkpoints resolved from repo root, optional TensorBoard/W&B.
- Data loaders: `src/data/transforms.py`, `src/data/datamodule.py` (ImageFolder, train/eval transforms, optional integrity summary).
- Models: `src/models/builder.py`, `src/models/custom.py` - resnet18/34/50, efficientnet_b0/b1/b2, convnext_tiny/small, or `custom_cnn`; head replaced to `num_classes=2`, optional dropout, `_adapt_first_conv` handles >3 channels, `freeze_backbone` trains only the head.
- Metrics (`src/validation/metrics.py`): accuracy, macro precision/recall/F1, Brier, ECE via reliability bins.
- Calibration (`src/validation/calibration.py`): validation-fitted temperature scaling (`logits / T`); apply during eval via `src/validation/evaluate.py` (collects logits/labels, optional temperature scaling).

## 8. Analysis Playbook
- `00_dataset_analysis.ipynb`: split/class counts, coordinate overlap/leakage checks, duplicates, spatial maps, augmentation previews.
- `01_cnn_activation_maps.ipynb`: Grad-CAMs/activation maps for interpretability.
- `02_train_networks.ipynb`: training sweeps across backbones; writes configs/plots/summaries under `training_runs/{session}_{model}`.
- `03_evaluate_networks.ipynb`: validation/test evaluation with optional temperature scaling; saves `eval_metrics.json` and reliability/confusion/score plots.
- `04_compare_networks.ipynb`: aggregates evaluation outputs to compare accuracy/F1/calibration across models and sessions.
- `05_final_training.ipynb`: final targeted run; exports best checkpoint and temperature for deployment.
All generated figures for the report are stored under `plots/` (organized by notebook/section).

## 9. Repo Map
- `src/config.py`: configuration objects and helpers.
- `src/data/`: transforms and datamodule (ImageFolder + sampler); leak-aware resampling script (`resample_dataset.py`).
- `src/models/`: backbone builder + custom CNN.
- `src/training/`: training loop, checkpointing, early stopping.
- `src/validation/`: evaluation, metrics (Brier/ECE), temperature scaling.
- `src/utils/`: logging, reproducibility, checkpoint loader.
- `training_runs/`: outputs from notebook runs (configs, summaries, checkpoints, plots).
- `plots/`: saved figures from EDA and experiment notebooks.
