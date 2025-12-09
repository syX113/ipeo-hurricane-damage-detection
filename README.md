# IPEO: Hurricane Damage Detection

Binary classifier for post-hurricane satellite tiles (`damage` vs `no_damage`) with emphasis on calibrated probabilities.

## Environment
- Python 3.10 recommended.
- Install PyTorch/torchvision first (GPU example: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`; CPU: `pip install torch torchvision`).
- Then install extras: `pip install -r requirements.txt`.
- Add `src` to your `PYTHONPATH` (`export PYTHONPATH=$(pwd)/src`).

## Data
1) Download the ZIP from [Google Drive](https://drive.google.com/file/d/1vWwI8c0tx1DEq9fjeTU53inhhoVmoDYN/view?usp=sharing).
2) Unzip so you have:
```
data/
  train/{damage,no_damage}
  validation/{damage,no_damage}
  test/{damage,no_damage}
```
3) Optional cleaning (recommended to avoid leakage/duplicates):
- Coordinate grouping only: `python src/utils/build_coordinate_split.py --data-root data --output-root data_resampled`
- Full cleaning + optional majority downsampling: `python src/utils/build_clean_dataset.py --data-root data --output-root data_clean --balance-multiplier 1.0 --drop-conflicts`
Point `TrainConfig.data_root` to the chosen cleaned directory.

## Workflow (overview)

```mermaid
graph TD
  A["Raw data in data/{train,validation,test}/damage,no_damage"] --> B["Split inspection\nnotebooks/00_split_inspection.ipynb"]
  B --> C["EDA\nnotebooks/00_exploratory_analysis.ipynb"]
  C --> D{Build cleaned splits}
  D --> D1["data_resampled\nbuild_coordinate_split.py\n(no cross-split coord leakage)"]
  D --> D2["data_clean\nbuild_clean_dataset.py\n(dedup, optional drop conflicts,\noptional downsample majority)"]
  D1 --> E["TrainConfig\nsrc/config.py"]
  D2 --> E
  E --> F["DataModule\ntransforms + sampler\nsrc/data"]
  F --> G["Model builder\nreplace classifier head\n_adapt_first_conv for non-RGB\noptional freeze backbone\nsrc/models"]
  G --> H["Trainer.fit\nCE+label smoothing (+class weights)\nAdamW/SGD + cosine LR\nAMP + grad clip + early stop\nsave best.pt\nsrc/training"]
  H --> I["Val metrics per epoch"]
  H --> J["best.pt checkpoint"]
  J --> K["Temperature scaling on val logits\nLBFGS scalar T\nsrc/validation/calibration.py"]
  K --> L["Eval raw + calibrated\nnotebooks/02_evaluate.ipynb"]
  L --> M["Metrics: accuracy, macro P/R/F1,\nBrier, ECE; plots (confusion,\nscore hist, reliability)"]
  M --> N["Compare runs\nnotebooks/03_comparison.ipynb"]
```

## Quick start (code)
```python
from config import TrainConfig
from training import Trainer

cfg = TrainConfig(
    data_root="data",   # or data_resampled
    model_name="resnet34",
    epochs=20,
    wandb_mode="disabled",
)
trainer = Trainer(cfg)
trainer.fit()
print(trainer.test())
```

## Training details
- Config: `src/config.py` (`TrainConfig`, `build_config_from_dict`).
  - Data/augs: resize, flip, rotation, color jitter; dataset mean/std.
  - Imbalance: `balance_strategy` = `weighted_sampler` (sampler) or `class_weights` (loss).
  - Optimization: AdamW/SGD, cosine LR, label smoothing, grad clip, AMP, early stopping on `macro_f1`.
  - Logging/checkpoints: `checkpoints_dir` (best.pt), optional TensorBoard/W&B.
- Data loaders: `src/data/transforms.py`, `src/data/datamodule.py`.
  - ImageFolder, train/eval transforms, optional integrity summary of coord overlaps/conflicts.
- Models: `src/models/builder.py`, `src/models/custom.py`.
  - Backbones: resnet18/34/50, efficientnet_b0/b1/b2, convnext_tiny/small, or lightweight `custom_cnn`.
  - Head replacement to `num_classes=2` with optional dropout.
  - `_adapt_first_conv` copies pretrained conv1 weights into a new conv when `in_channels` differs (e.g., >3), avoiding random reinit.
  - `freeze_backbone=True` trains only the head (compare frozen vs fine-tune).
- Training loop: `src/training/trainer.py`.
  - Loss = cross-entropy + label smoothing (+ optional class weights).
  - Scheduler = cosine annealing; AMP + GradScaler; gradient clipping.
  - Early stopping on validation `checkpoint_metric`; stores best checkpoint.

## Calibration and metrics
- Metrics (`src/validation/metrics.py`): accuracy, macro precision/recall/F1, Brier score (probability quality), ECE via reliability bins (calibration gap), plus reliability bin details.
- Temperature scaling (`src/validation/calibration.py`): fits scalar T on validation logits using LBFGS; saves `temperature.pt`. Applied as `logits / T` before softmax to improve calibration.
- Evaluation loop (`src/validation/evaluate.py`): collects logits/labels, computes metrics, supports optional temperature scaling at eval time.

## Notebooks
- `00_split_inspection.ipynb`: split/class counts, coord overlaps, near neighbors, conflicts, duplicates, visual samples.
- `00_exploratory_analysis.ipynb`: spatial maps, pixel stats, brightness/contrast, augmentation previews, corruption/duplicate checks.
- `01_train.ipynb`: runs experiments with per-model overrides (resnets, efficientnets, convnexts, custom CNN) and saves configs/plots/summaries under `training_runs/{session}_{model}`.
- `02_evaluate.ipynb`: loads each run, evaluates val/test raw and calibrated, writes `eval_metrics.json` and plots (confusion, score hist, reliability).
- `03_comparison.ipynb`: aggregates runs for a session, compares test macro-F1 and ECE (raw vs calibrated), temperatures, and validation trajectories.

## Repo map
- `src/config.py`: configuration objects and helpers.
- `src/data/`: transforms and datamodule (ImageFolder + sampler).
- `src/models/`: backbone builder + custom CNN.
- `src/training/`: training loop, checkpointing, early stopping.
- `src/validation/`: evaluation, metrics (Brier/ECE), temperature scaling.
- `src/utils/`: logging, reproducibility, checkpoint loader, data cleaning scripts.
- `training_runs/`: outputs from notebook runs (configs, summaries, checkpoints, plots).
- `eda_plots/`: saved figures from EDA notebooks.
