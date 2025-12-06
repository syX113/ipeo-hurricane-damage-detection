# IPEO: Hurricane Damage Detection

## Data Setup
Download the data ZIP from [Google Drive](https://drive.google.com/file/d/1vWwI8c0tx1DEq9fjeTU53inhhoVmoDYN/view?usp=sharing) and unzip it so `train/`, `validation/`, and `test/` folders are under `data/` as:
```
data/
  train/damage
  train/no_damage
  validation/damage
  validation/no_damage
  test/damage
  test/no_damage
```

## Python Setup

*TODO* CHECK THIS!

**Setup with Conda**
1. *GPU (CUDA 11.8)*: `conda create -n c-venv-ipeo-hurricane python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia`  
**or**
2. *CPU-only*: `conda create -n c-venv-ipeo-hurricane python=3.10 pytorch torchvision cpuonly -c pytorch`
3. Activate environment: `conda activate c-venv-ipeo-hurricane`
4. Install the rest of the dependencies: `pip install -r requirements.txt`

If only `pip` is available, add the respective URLs and use pip:

**Setup with pip only**
- CPU (default PyPI wheels): `pip3 install torch torchvision`
- GPU (CUDA 11.8): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Then `pip3 install -r requirements.txt`

## Quick start
Add `src` to your `PYTHONPATH` (e.g., `export PYTHONPATH=./src` or `sys.path.insert(0, "src")`) so the modules import correctly.
```python
from config import TrainConfig
from training import Trainer

cfg = TrainConfig(data_root="data", epochs=5, wandb_mode="disabled")
trainer = Trainer(cfg)
trainer.fit()
print(trainer.test())
```

## Module overview
- `config`: `TrainConfig` and `build_config_from_dict` for easy overrides.
- `data`: ImageFolder datamodule + transforms.
- `models`: torchvision builders (swap via `model_name`: resnet18/34, efficientnet_b0, convnext_tiny).
- `training`: Trainer orchestrating train/val/test, checkpoints, optional temperature scaling.
- `validation`: metrics (accuracy/F1/Brier/ECE), evaluation loop, reliability bins, temperature scaler.
- `utils`: logging, reproducibility, checkpoint loader.

## Key classes & interactions
- `TrainConfig`: single source of truth for data, model, and optimization knobs; passed everywhere to keep runs reproducible.
- `DataModule`: builds train/val/test `DataLoader`s with augmentations and optional class balancing; consumed by `Trainer`.
- Model builders (`models.builder.build_model`): instantiate the chosen backbone and replace the classification head according to `cfg.num_classes`.
- `Trainer`: drives epochs, logging, checkpointing, early stopping, and calls `evaluate` on val/test splits.
- `TemperatureScaler` (`validation.calibration`): fits a scalar on validation logits to improve probability calibration; applied before test metrics and exported as `artifacts/checkpoints/temperature.pt` for reuse in inference.
- `utils.checkpoint.load_checkpoint`: recreates a model + config from `artifacts/checkpoints/best.pt` for evaluation or inference.

## Model training/approach
1. Prepared Image folder data (`damage`, `no_damage`) into `train/`, `validation/`, `test/`.
2. Choose a backbone via `cfg.model_name` (i.e.: `resnet18`, `resnet34`, `efficientnet_b0`, `convnext_tiny`) and set common hyperparameters (LR, weight decay, epochs, augmentations, balancing).
3. Train with cross-entropy + label smoothing, mixed precision, weighted sampling or class weights, cosine LR decay, gradient clipping, and optional early stopping.
4. Pick the best checkpoint by validation macro-F1, then fit a temperature scaler on the saved val logits to calibrate probabilities.
5. Evaluate the calibrated model on the test split; compare metrics (macro-F1, accuracy, Brier, ECE, confusion matrices) across the four backbones to select the best trade-off.

## Calibration & tuning
- Set `apply_temperature=True` to fit a temperature scaler on the best val logits.
- Use `balance_strategy` (`weighted_sampler` or `class_weights`) to handle class imbalance.
- Compose W&B sweeps from the config fields (see `TrainConfig`).
