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

## Calibration & tuning
- Set `apply_temperature=True` to fit a temperature scaler on the best val logits.
- Use `balance_strategy` (`weighted_sampler` or `class_weights`) to handle class imbalance.
- Compose W&B sweeps from the config fields (see `TrainConfig`).
