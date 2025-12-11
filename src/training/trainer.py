import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import TrainConfig
from data.datamodule import DataModule
from models.builder import build_model
from validation.evaluate import evaluate
from validation.calibration import TemperatureScaler
from utils.logging import get_logger, init_wandb
from utils.paths import resolve_project_root
from utils.reproducibility import set_seed as set_global_seed


class Trainer:
    """
    Train → validate → optional temperature scaling → test.
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        set_global_seed(seed)

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(cfg.seed)
        self.logger = get_logger()
        self.run = init_wandb(cfg)
        self.project_root = resolve_project_root()

        # Build dataloaders
        self.datamodule = DataModule(cfg)
        self.datamodule.setup()
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()

        # Build model and loss
        self.model = build_model(cfg).to(self.device)
        class_weights = self._build_class_weights()
        self.loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=cfg.label_smoothing)

        # Cosine schedule by default for decay
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=1e-6)
        self.scaler = torch.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

        self.best_metric = -float("inf")
        self.best_val_outputs: Optional[Dict] = None
        self.temperature: Optional[TemperatureScaler] = None

        ckpt_dir = Path(cfg.checkpoints_dir).expanduser()
        # Keep artifacts under the project root regardless of working directory
        self.checkpoints_dir = ckpt_dir if ckpt_dir.is_absolute() else (self.project_root / ckpt_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = None
        if cfg.tensorboard:
            if SummaryWriter is None:
                self.logger.warning("TensorBoard enabled but torch.utils.tensorboard is unavailable. Install tensorboard to log events.")
            else:
                tb_dir_base = Path(cfg.tensorboard_dir).expanduser() if cfg.tensorboard_dir else Path("training_runs") / "tensorboard"
                tb_dir_base = tb_dir_base if tb_dir_base.is_absolute() else (self.project_root / tb_dir_base)
                run_name = cfg.wandb_run_name or f"{cfg.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                tb_run_dir = tb_dir_base / run_name
                tb_run_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_run_dir), filename_suffix=f"-{run_name}")

    def _build_class_weights(self):
        if self.cfg.balance_strategy != "class_weights":
            return None
        # Inverse-frequency weighting for class imbalance
        counts = {}
        for label in self.train_loader.dataset.targets:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        weights = [total / counts[c] for c in sorted(counts)]
        return torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.betas)
        if self.cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        raise ValueError(f"Unsupported optimizer {self.cfg.optimizer}")

    def _iter_loader(self, loader):
        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch
            # Move tensors to device lazily to keep loop concise
            yield images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in self._iter_loader(self.train_loader):
            self.optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.cfg.amp and self.device.type == "cuda",
            ):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        self.scheduler.step()
        return total_loss / max(1, total_samples)

    def fit(self):
        history = []
        epochs_no_improve = 0
        best_epoch = -1

        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics, val_outputs = evaluate(self.model, self.val_loader, self.device, self.cfg, temperature=None)
            val_logits = val_outputs["logits"]
            val_labels = val_outputs["labels"]
            val_loss = self.loss_fn(val_logits, val_labels).item()

            payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.run:
                self.run.log(payload)
            self.logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_f1={val_metrics['macro_f1']:.3f}")

            if self.tb_writer:
                self.tb_writer.add_scalar("loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("loss/val", val_loss, epoch)
                for k, v in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{k}", v, epoch)

            history.append(payload)
            metric = val_metrics.get(self.cfg.checkpoint_metric)
            if metric is not None and metric > self.best_metric:
                self.best_metric = metric
                self.best_val_outputs = {
                    "logits": val_logits.detach().cpu(),
                    "labels": val_labels.detach().cpu(),
                    "probs": val_outputs["probs"],
                }
                best_epoch = epoch
                epochs_no_improve = 0
                self._save_checkpoint(epoch)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.cfg.early_stopping:
                self.logger.info("Early stopping.")
                break

        if best_epoch >= 0:
            self._load_best_checkpoint()
        else:
            self.logger.warning("No checkpoint saved; using last epoch weights.")

        if self.cfg.apply_temperature and self.best_val_outputs:
            self._fit_temperature(self.best_val_outputs["logits"], self.best_val_outputs["labels"])

        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()

        return {"history": history, "best_metric": self.best_metric, "best_epoch": best_epoch}

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_path = self.checkpoints_dir / "best.pt"
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(), "config": self.cfg.to_dict()}, ckpt_path)
        if self.run:
            self.run.save(str(ckpt_path))

    def _load_best_checkpoint(self):
        ckpt_path = self.checkpoints_dir / "best.pt"
        if not ckpt_path.exists():
            self.logger.warning(f"Checkpoint not found at {ckpt_path}")
            return None
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        return state

    def _fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.temperature = TemperatureScaler().to(self.device)
        self.temperature.fit(logits.to(self.device), labels.to(self.device))
        temp_path = self.checkpoints_dir / "temperature.pt"
        self.temperature.save(str(temp_path))
        if self.run:
            self.run.summary["temperature"] = float(self.temperature.temperature.item())
            self.run.save(str(temp_path))

    def test(self) -> Dict:
        temp = getattr(self, "temperature", None) if self.cfg.apply_temperature else None
        metrics, _ = evaluate(self.model, self.test_loader, self.device, self.cfg, temperature=temp)
        if self.run:
            self.run.log({f"test/{k}": v for k, v in metrics.items()})
        return metrics
