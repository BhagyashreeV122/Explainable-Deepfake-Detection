"""Model training and validation loop manager."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger
from src.utils.visualize import plot_training_curves

logger = get_logger("trainer")


class Trainer:
    """Trainer orchestrating AdamW optimization, cosine LR scheduling, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            config: Full experiment configuration dictionary.
            device: torch.device (cuda or cpu).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 5)
        self.lr = float(train_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))

        # Mixed precision setup
        self.use_amp = (
            train_cfg.get("mixed_precision", True)
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)

        # Output paths
        outputs_cfg = config.get("outputs", {})
        self.checkpoint_dir = Path(outputs_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir = Path(outputs_cfg.get("log_dir", "outputs/logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_val_auc = -1.0
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (epoch_loss, epoch_accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "Val") -> Dict[str, Any]:
        """Evaluate the model on a given data loader.

        Returns:
            Dictionary with loss and evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total = 0
        all_labels = []
        all_fake_probs = []

        pbar = tqdm(data_loader, desc=desc, leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_fake_probs.extend(probs.cpu().numpy().tolist())

        avg_loss = total_loss / total
        metrics = compute_metrics(all_labels, all_fake_probs)
        metrics["loss"] = avg_loss
        return metrics

    def fit(self) -> Dict[str, Any]:
        """Run complete training across all epochs with validation, logging, and best checkpoint saving."""
        logger.info(f"Starting training on device: {self.device} (AMP enabled: {self.use_amp})")
        logger.info(f"Epochs: {self.epochs}, Initial LR: {self.lr}, Batch size: {self.train_loader.batch_size}")

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, self.epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_auc = val_metrics["auc"]
            val_f1 = val_metrics["f1"]

            self.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
            }
            self.history.append(epoch_record)

            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} - LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}"
            )

            # Model selection: best checkpoint by validation AUC
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_auc": self.best_val_auc,
                        "config": self.config,
                    },
                    self.best_checkpoint_path,
                )
                logger.info(f"[*] New best model saved (Val AUC: {val_auc:.4f}) to {self.best_checkpoint_path}")

        # Save history CSV
        history_csv_path = self.log_dir / "training_history.csv"
        pd.DataFrame(self.history).to_csv(history_csv_path, index=False)
        logger.info(f"Training history saved to {history_csv_path}")

        # Save training curves plot
        curves_png_path = self.log_dir / "curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, str(curves_png_path))
        logger.info(f"Training loss/accuracy curves saved to {curves_png_path}")

        return {
            "best_val_auc": self.best_val_auc,
            "best_checkpoint": str(self.best_checkpoint_path),
            "history": self.history,
        }
