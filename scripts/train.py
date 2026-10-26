"""Training script for baseline Deepfake Detector."""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import yaml

from src.data.dataset import FaceDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.detector import DeepfakeDetector
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.visualize import plot_confusion_matrix

logger = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline Deepfake Detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Datasets and Loaders
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Please run prepare_ffpp.py first.")

    img_size = cfg["dataset"].get("image_size", 224)
    train_tf = get_train_transforms(image_size=img_size)
    val_tf = get_val_transforms(image_size=img_size)

    train_ds = FaceDataset(manifest_df=manifest_path, split="train", transform=train_tf)
    val_ds = FaceDataset(manifest_df=manifest_path, split="val", transform=val_tf)
    test_ds = FaceDataset(manifest_df=manifest_path, split="test", transform=val_tf)

    batch_size = cfg["training"].get("batch_size", 32)
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Model initialization
    model_name = cfg["model"].get("name", "efficientnet_b0")
    pretrained = cfg["model"].get("pretrained", True)
    num_classes = cfg["model"].get("num_classes", 2)
    dropout = cfg["model"].get("dropout", 0.2)

    logger.info(f"Initializing {model_name} (pretrained={pretrained}, num_classes={num_classes})")
    model = DeepfakeDetector(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
    )

    train_results = trainer.fit()
    logger.info(f"Training completed. Best Val AUC: {train_results['best_val_auc']:.4f}")

    # Evaluate best checkpoint on TEST split once
    logger.info(f"Evaluating best checkpoint on unseen TEST split: {train_results['best_checkpoint']}")
    checkpoint = torch.load(train_results["best_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = trainer.evaluate(test_loader, desc="Test Split")

    # Format and save test metrics
    metrics_json_path = Path(cfg["outputs"]["log_dir"]) / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics JSON to {metrics_json_path}")

    # Plot and save confusion matrix
    import numpy as np
    cm = np.array(test_metrics["confusion_matrix"])
    cm_path = Path(cfg["outputs"]["log_dir"]) / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=["Real", "Fake"], save_path=str(cm_path), title=f"Test Confusion Matrix ({model_name})")
    logger.info(f"Saved test confusion matrix plot to {cm_path}")

    print("\n" + "=" * 50)
    print("           FINAL TEST SPLIT METRICS")
    print("=" * 50)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"F1-Score : {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"Loss     : {test_metrics['loss']:.4f}")
    print("Confusion Matrix (rows: true [Real, Fake], cols: pred [Real, Fake]):")
    print(cm)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
