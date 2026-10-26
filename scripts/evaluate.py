"""Standalone checkpoint evaluation script."""

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
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import compute_metrics
from src.models.detector import DeepfakeDetector
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.visualize import plot_confusion_matrix

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector Checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--save", action="store_true", help="Save evaluation metrics JSON and confusion matrix plot")
    parser.add_argument("--output-dir", type=str, default="outputs/logs", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found at: {args.config} or {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation device: {device}")

    # Checkpoint path resolution
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            checkpoint_path = PROJECT_ROOT / args.checkpoint
    else:
        checkpoint_path = PROJECT_ROOT / cfg["outputs"].get("checkpoint_dir", "outputs/checkpoints") / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = Path(cfg["outputs"].get("checkpoint_dir", "outputs/checkpoints")) / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Dataset & loader
    manifest_raw = cfg["dataset"]["manifest_path"]
    manifest_path = Path(manifest_raw)
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / manifest_raw
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {manifest_raw} or {manifest_path}")

    val_tf = get_val_transforms(image_size=cfg["dataset"].get("image_size", 224))
    dataset = FaceDataset(manifest_df=manifest_path, split=args.split, transform=val_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loaded {len(dataset)} samples for split '{args.split}'")

    # Model
    model_name = cfg["model"].get("name", "efficientnet_b0")
    model = DeepfakeDetector(model_name=model_name, pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_labels = []
    all_fake_probs = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy().tolist())
            all_fake_probs.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_fake_probs)

    print("\n" + "=" * 50)
    print(f"        EVALUATION RESULTS ({args.split.upper()} SPLIT)")
    print("=" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"AUC      : {metrics['auc']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 50 + "\n")

    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_save_path = out_dir / f"{args.split}_metrics.json"
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics JSON to: {metrics_save_path}")

        cm_save_path = out_dir / f"{args.split}_confusion_matrix.png"
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=["Real", "Fake"],
            save_path=str(cm_save_path),
            title=f"Confusion Matrix ({args.split.upper()} Split)",
        )
        logger.info(f"Saved confusion matrix plot to: {cm_save_path}")


if __name__ == "__main__":
    main()
