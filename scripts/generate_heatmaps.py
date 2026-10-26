"""Generate explainability heatmaps (Grad-CAM / Attention Rollout) on sample test images."""

import argparse
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # type: ignore
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from src.data.transforms import get_val_transforms
from src.explainability.explainer import get_explainer
from src.models.detector import DeepfakeDetector
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.visualize import apply_heatmap_overlay, create_gallery_grid

logger = get_logger("generate_heatmaps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Explainability Heatmaps")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of balanced test images to explain (>=12)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found at: {args.config} or {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Checkpoint path
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

    # Load model
    model_name = cfg["model"].get("name", "efficientnet_b0")
    num_classes = cfg["model"].get("num_classes", 2)
    model = DeepfakeDetector(model_name=model_name, pretrained=False, num_classes=num_classes)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Explainer instance
    explainer = get_explainer(model=model, model_name=model_name)

    # Manifest and sample selection
    manifest_raw = cfg["dataset"]["manifest_path"]
    manifest_path = Path(manifest_raw)
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / manifest_raw
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {manifest_raw} or {manifest_path}")

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Select balanced real and fake images
    num_per_class = max(6, args.num_samples // 2)
    real_df = test_df[test_df["label"] == "real"]
    fake_df = test_df[test_df["label"] == "fake"]

    selected_real = real_df.sample(n=min(num_per_class, len(real_df)), random_state=seed)
    selected_fake = fake_df.sample(n=min(num_per_class, len(fake_df)), random_state=seed)
    samples_df = pd.concat([selected_real, selected_fake]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(f"Selected {len(samples_df)} balanced test images for explainability analysis ({len(selected_real)} real, {len(selected_fake)} fake)")

    # Output directory
    out_dir = Path(cfg["outputs"].get("heatmap_dir", "outputs/heatmaps"))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = cfg["dataset"].get("image_size", 224)
    val_tf = get_val_transforms(image_size=img_size)

    gallery_items = []
    scores_records = []

    for idx, row in samples_df.iterrows():
        img_path = Path(row["filepath"])
        if not img_path.exists():
            img_path = PROJECT_ROOT / row["filepath"]
        true_label = str(row["label"])
        pair_id = str(row.get("pair_id", "unknown"))

        if not img_path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            continue

        try:
            # Load raw image
            with Image.open(img_path) as raw_pil:
                raw_rgb = raw_pil.convert("RGB").resize((img_size, img_size))
                orig_np = np.array(raw_rgb)

            # Preprocess tensor for model inference
            tensor_input = val_tf(raw_rgb).unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.softmax(logits, dim=1)[0]
                fake_prob = float(probs[1].item())
                pred_idx = int(torch.argmax(logits, dim=1).item())
                pred_label = "fake" if pred_idx == 1 else "real"

            # Explain: compute heatmap for the predicted class
            heatmap = explainer.explain(tensor_input, class_idx=pred_idx)

            # Generate overlay
            overlay_np = apply_heatmap_overlay(orig_np, heatmap, alpha=0.45, colormap_name="jet")

            # Save individual overlay image
            # Name format: true_pred_prob_idx.png
            save_name = f"sample_{idx:02d}_true_{true_label}_pred_{pred_label}_pfake_{fake_prob:.2f}.png"
            save_path = out_dir / save_name
            Image.fromarray(overlay_np).save(save_path)

            scores_records.append({
                "sample_idx": idx,
                "filename": save_name,
                "filepath": str(img_path).replace("\\", "/"),
                "pair_id": pair_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "fake_probability": fake_prob,
                "is_correct": (true_label.lower() == pred_label.lower()),
            })

            gallery_items.append({
                "orig_image": orig_np,
                "overlay_image": overlay_np,
                "true_label": true_label,
                "pred_label": pred_label,
                "fake_prob": fake_prob,
            })

            logger.info(f"[{idx+1}/{len(samples_df)}] Generated: {save_name}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            continue

    # Save scores CSV
    scores_df = pd.DataFrame(scores_records)
    scores_csv_path = out_dir / "scores.csv"
    scores_df.to_csv(scores_csv_path, index=False)
    logger.info(f"Saved scores CSV to: {scores_csv_path}")

    # Build gallery image
    gallery_path = out_dir / "gallery.png"
    create_gallery_grid(gallery_items, save_path=str(gallery_path), cols=4)
    logger.info(f"Saved combined explainability gallery to: {gallery_path}")

    print("\n" + "=" * 50)
    print("        EXPLAINABILITY HEATMAPS COMPLETED")
    print("=" * 50)
    print(f"Total overlays generated : {len(scores_records)}")
    print(f"Scores CSV saved to       : {scores_csv_path}")
    print(f"Combined gallery image   : {gallery_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
