"""Input-driven inference pipeline and CLI for Deepfake Detection & Explainability."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # type: ignore
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml

from src.data.transforms import get_val_transforms
from src.explainability.explainer import BaseExplainer, get_explainer
from src.models.detector import DeepfakeDetector
from src.utils.logger import get_logger
from src.utils.visualize import apply_heatmap_overlay

logger = get_logger("inference")


def get_default_font(size: int = 14) -> ImageFont.ImageFont:
    """Load a clean truetype font if available, else default."""
    try:
        # Standard fonts available on Windows
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


class InferencePipeline:
    """Complete inference and explainability pipeline for user-provided images."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        crop_margin: float = 0.25,
        target_size: int = 224,
    ) -> None:
        """Initialize detector model, explainability module, and face detector.

        Args:
            config_path: Path to config.yaml.
            checkpoint_path: Path to model checkpoint .pt file.
            device: 'cuda', 'cpu', or None (auto-detect).
            crop_margin: Margin fraction around detected face bounding box.
            target_size: Spatial dimension for model input (default 224).
        """
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")

        # Checkpoint resolution
        if checkpoint_path is None:
            checkpoint_path = str(
                Path(self.cfg.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints")) / "best_model.pt"
            )
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Load detector model
        model_name = self.cfg.get("model", {}).get("name", "efficientnet_b0")
        num_classes = self.cfg.get("model", {}).get("num_classes", 2)
        dropout = self.cfg.get("model", {}).get("dropout", 0.2)

        self.model = DeepfakeDetector(
            model_name=model_name,
            pretrained=False,
            num_classes=num_classes,
            dropout=dropout,
        )
        logger.info(f"Loading checkpoint weights from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Explainer module
        self.model_name = model_name
        self.explainer: BaseExplainer = get_explainer(model=self.model, model_name=model_name)

        # Transforms & parameters
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.val_transform = get_val_transforms(image_size=self.target_size)

        # OpenCV Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning(f"Could not load OpenCV Haar cascade from {cascade_path}")

    def detect_and_crop_face(
        self,
        rgb_image: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        """Detect the largest face in an RGB image and crop with margin.

        If no face is detected, returns the full image resized to target_size and face_detected=False.

        Args:
            rgb_image: RGB numpy array of shape (H, W, 3), uint8.

        Returns:
            Tuple of (cropped_face_rgb_224, face_detected_bool, bbox_or_None).
        """
        H, W = rgb_image.shape[:2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Standard face detection pass
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(40, 40),
        )

        # Fallback to slightly more permissive parameters if no face found
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(25, 25),
            )

        if len(faces) == 0:
            # Fallback to full image
            fallback_crop = cv2.resize(rgb_image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            return fallback_crop, False, None

        # Pick largest face by area
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]

        cx = x + w / 2.0
        cy = y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * self.crop_margin)

        x1 = max(0, int(cx - side / 2.0))
        y1 = max(0, int(cy - side / 2.0))
        x2 = min(W, int(cx + side / 2.0))
        y2 = min(H, int(cy + side / 2.0))

        crop = rgb_image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            fallback_crop = cv2.resize(rgb_image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            return fallback_crop, False, None

        face_crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        return face_crop, True, (x1, y1, x2, y2)

    def create_combined_panel(
        self,
        orig_rgb: np.ndarray,
        face_crop_rgb: np.ndarray,
        overlay_rgb: np.ndarray,
        verdict: str,
        prob_real: float,
        prob_fake: float,
        face_detected: bool,
        image_name: str,
    ) -> np.ndarray:
        """Create a stylized side-by-side panel (Original | Face Crop | Heatmap Overlay)."""
        card_h = 240
        banner_h = 100
        padding = 15

        # Resize original maintaining aspect ratio to fit inside (card_w, card_h)
        orig_pil = Image.fromarray(orig_rgb)
        orig_w, orig_h = orig_pil.size
        scale = min(card_h / orig_h, 320 / orig_w)
        scaled_w, scaled_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
        orig_fitted = orig_pil.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)

        panel1 = Image.new("RGB", (320, card_h), color=(30, 30, 30))
        offset_x = (320 - scaled_w) // 2
        offset_y = (card_h - scaled_h) // 2
        panel1.paste(orig_fitted, (offset_x, offset_y))

        # Panel 2: Face crop (224 -> 240)
        crop_pil = Image.fromarray(face_crop_rgb).resize((240, card_h), Image.Resampling.BILINEAR)
        # Panel 3: Heatmap overlay (224 -> 240)
        overlay_pil = Image.fromarray(overlay_rgb).resize((240, card_h), Image.Resampling.BILINEAR)

        total_w = 320 + 240 + 240 + (padding * 4)
        total_h = banner_h + card_h + 40  # banner + images + label footer

        canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 20))
        draw = ImageDraw.Draw(canvas)

        font_title = get_default_font(18)
        font_sub = get_default_font(13)
        font_verdict = get_default_font(22)

        # Header Title
        draw.text((padding, 12), f"Deepfake Detection Analysis: {image_name}", fill=(240, 240, 240), font=font_title)

        # Verdict badge styling
        is_fake = (verdict.upper() == "FAKE")
        verdict_color = (255, 75, 75) if is_fake else (50, 215, 75)
        verdict_bg = (60, 15, 15) if is_fake else (10, 45, 15)

        # Badge pill
        badge_w, badge_h = 160, 34
        badge_x = padding
        badge_y = 42
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=6, fill=verdict_bg, outline=verdict_color, width=2)
        draw.text((badge_x + 14, badge_y + 4), f"VERDICT: {verdict}", fill=verdict_color, font=font_verdict)

        # Probabilities
        prob_text = f"P(Real): {prob_real * 100:.1f}%    |    P(Fake): {prob_fake * 100:.1f}%"
        draw.text((badge_x + badge_w + 20, badge_y + 8), prob_text, fill=(220, 220, 220), font=font_title)

        # Face Warning if not detected
        if not face_detected:
            warn_text = "⚠️ WARNING: No face detected. Model evaluated on resized full image."
            draw.text((padding, 80), warn_text, fill=(255, 190, 40), font=font_sub)

        # Paste the 3 panels
        y_pos = banner_h
        x1 = padding
        canvas.paste(panel1, (x1, y_pos))

        x2 = x1 + 320 + padding
        canvas.paste(crop_pil, (x2, y_pos))

        x3 = x2 + 240 + padding
        canvas.paste(overlay_pil, (x3, y_pos))

        # Footer subtitles
        draw.text((x1 + 10, y_pos + card_h + 8), "Original Image", fill=(170, 170, 170), font=font_sub)
        crop_label = "Face Crop (224x224)" if face_detected else "Full Image Fallback (224x224)"
        draw.text((x2 + 10, y_pos + card_h + 8), crop_label, fill=(170, 170, 170), font=font_sub)
        draw.text((x3 + 10, y_pos + card_h + 8), f"Grad-CAM Heatmap ({verdict})", fill=(170, 170, 170), font=font_sub)

        return np.array(canvas)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        image_name: Optional[str] = None,
        save_outputs: bool = True,
        output_dir: Union[str, Path] = "outputs/inference",
    ) -> Dict:
        """Run complete detection and explainability on an image.

        Args:
            image_input: Filepath string/Path, RGB numpy array, or PIL Image.
            image_name: Optional custom identifier for saving outputs.
            save_outputs: Whether to save artifacts to disk and append to results.csv.
            output_dir: Directory where outputs are saved.

        Returns:
            Dictionary containing prediction scores, heatmap, and visual arrays.
        """
        output_dir = Path(output_dir)
        source_path = None

        # 1. Load image to RGB numpy array
        if isinstance(image_input, (str, Path)):
            source_path = str(image_input)
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            if image_name is None:
                image_name = img_path.stem
            with Image.open(img_path) as pil_img:
                orig_rgb = np.array(pil_img.convert("RGB"))
        elif isinstance(image_input, Image.Image):
            orig_rgb = np.array(image_input.convert("RGB"))
            if image_name is None:
                image_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif isinstance(image_input, np.ndarray):
            orig_rgb = image_input.copy()
            if orig_rgb.dtype != np.uint8:
                orig_rgb = (np.clip(orig_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            if image_name is None:
                image_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        # 2. Face detection and cropping
        face_crop_rgb, face_detected, bbox = self.detect_and_crop_face(orig_rgb)
        if not face_detected:
            warning_msg = f"[WARNING] No face detected in '{image_name}'! Falling back to full image."
            logger.warning(warning_msg)
            print(f"\n>>> {warning_msg}")
        else:
            warning_msg = None
            logger.info(f"Face detected in '{image_name}' at bbox: {bbox}")

        # 3. Model inference
        face_pil = Image.fromarray(face_crop_rgb)
        tensor_input = self.val_transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)[0]
            prob_real = float(probs[0].item())
            prob_fake = float(probs[1].item())

        pred_idx = 1 if prob_fake >= 0.5 else 0
        verdict = "FAKE" if pred_idx == 1 else "REAL"

        # 4. Explainability: generate heatmap for predicted class
        heatmap = self.explainer.explain(tensor_input, class_idx=pred_idx)
        # Ensure heatmap is exactly (224, 224)
        if heatmap.shape != (self.target_size, self.target_size):
            heatmap = cv2.resize(heatmap, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # Overlay heatmap on face crop
        heatmap_overlay = apply_heatmap_overlay(face_crop_rgb, heatmap, alpha=0.45, colormap_name="jet")

        # 5. Combined panel
        combined_panel = self.create_combined_panel(
            orig_rgb=orig_rgb,
            face_crop_rgb=face_crop_rgb,
            overlay_rgb=heatmap_overlay,
            verdict=verdict,
            prob_real=prob_real,
            prob_fake=prob_fake,
            face_detected=face_detected,
            image_name=image_name,
        )

        saved_files = {}
        # 6. Save outputs if requested
        if save_outputs:
            target_folder = output_dir / image_name
            target_folder.mkdir(parents=True, exist_ok=True)

            orig_path = target_folder / "original.png"
            face_crop_path = target_folder / "face_crop.png"
            overlay_path = target_folder / "heatmap_overlay.png"
            panel_path = target_folder / "combined_panel.png"

            Image.fromarray(orig_rgb).save(orig_path)
            Image.fromarray(face_crop_rgb).save(face_crop_path)
            Image.fromarray(heatmap_overlay).save(overlay_path)
            Image.fromarray(combined_panel).save(panel_path)

            saved_files = {
                "folder": str(target_folder),
                "original": str(orig_path),
                "face_crop": str(face_crop_path),
                "heatmap_overlay": str(overlay_path),
                "combined_panel": str(panel_path),
            }

            # Append to results.csv
            results_csv_path = output_dir / "results.csv"
            row_data = {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "source_path": source_path or "in-memory",
                "face_detected": face_detected,
                "verdict": verdict,
                "prob_real": round(prob_real, 4),
                "prob_fake": round(prob_fake, 4),
                "combined_panel": str(panel_path).replace("\\", "/"),
            }
            results_df = pd.DataFrame([row_data])
            if results_csv_path.exists():
                results_df.to_csv(results_csv_path, mode="a", header=False, index=False)
            else:
                results_df.to_csv(results_csv_path, mode="w", header=True, index=False)
            logger.info(f"Saved results and artifacts to {target_folder}")

        return {
            "image_name": image_name,
            "source_path": source_path,
            "face_detected": face_detected,
            "warning": warning_msg,
            "verdict": verdict,
            "probabilities": {
                "real": prob_real,
                "fake": prob_fake,
            },
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "heatmap": heatmap,
            "face_crop": face_crop_rgb,
            "heatmap_overlay": heatmap_overlay,
            "original_image": orig_rgb,
            "combined_panel": combined_panel,
            "saved_files": saved_files,
        }

    def predict_folder(
        self,
        folder_path: Union[str, Path],
        output_dir: Union[str, Path] = "outputs/inference",
    ) -> List[Dict]:
        """Process all images in a directory, log and continue on failures, and create gallery."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {folder}")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
        image_files = sorted([p for p in folder.iterdir() if p.suffix in valid_exts])

        logger.info(f"Found {len(image_files)} image files in {folder}")
        if not image_files:
            logger.warning(f"No valid image files found in {folder}")
            return []

        results = []
        gallery_items = []

        for idx, img_p in enumerate(image_files, 1):
            try:
                res = self.predict(image_input=img_p, save_outputs=True, output_dir=output_dir)
                results.append(res)
                gallery_items.append({
                    "orig_image": res["face_crop"],
                    "overlay_image": res["heatmap_overlay"],
                    "true_label": res["image_name"][:12],
                    "pred_label": res["verdict"],
                    "fake_prob": res["prob_fake"],
                })
                print(
                    f"[{idx}/{len(image_files)}] {img_p.name} -> {res['verdict']} "
                    f"(P(real)={res['prob_real']:.3f}, P(fake)={res['prob_fake']:.3f})"
                )
            except Exception as e:
                logger.error(f"Failed processing image {img_p.name}: {e}", exc_info=True)
                print(f"[{idx}/{len(image_files)}] [ERROR] Skipping {img_p.name}: {e}")
                continue

        # Build combined gallery image
        if gallery_items:
            gallery_path = Path(output_dir) / "gallery.png"
            from src.utils.visualize import create_gallery_grid
            create_gallery_grid(gallery_items, save_path=str(gallery_path), cols=min(4, len(gallery_items)))
            logger.info(f"Gallery grid saved to: {gallery_path}")
            print(f"\nGallery successfully saved to: {gallery_path}")

        return results


def predict_single_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    pipeline: Optional[InferencePipeline] = None,
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    save_outputs: bool = False,
    output_dir: str = "outputs/inference",
) -> Dict:
    """Convenience function for predicting on a single image.

    Args:
        image_input: Path, PIL Image, or RGB numpy array.
        pipeline: Existing InferencePipeline instance (reused if provided).
        config_path: Path to config.yaml.
        checkpoint_path: Path to checkpoint .pt file.
        save_outputs: Whether to save panel and overlay to disk.
        output_dir: Target output directory.

    Returns:
        Prediction dictionary with verdict, probabilities, and heatmap.
    """
    if pipeline is None:
        pipeline = InferencePipeline(config_path=config_path, checkpoint_path=checkpoint_path)
    return pipeline.predict(image_input=image_input, save_outputs=save_outputs, output_dir=output_dir)


def run_cli() -> None:
    """Entrypoint for inference command-line interface."""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection and Explainability Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file (JPG/PNG)")
    group.add_argument("--folder", type=str, help="Path to a directory containing images")

    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="outputs/inference", help="Directory to save inference artifacts")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")

    args = parser.parse_args()

    pipeline = InferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.image:
        res = pipeline.predict(image_input=args.image, save_outputs=True, output_dir=args.output_dir)
        print("\n" + "=" * 55)
        print("               INFERENCE RESULT")
        print("=" * 55)
        print(f"Image File       : {res['source_path']}")
        print(f"Face Detected    : {'YES' if res['face_detected'] else 'NO (Full image fallback used)'}")
        if res.get("warning"):
            print(f"Warning          : {res['warning']}")
        print(f"VERDICT          : {res['verdict']}")
        print(f"P(Real)          : {res['prob_real'] * 100:.2f}%")
        print(f"P(Fake)          : {res['prob_fake'] * 100:.2f}%")
        print("-" * 55)
        print("Saved Outputs    :")
        for k, v in res.get("saved_files", {}).items():
            print(f"  • {k.ljust(16)}: {v}")
        print("=" * 55 + "\n")

    elif args.folder:
        results = pipeline.predict_folder(folder_path=args.folder, output_dir=args.output_dir)
        print("\n" + "=" * 55)
        print(f"  BATCH INFERENCE COMPLETE: {len(results)} images processed")
        print(f"  Outputs saved in: {args.output_dir}")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    run_cli()
