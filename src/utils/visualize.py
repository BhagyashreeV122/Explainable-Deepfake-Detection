"""Visualization utilities for training curves, confusion matrix, and heatmap overlays."""

from pathlib import Path
from typing import List, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
) -> None:
    """Plot and save loss and accuracy curves across training epochs.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accs: List of training accuracies per epoch.
        val_accs: List of validation accuracies per epoch.
        save_path: Destination path for the saved PNG figure.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2)
    ax1.set_title("Epoch Loss Curve", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    ax1_acc = ax2
    ax1_acc.plot(epochs, train_accs, "b-o", label="Train Acc", linewidth=2)
    ax1_acc.plot(epochs, val_accs, "r-s", label="Val Acc", linewidth=2)
    ax1_acc.set_title("Epoch Accuracy Curve", fontsize=14, fontweight="bold")
    ax1_acc.set_xlabel("Epoch", fontsize=12)
    ax1_acc.set_ylabel("Accuracy", fontsize=12)
    ax1_acc.legend(fontsize=11)
    ax1_acc.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: Union[np.ndarray, List],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a stylized confusion matrix.

    Args:
        cm: 2D numpy array or nested list of shape (num_classes, num_classes).
        class_names: Names for class 0 and class 1.
        save_path: Path to save the figure.
        title: Title of the plot.
    """
    cm = np.asarray(cm)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Text annotations inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def apply_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Overlay a normalized 2D heatmap [0, 1] onto an RGB image [0, 255].

    Args:
        image: Original RGB image array of shape (H, W, 3), uint8.
        heatmap: 2D float array of shape (H, W) in [0, 1].
        alpha: Blending weight for heatmap overlay (0 to 1).
        colormap_name: Matplotlib colormap name ('jet', 'viridis', etc.).

    Returns:
        Blended RGB image array of shape (H, W, 3), uint8.
    """
    cmap = plt.get_cmap(colormap_name)
    heatmap_colored = cmap(np.clip(heatmap, 0.0, 1.0))[:, :, :3]  # shape (H, W, 3), float in [0, 1]
    heatmap_rgb = (heatmap_colored * 255).astype(np.uint8)

    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_gallery_grid(
    items: List[dict],
    save_path: str,
    cols: int = 4,
) -> None:
    """Create a structured gallery image of original images side-by-side with explainability overlays.

    Args:
        items: List of dictionaries with keys:
            - 'orig_image': RGB array or PIL Image
            - 'overlay_image': RGB array or PIL Image
            - 'true_label': str
            - 'pred_label': str
            - 'fake_prob': float
        save_path: Destination path for the saved gallery image.
        cols: Number of sample pairs per row.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    num_items = len(items)
    rows = (num_items + cols - 1) // cols

    card_w, card_h = 460, 260
    grid_img = Image.new("RGB", (cols * card_w, rows * card_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(grid_img)

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        x_offset = c * card_w + 10
        y_offset = r * card_h + 10

        orig = item["orig_image"]
        if isinstance(orig, np.ndarray):
            orig = Image.fromarray(orig)
        orig = orig.resize((210, 210))

        overlay = item["overlay_image"]
        if isinstance(overlay, np.ndarray):
            overlay = Image.fromarray(overlay)
        overlay = overlay.resize((210, 210))

        grid_img.paste(orig, (x_offset, y_offset + 25))
        grid_img.paste(overlay, (x_offset + 220, y_offset + 25))

        # Title text
        true_lbl = item.get("true_label", "?")
        pred_lbl = item.get("pred_label", "?")
        fake_prob = item.get("fake_prob", 0.0)
        is_correct = (true_lbl.lower() == pred_lbl.lower())
        color = (100, 255, 100) if is_correct else (255, 100, 100)

        header_text = f"True: {true_lbl} | Pred: {pred_lbl} (p_fake={fake_prob:.2f})"
        draw.text((x_offset + 5, y_offset + 5), header_text, fill=color)

    grid_img.save(save_path, quality=95)
