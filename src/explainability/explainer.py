"""Unified explainability interface supporting CNNs (Grad-CAM) and ViTs (Attention Rollout)."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from src.explainability.gradcam import GradCAM
from src.explainability.rollout import AttentionRollout
from src.utils.logger import get_logger

logger = get_logger("explainer")


class BaseExplainer(ABC):
    """Abstract base class for model explainability."""

    @abstractmethod
    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute an explainability heatmap.

        Args:
            image_tensor: Normalized image tensor of shape (3, H, W) or (1, 3, H, W).
            class_idx: Target class index to explain (0=real, 1=fake). If None, predicted class is used.

        Returns:
            Normalized 2D heatmap numpy array of shape (H, W) in range [0, 1].
        """
        pass


class CNNExplainer(BaseExplainer):
    """Grad-CAM explainer for CNN architectures."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        if target_layer is None:
            if hasattr(model, "get_target_layer_for_cam"):
                target_layer = model.get_target_layer_for_cam()
            elif hasattr(model, "conv_head"):
                target_layer = model.conv_head
            else:
                raise ValueError("A target convolutional layer must be provided for CNNExplainer.")
        self.grad_cam = GradCAM(model=model, target_layer=target_layer)

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        try:
            return self.grad_cam.generate(image_tensor, class_idx=class_idx)
        except Exception as e:
            logger.error(f"Grad-CAM explain failed: {e}", exc_info=True)
            self.grad_cam.remove_hooks()
            raise e


class ViTExplainer(BaseExplainer):
    """Attention Rollout explainer for Vision Transformer architectures."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.rollout = AttentionRollout(model=model)

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        try:
            return self.rollout.generate(image_tensor)
        except Exception as e:
            logger.error(f"ViT Rollout explain failed: {e}", exc_info=True)
            self.rollout.remove_hooks()
            raise e


def get_explainer(model: nn.Module, model_name: str = "efficientnet_b0") -> BaseExplainer:
    """Factory function returning the unified explainer appropriate for the model architecture.

    Args:
        model: PyTorch model instance.
        model_name: Name of the model architecture.

    Returns:
        Instance of BaseExplainer (CNNExplainer or ViTExplainer).
    """
    if "vit" in model_name.lower():
        return ViTExplainer(model=model)
    else:
        return CNNExplainer(model=model)
