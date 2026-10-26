"""Grad-CAM explainability module for Convolutional Neural Networks."""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize Grad-CAM with target model and convolutional layer.

        Args:
            model: The neural network model.
            target_layer: The target convolutional layer (e.g. conv_head in EfficientNet).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self) -> None:
        self.remove_hooks()
        self.gradients.clear()
        self.activations.clear()

        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            self.activations.append(output)

        def backward_hook(
            module: nn.Module,
            grad_input: Tuple[torch.Tensor, ...],
            grad_output: Tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients.append(grad_output[0])

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Safely remove all registered forward and backward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a 2D Grad-CAM heatmap for the given input image.

        Args:
            image_tensor: Tensor of shape (1, 3, H, W) or (3, H, W).
            class_idx: Target class index. If None, uses the highest scoring class.

        Returns:
            Normalized 2D numpy array heatmap of shape (H, W) in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(True)

        self._register_hooks()
        try:
            self.model.zero_grad()
            logits = self.model(image_tensor)

            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[0, class_idx]
            score.backward(retain_graph=False)

            if not self.gradients or not self.activations:
                raise RuntimeError("Failed to capture gradients or activations during Grad-CAM pass.")

            grads = self.gradients[0]      # shape: (1, C, H', W')
            acts = self.activations[0]     # shape: (1, C, H', W')

            # Global average pooling of gradients
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize to input spatial dimensions
            H, W = image_tensor.shape[2], image_tensor.shape[3]
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
            heatmap = cam.squeeze().detach().cpu().numpy()

            # Min-Max Normalization to [0, 1]
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            return heatmap.astype(np.float32)
        finally:
            self.remove_hooks()
            image_tensor.requires_grad_(False)
