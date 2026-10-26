"""Attention Rollout explainability module for Vision Transformers."""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionRollout:
    """Computes Attention Rollout heatmaps for Vision Transformer models."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize Attention Rollout.

        Args:
            model: Vision Transformer model (e.g. vit_tiny_patch16_224).
        """
        self.model = model
        self.attentions: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Ensure fused attention is disabled across all blocks
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "fused_attn"):
                    block.attn.fused_attn = False
        elif hasattr(self.model, "backbone") and hasattr(self.model.backbone, "blocks"):
            for block in self.model.backbone.blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "fused_attn"):
                    block.attn.fused_attn = False

    def _register_hooks(self) -> None:
        self.remove_hooks()
        self.attentions.clear()

        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            # Capture attention tensor input to attn_drop
            self.attentions.append(input[0].detach().cpu())

        blocks = getattr(self.model, "blocks", None)
        if blocks is None and hasattr(self.model, "backbone"):
            blocks = getattr(self.model.backbone, "blocks", None)

        if blocks is not None:
            for block in blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "attn_drop"):
                    self.hooks.append(block.attn.attn_drop.register_forward_hook(hook_fn))

    def remove_hooks(self) -> None:
        """Safely remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate Attention Rollout heatmap.

        Args:
            image_tensor: Tensor of shape (1, 3, H, W) or (3, H, W).

        Returns:
            Normalized 2D numpy array heatmap of shape (H, W) in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        self._register_hooks()
        try:
            with torch.no_grad():
                _ = self.model(image_tensor)

            if not self.attentions:
                raise RuntimeError("No attention maps were captured during ViT forward pass.")

            # Rollout computation across layers
            num_tokens = self.attentions[0].size(-1)
            result = torch.eye(num_tokens)

            for attn in self.attentions:
                # Average across attention heads: shape (num_tokens, num_tokens)
                attn_heads_avg = attn[0].mean(dim=0)
                # Add residual connection identity and re-normalize rows
                attn_with_identity = 0.5 * attn_heads_avg + 0.5 * torch.eye(num_tokens)
                attn_norm = attn_with_identity / attn_with_identity.sum(dim=-1, keepdim=True)
                result = torch.matmul(attn_norm, result)

            # Extract CLS token attention to patch tokens (index 1 onwards)
            mask = result[0, 1:]
            grid_size = int(np.sqrt(mask.size(0)))
            heatmap = mask.reshape(1, 1, grid_size, grid_size)

            H, W = image_tensor.shape[2], image_tensor.shape[3]
            heatmap = F.interpolate(heatmap, size=(H, W), mode="bilinear", align_corners=False)
            heatmap = heatmap.squeeze().numpy()

            # Normalize to [0, 1]
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            return heatmap.astype(np.float32)
        finally:
            self.remove_hooks()
