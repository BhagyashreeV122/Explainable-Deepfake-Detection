"""Deepfake detection model architectures wrapping timm backbones."""

import timm
import torch
import torch.nn as nn


class DeepfakeDetector(nn.Module):
    """Classifier for deepfake detection using standard timm vision backbones."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the detector.

        Args:
            model_name: Timm model architecture name (e.g. 'efficientnet_b0' or 'vit_tiny_patch16_224').
            pretrained: Whether to load ImageNet pre-trained weights.
            num_classes: Number of output classes (2 for real vs fake).
            dropout: Dropout probability in the classifier head.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.backbone(x)

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the appropriate convolutional layer for Grad-CAM explainability.

        Returns:
            Target torch.nn.Module (e.g., conv_head for EfficientNet).
        """
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        elif hasattr(self.backbone, "act2"):
            return self.backbone.act2
        elif hasattr(self.backbone, "bn2"):
            return self.backbone.bn2
        else:
            # Fallback: traverse modules and select last Conv2d
            last_conv = None
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
            raise ValueError(f"No convolutional layer found in model {self.model_name}")
