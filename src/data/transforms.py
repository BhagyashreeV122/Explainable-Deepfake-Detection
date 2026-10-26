"""Image transformation and data augmentation pipelines."""

import io
import random
from PIL import Image
import torchvision.transforms as T


class RandomJPEGCompression:
    """Randomly re-encodes an image using JPEG compression with random quality factors."""

    def __init__(self, quality_min: int = 40, quality_max: int = 90, p: float = 0.5) -> None:
        """Initialize the transform.

        Args:
            quality_min: Minimum JPEG quality (1-100).
            quality_max: Maximum JPEG quality (1-100).
            p: Probability of applying compression.
        """
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random JPEG re-encoding.

        Args:
            img: Input PIL Image.

        Returns:
            Compressed and decoded PIL Image.
        """
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Build data augmentation transform pipeline for training.

    Augmentations included per specification:
    - Random horizontal flip
    - Color jitter
    - Light Gaussian blur
    - Random JPEG re-encoding

    Args:
        image_size: Target image dimension (square).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        RandomJPEGCompression(quality_min=45, quality_max=90, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Build validation/test transform pipeline without stochastic augmentations.

    Args:
        image_size: Target image dimension.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
