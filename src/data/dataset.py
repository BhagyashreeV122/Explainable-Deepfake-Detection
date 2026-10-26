"""PyTorch Dataset for FaceForensics++ extracted face crops."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

LABEL_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_LABEL = {0: "real", 1: "fake"}


class FaceDataset(Dataset):
    """Dataset for binary deepfake detection (real vs fake face crops)."""

    def __init__(
        self,
        manifest_df: Union[pd.DataFrame, str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            manifest_df: DataFrame or path to manifest CSV.
            split: Filter to specific split ('train', 'val', 'test') if provided.
            transform: Transform pipeline to apply to images.
        """
        if isinstance(manifest_df, (str, Path)):
            self.df = pd.read_csv(manifest_df)
        else:
            self.df = manifest_df.copy()

        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Union[str, int]]]:
        row = self.df.iloc[idx]
        img_path = Path(row["filepath"])
        if not img_path.exists():
            candidate = Path(__file__).resolve().parent.parent.parent / row["filepath"]
            if candidate.exists():
                img_path = candidate
        label_str = row["label"]
        label = LABEL_TO_IDX[label_str.lower()]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform:
                image_tensor = self.transform(img)
            else:
                image_tensor = img

        meta = {
            "filepath": str(img_path),
            "split": str(row.get("split", "")),
            "label_str": label_str,
            "pair_id": str(row.get("pair_id", "")),
            "source_video": str(row.get("source_video", "")),
        }

        return image_tensor, label, meta
