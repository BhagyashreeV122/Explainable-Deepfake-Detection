"""Data preparation script for FaceForensics++: face detection, cropping, identity-preserving splitting, and balancing."""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("prepare_ffpp")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_undirected_pairs(raw_dir: str, fake_folders: List[str]) -> List[Tuple[str, str]]:
    """Scan manipulation folders to find all unique undirected person pairs (A, B) where A < B."""
    pairs = set()
    for folder in fake_folders:
        folder_path = Path(raw_dir) / folder
        if not folder_path.exists():
            continue
        for file in folder_path.glob("*.mp4"):
            stem = file.stem
            parts = stem.split("_")
            if len(parts) == 2:
                p1, p2 = parts[0], parts[1]
                pair = (min(p1, p2), max(p1, p2))
                pairs.add(pair)
    return sorted(list(pairs))


def extract_face_crop(
    frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
    margin: float = 0.25,
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """Detect the largest face in frame, apply margin, and resize to target_size."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None

    # Pick the largest face detected
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = faces[0]

    H, W = frame.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)

    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(W, int(cx + side / 2.0))
    y2 = min(H, int(cy + side / 2.0))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def sample_frames_from_video(
    video_path: Path,
    num_samples: int,
    cascade: cv2.CascadeClassifier,
    margin: float,
    target_size: int,
) -> List[Tuple[int, np.ndarray]]:
    """Sample evenly spaced frames and extract face crops."""
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample indices spaced across middle 80% of video to avoid transition artifacts
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    if end_frame <= start_frame:
        start_frame, end_frame = 0, total_frames - 1

    step = max(1, (end_frame - start_frame) // max(1, num_samples))
    indices = [min(end_frame, start_frame + i * step) for i in range(num_samples)]

    extracted = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        crop = extract_face_crop(frame, cascade, margin=margin, target_size=target_size)
        if crop is not None:
            extracted.append((idx, crop))

    cap.release()
    return extracted


def process_pair(
    pair: Tuple[str, str],
    split: str,
    raw_dir: Path,
    out_dir: Path,
    fake_folders: List[str],
    cascade: cv2.CascadeClassifier,
    margin: float,
    target_size: int,
    dry_run: bool = False,
) -> List[dict]:
    """Extract and save face crops for real and fake videos of one pair."""
    p1, p2 = pair
    pair_id = f"{p1}_{p2}"
    records = []

    # 1. Real videos: p1.mp4 and p2.mp4
    real_paths = [
        raw_dir / "original" / f"{p1}.mp4",
        raw_dir / "original" / f"{p2}.mp4",
    ]
    for rpath in real_paths:
        if not rpath.exists():
            continue
        crops = sample_frames_from_video(
            rpath, num_samples=4, cascade=cascade, margin=margin, target_size=target_size
        )
        for fidx, crop in crops:
            fname = f"{split}_real_{rpath.stem}_f{fidx:04d}.png"
            dest = out_dir / split / "real" / fname
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), crop)
            records.append({
                "filepath": str(dest).replace("\\", "/"),
                "split": split,
                "label": "real",
                "label_idx": 0,
                "pair_id": pair_id,
                "source_video": rpath.name,
                "frame_idx": fidx,
            })

    # 2. Fake videos: p1_p2 and p2_p1 across 4 methods (sample 1 frame per video)
    fake_names = [f"{p1}_{p2}.mp4", f"{p2}_{p1}.mp4"]
    for method in fake_folders:
        for fname_vid in fake_names:
            fpath = raw_dir / method / fname_vid
            if not fpath.exists():
                continue
            crops = sample_frames_from_video(
                fpath, num_samples=1, cascade=cascade, margin=margin, target_size=target_size
            )
            for fidx, crop in crops:
                fname_img = f"{split}_fake_{method}_{fpath.stem}_f{fidx:04d}.png"
                dest = out_dir / split / "fake" / fname_img
                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), crop)
                records.append({
                    "filepath": str(dest).replace("\\", "/"),
                    "split": split,
                    "label": "fake",
                    "label_idx": 1,
                    "pair_id": pair_id,
                    "source_video": f"{method}/{fpath.name}",
                    "frame_idx": fidx,
                })

    return records


def balance_split(records: List[dict], split: str, rng: random.Random) -> List[dict]:
    """Ensure exact 1:1 balance between real and fake samples in a split."""
    real_items = [r for r in records if r["label"] == "real"]
    fake_items = [r for r in records if r["label"] == "fake"]

    target_count = min(len(real_items), len(fake_items))
    logger.info(f"Balancing {split}: Raw Real={len(real_items)}, Fake={len(fake_items)} -> Target Balanced={target_count} each")

    rng.shuffle(real_items)
    rng.shuffle(fake_items)

    kept_real = real_items[:target_count]
    kept_fake = fake_items[:target_count]

    # Delete discarded images if they were written to disk
    kept_paths = {r["filepath"] for r in kept_real + kept_fake}
    discarded = (real_items[target_count:] + fake_items[target_count:])
    for d in discarded:
        p = Path(d["filepath"])
        if p.exists() and str(p).replace("\\", "/") not in kept_paths:
            try:
                p.unlink()
            except OSError:
                pass

    return kept_real + kept_fake


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FaceForensics++ Face Crops Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Limit number of person pairs (e.g. 250)")
    parser.add_argument("--dry-run", action="store_true", help="Perform split calculation without extracting frames")
    parser.add_argument("--workers", type=int, default=6, help="Worker threads for video decoding")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    rng = random.Random(seed)

    raw_dir = Path(cfg["dataset"]["raw_dataset_dir"])
    out_dir = Path(cfg["dataset"]["processed_dir"])
    fake_folders = cfg["dataset"]["fake_folders"]
    margin = cfg["dataset"]["crop_margin"]
    target_size = cfg["dataset"]["image_size"]

    limit_pairs = args.limit_pairs or cfg["dataset"].get("limit_pairs", 250)

    logger.info(f"Scanning raw FF++ dataset at: {raw_dir}")
    pairs = get_undirected_pairs(str(raw_dir), fake_folders)
    logger.info(f"Found {len(pairs)} unique person pairs in dataset.")

    # Shuffle pairs deterministically
    rng.shuffle(pairs)
    if limit_pairs and limit_pairs < len(pairs):
        pairs = pairs[:limit_pairs]
        logger.info(f"Constrained dataset to {len(pairs)} pairs per limit.")

    # Split 70% train, 15% val, 15% test at pair level
    ratios = cfg["dataset"]["split_ratios"]
    n_total = len(pairs)
    n_train = int(round(ratios["train"] * n_total))
    n_val = int(round(ratios["val"] * n_total))
    _n_test = n_total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    logger.info(f"Split breakdown: Train={len(train_pairs)} pairs, Val={len(val_pairs)} pairs, Test={len(test_pairs)} pairs.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    all_records: List[dict] = []

    splits_data = [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]

    for split_name, split_pairs in splits_data:
        logger.info(f"Extracting face crops for split '{split_name}' ({len(split_pairs)} pairs)...")
        split_records: List[dict] = []

        # Process each pair
        with tqdm(total=len(split_pairs), desc=f"Processing {split_name}") as pbar:
            for pair in split_pairs:
                recs = process_pair(
                    pair=pair,
                    split=split_name,
                    raw_dir=raw_dir,
                    out_dir=out_dir,
                    fake_folders=fake_folders,
                    cascade=cascade,
                    margin=margin,
                    target_size=target_size,
                    dry_run=args.dry_run,
                )
                split_records.extend(recs)
                pbar.update(1)

        # Balance real vs fake in this split
        balanced = balance_split(split_records, split_name, rng)
        all_records.extend(balanced)

    # Save manifest
    manifest_df = pd.DataFrame(all_records)
    manifest_path = Path(cfg["dataset"].get("manifest_path", "data/faces/manifest.csv"))
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Manifest written to {manifest_path} ({len(manifest_df)} total images)")

    # Print and save summary
    summary = {}
    print("\n" + "=" * 55)
    print("      DATASET PREPARATION SUMMARY (BALANCED)")
    print("=" * 55)
    print(f"{'Split':<10} | {'Real':<8} | {'Fake':<8} | {'Total':<8}")
    print("-" * 55)
    for s in ["train", "val", "test"]:
        sub = manifest_df[manifest_df["split"] == s]
        r_cnt = int((sub["label"] == "real").sum())
        f_cnt = int((sub["label"] == "fake").sum())
        tot = len(sub)
        print(f"{s:<10} | {r_cnt:<8} | {f_cnt:<8} | {tot:<8}")
        summary[s] = {"real": r_cnt, "fake": f_cnt, "total": tot}

    print("-" * 55)
    tot_real = int((manifest_df["label"] == "real").sum())
    tot_fake = int((manifest_df["label"] == "fake").sum())
    tot_all = len(manifest_df)
    print(f"{'Total':<10} | {tot_real:<8} | {tot_fake:<8} | {tot_all:<8}")
    print("=" * 55 + "\n")
    summary["overall"] = {"real": tot_real, "fake": tot_fake, "total": tot_all}

    summary_path = out_dir / "dataset_summary.json"
    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
