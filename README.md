# FaceForensics++ Deepfake Detection & Explainability (Grad-CAM)

A production-grade, reproducible Deepfake Detection system built from scratch on **FaceForensics++ (C23)** using PyTorch, timm, OpenCV, and Gradio. Features identity-preserving person-pair dataset splitting, 1:1 real/fake balancing, EfficientNet-B0 training with AdamW and cosine annealing schedule, a unified explainability module supporting **Grad-CAM** for CNNs and **Attention Rollout** for Vision Transformers, a robust input-driven inference CLI, and an interactive Gradio web application.

---

## 📁 Repository Structure

```
deepfake-detector/
├── configs/
│   └── config.yaml                 # Master configuration (hyperparameters, paths, thresholds)
├── data/
│   ├── faces/                      # Processed face crops (224x224)
│   │   ├── train/{real,fake}/      # 1,387 real / 1,387 fake
│   │   ├── val/{real,fake}/        # 300 real / 300 fake
│   │   ├── test/{real,fake}/       # 295 real / 295 fake
│   │   ├── dataset_summary.json    # Verified sample distribution
│   │   └── manifest.csv            # Detailed split and pair metadata manifest
│   └── test_samples/               # Verification samples (mix of real, fake, and no-face)
├── docs/
│   └── writeup.md                  # Comprehensive engineering report, real metrics & findings
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt           # Best model checkpoint selected via validation ROC-AUC
│   ├── logs/
│   │   ├── training_history.csv    # Per-epoch training & validation metrics log
│   │   ├── curves.png              # Epoch loss and accuracy curves
│   │   ├── test_metrics.json       # Real test evaluation metrics (accuracy, AUC, F1, etc.)
│   │   └── confusion_matrix.png    # Annotated test confusion matrix
│   ├── heatmaps/
│   │   ├── sample_*.png            # Sample Grad-CAM overlays
│   │   ├── scores.csv              # Explainability sample prediction scores
│   │   └── gallery.png             # Test split explainability comparison gallery
│   └── inference/                  # Outputs from CLI & web app inference
│       ├── <image_name>/           # Subfolder per analyzed image
│       │   ├── original.png        # Original input image
│       │   ├── face_crop.png       # Extracted 224x224 face crop
│       │   ├── heatmap_overlay.png # Grad-CAM overlay on face crop
│       │   └── combined_panel.png  # Diagnostic card (Original | Crop | Overlay + Verdict)
│       ├── gallery.png             # Combined gallery grid of batch inference results
│       └── results.csv             # Structured ledger of all inference runs
├── scripts/
│   ├── prepare_ffpp.py             # Face detection, cropping, pair splitting, and balancing
│   ├── train.py                    # Training pipeline with validation AUC model selection
│   ├── evaluate.py                 # Standalone checkpoint evaluation on test split
│   └── generate_heatmaps.py        # Heatmap overlay generation & gallery builder
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset implementation
│   │   └── transforms.py           # Augmentations (JPEG re-encoding, color jitter, blur, flip)
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, ROC-AUC, F1, Precision, Recall, Confusion Matrix
│   ├── explainability/
│   │   ├── explainer.py            # Unified interface: explain(image, class_idx) -> [0, 1] heatmap
│   │   ├── gradcam.py              # Grad-CAM implementation with robust hook management
│   │   └── rollout.py              # Attention Rollout for Vision Transformers
│   ├── inference/
│   │   ├── __init__.py             # Inference package exports
│   │   └── predict.py              # Command-line interface and inference pipeline
│   ├── models/
│   │   └── detector.py             # EfficientNet-B0 / ViT-Tiny wrapper
│   └── utils/
│       ├── logger.py               # Structured logger with console and file handlers
│       ├── seed.py                 # set_seed(42) for total reproducibility
│       └── visualize.py            # Curves, confusion matrices, and gallery grids
├── tests/
│   └── test_pipeline.py            # Pytest test suite (6 tests: explainability, reload, seed, ViT rollout, inference)
├── app.py                          # Local interactive web interface (Gradio)
├── PROGRESS.md                     # Step-by-step progress and audit tracker
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation & run instructions
```

---

## ⚙️ Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd "c:/Users/ashna/Deepfake detection"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Unit Tests (Verification):**
   ```bash
   pytest -v tests/
   ```
   Ensures that:
   - Heatmap dimensions match input size `(224, 224)`.
   - Heatmap values are strictly bounded in `[0, 1]` without NaNs or Infs.
   - Checkpoint saving and reloading yields identical predictions.
   - Seed `42` provides deterministic reproducibility across all generators.
   - Inference function returns valid verdicts, probabilities, and heatmaps for both face and no-face inputs.

---

## 🚀 Execution Guide

### 1. Data Preparation
Extracts face crops with a 25% margin, partitions videos into Train/Val/Test (70/15/15) at the person-pair level (zero identity leakage), and enforces exact 1:1 real/fake balancing:

```bash
# Full dataset preparation (250 person pairs)
python scripts/prepare_ffpp.py --limit-pairs 250
```

### 2. Model Training
Trains the primary EfficientNet-B0 baseline with AdamW optimizer, Cosine Annealing learning rate schedule, data augmentations (flip, color jitter, blur, JPEG compression), and selects the best checkpoint based on validation ROC-AUC. Once complete, it evaluates the best checkpoint on the unseen test split:

```bash
python scripts/train.py --epochs 5 --batch-size 32
```
*Outputs generated:*
- Best checkpoint: `outputs/checkpoints/best_model.pt`
- Training history: `outputs/logs/training_history.csv`
- Loss and accuracy curves: `outputs/logs/curves.png`
- Test metrics: `outputs/logs/test_metrics.json`
- Test confusion matrix: `outputs/logs/confusion_matrix.png`

### 3. Input-Driven Inference (CLI)
Analyze arbitrary user-supplied images or folders from the command line:

```bash
# Single image prediction
python -m src.inference.predict --image <path_to_image> [--config configs/config.yaml] [--checkpoint outputs/checkpoints/best_model.pt]

# Example with test sample:
python -m src.inference.predict --image data/test_samples/test_real_031_f0048.png

# Batch prediction on an entire folder:
python -m src.inference.predict --folder data/test_samples
```

**What the pipeline does:**
1. Loads the image (JPG/PNG).
2. Detects the largest face using OpenCV Haar cascade, adds a 25% margin, and extracts a 224x224 crop. If no face is detected, it logs a visible warning and falls back to the resized full image.
3. Computes $P(\text{real})$, $P(\text{fake})$, and the final `REAL` or `FAKE` verdict.
4. Generates a Grad-CAM heatmap for the predicted class, and blends it with a Jet colormap onto the face crop.
5. Saves all artifacts to `outputs/inference/<image_name>/`:
   - `original.png`: Original input image
   - `face_crop.png`: Extracted 224x224 face crop
   - `heatmap_overlay.png`: Blended Grad-CAM overlay
   - `combined_panel.png`: 3-panel diagnostic card labeled with verdict & probabilities
   - Appends a structured record to `outputs/inference/results.csv`.
6. For `--folder`, processes all images, logs and continues past any errors, and generates `outputs/inference/gallery.png`.

### 4. Interactive Web Interface (Gradio)
Start the local web application:

```bash
python app.py
```
Open your browser to `http://127.0.0.1:7860`.
- Upload any face image or select one of the built-in benchmark examples.
- Click **Analyze Image & Explain**.
- View the classification verdict badge, probability breakdown meters, face crop, Grad-CAM saliency overlay, and the combined 3-panel card.
- A prominent disclaimer informs users that results are machine-learning statistical estimates rather than legal/forensic proof.

---

## 📊 Benchmark Test Performance (Real Results)

Evaluated on the unseen test split ($N = 590$ images, 295 real / 295 fake):

| Metric | Score |
|---|---|
| **Test Accuracy** | **70.34%** |
| **Test ROC-AUC** | **79.14%** |
| **Test F1-Score** | **70.49%** |
| **Test Precision** | **70.13%** |
| **Test Recall** | **70.85%** |
| **Test Loss** | **0.7274** |

---

## 🔬 Explainability Architecture

- **Grad-CAM (CNNs)**: Targets the last convolutional stage `conv_head` of EfficientNet-B0. Global average pools gradients:
  $$\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$
  Computes activation map $\text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$, resizes via bilinear interpolation to $(224, 224)$, and normalizes to $[0, 1]$.
- **Attention Rollout (ViTs)**: Deactivates fused attention in timm's Vision Transformer blocks, hooks self-attention maps across all 12 blocks, adds residual identity $\hat{A} = 0.5 A + 0.5 I$, recursively multiplies matrices $R_{l} = \hat{A}_l R_{l-1}$, and projects CLS token attention onto the spatial patch grid.
- **Fail-Safe Cleanup**: All hooks are isolated in `try ... finally` blocks to ensure hooks are removed immediately even if an image fails.
