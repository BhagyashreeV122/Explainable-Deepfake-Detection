# Deepfake Detection & Explainability Report (FaceForensics++)

## 1. Executive Summary

This project implements an end-to-end, fully reproducible deepfake detection and explainability system trained on the **FaceForensics++ (C23)** benchmark dataset.

The system addresses the core engineering and forensic requirements of modern deepfake detection:
1. **Identity Leakage Prevention**: Data splitting is performed strictly at the person-pair level ($A$ and $B$, their original videos $A.mp4$ and $B.mp4$, and all manipulated videos $A\_B.mp4$ and $B\_A.mp4$ across all 4 manipulation methods reside exclusively in the same split). There is **0% identity leakage** across train, validation, and test splits.
2. **Strict Class Balancing**: 1:1 real-to-fake sample ratio enforced across train, validation, and test splits.
3. **Deep Learning Detector**: EfficientNet-B0 backbone with AdamW optimizer, cosine annealing schedule, and validation ROC-AUC checkpoint selection.
4. **Visual Explainability**: Unified explainer computing post-hoc visual explanations using **Grad-CAM** for convolutional architectures (hooking `conv_head`) and **Attention Rollout** for vision transformers, with fail-safe hook isolation in `try ... finally` blocks.
5. **Production-Ready Inference**: Both a command-line interface (`src.inference.predict`) and a local interactive web app (`app.py` via Gradio) for arbitrary user-supplied images and folders. Includes automatic OpenCV Haar cascade face detection (with 0.25 margin and 224x224 crop), visible fallback warning if no face is detected, 3-panel visual cards, results CSV logging, and gallery generation.

---

## 2. Dataset Preparation & Balancing

### 2.1 Video Ingestion & Person-Pair Graph
- The raw dataset at `C:\Users\ashna\Downloads\FaceForensics++_C23` was ingested across:
  - Real: `original` (1,000 videos).
  - Fake: `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` (1,000 videos per method). `DeepFakeDetection` and `FaceShifter` were excluded per specification.
- Graph analysis proved that the videos partition cleanly into **500 disjoint, undirected person pairs** $(A, B)$ with a degree of exactly 1.0 (no overlap between pairs).

### 2.2 Face Detection & Cropping
- Bounding boxes are detected using OpenCV's frontal face Haar cascade (`haarcascade_frontalface_default.xml`).
- To preserve facial context (jawlines, hair boundaries, forehead artifacts), a margin of $0.25$ is added around the detected face bounding box:
  $$\text{size} = \max(w, h) \times (1 + 2 \times \text{margin})$$
- Crops are resized to $224 \times 224$ pixels and written to `data/faces/{train,val,test}/{real,fake}/`.

### 2.3 Partitioning and Class Balancing
- The dataset was split into **70% Train, 15% Validation, and 15% Test** at the person-pair level.
- Each split is balanced to an exact 1:1 real-to-fake ratio via deterministic downsampling of the majority class:
  $$\text{Real Count} = \text{Fake Count}$$
- Individual person ID verification confirmed zero overlap across splits:
  - **Train**: 350 unique persons
  - **Validation**: 76 unique persons
  - **Test**: 74 unique persons
  - **Overlap between any two splits**: **0 persons**

---

## 3. Model Architecture & Training Methodology

### 3.1 Primary Model: EfficientNet-B0
- **Backbone**: EfficientNet-B0 (`timm`), initialized with ImageNet-1k pre-trained weights.
- **Classification Head**: Dropout ($p=0.2$) followed by a linear projection layer `Linear(1280, 2)`.
- **Secondary Model Support**: ViT-Tiny (`vit_tiny_patch16_224`) with attention rollout hooks.

### 3.2 Data Augmentations
To prevent overfitting on compression and camera artifacts, training images undergo:
- Random Horizontal Flip ($p=0.5$)
- Color Jitter (Brightness=0.2, Contrast=0.2, Saturation=0.2, Hue=0.05)
- Light Gaussian Blur (Kernel=(3, 3), $\sigma \in [0.1, 2.0]$)
- Random JPEG Re-encoding (Quality $\in [45, 90]$, $p=0.5$)
- Standard ImageNet Normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$)

### 3.3 Training Configuration
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$).
- **Learning Rate Schedule**: Cosine Annealing scheduler initialized at $3 \times 10^{-4}$, decaying to $10^{-6}$.
- **Loss Function**: Binary Cross-Entropy (implemented via multi-class CrossEntropyLoss over 2 logits).
- **Epochs**: 5 epochs.
- **Model Selection**: Checkpoint with the highest validation ROC-AUC saved as `outputs/checkpoints/best_model.pt`.
- **Test Evaluation**: Evaluated exactly once on the unseen test split after training completion.

---

## 4. Explainability Framework

### 4.1 Grad-CAM (CNN)
For EfficientNet-B0, forward and backward hooks capture activations and gradients from `conv_head`:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{ij}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
The resulting activation map is upsampled to $(224, 224)$ via bilinear interpolation and min-max normalized to $[0, 1]$.

### 4.2 Attention Rollout (ViT)
For Vision Transformers, fused attention is disabled across all 12 blocks, capturing raw attention matrices $A_l \in \mathbb{R}^{H \times N \times N}$. Residual connections are accounted for:
$$\hat{A}_l = 0.5 \bar{A}_l + 0.5 I$$
$$R_l = \hat{A}_l \times R_{l-1}$$
The CLS token attention to patch tokens $R[\text{cls}, 1:]$ is extracted, reshaped to a $14 \times 14$ grid, bilinearly interpolated to $(224, 224)$, and normalized to $[0, 1]$.

### 4.3 Fail-Safe Hook Isolation
All explainability passes wrap model forward and backward hooks in `try ... finally` blocks, guaranteeing that hooks are detached even if an individual image encounters runtime errors.

---

## 5. Experimental Results (Real Measurements)

### 5.1 Dataset Split Breakdown
| Split | Real Crops | Fake Crops | Total Images | Unique Person IDs |
|---|---|---|---|---|
| **Train** | 1,387 | 1,387 | 2,774 | 350 |
| **Validation** | 300 | 300 | 600 | 76 |
| **Test** | 295 | 295 | 590 | 74 |
| **Total** | **1,982** | **1,982** | **3,964** | **500** |

*Zero person overlap between train, val, and test splits ($0$ identity leakage).*

### 5.2 Training History Excerpt
| Epoch | Learning Rate | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Val ROC-AUC | Val F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.000300 | 2.2301 | 54.07% | 1.1998 | 62.33% | 0.6546 | 0.6049 |
| **2** | 0.000271 | 1.0376 | 63.77% | 0.9261 | 68.67% | 0.7326 | 0.6887 |
| **3** | 0.000197 | 0.7512 | 71.05% | 0.7236 | 72.00% | 0.7809 | 0.7042 |
| **4** | 0.000104 | 0.5218 | 78.62% | 0.7030 | 71.83% | 0.7970 | 0.7051 |
| **5** | 0.000030 | 0.4037 | 83.17% | 0.6913 | 73.00% | **0.8051** | 0.7158 |

*Best checkpoint was selected at Epoch 5 based on peak validation ROC-AUC ($0.8051$).*

### 5.3 Test Split Performance Metrics (Evaluated on Unseen Split)
| Metric | Value | Percentage |
|---|---|---|
| **Accuracy** | 0.7034 | **70.34%** |
| **ROC-AUC** | 0.7914 | **79.14%** |
| **F1-Score** | 0.7049 | **70.49%** |
| **Precision** | 0.7013 | **70.13%** |
| **Recall** | 0.7085 | **70.85%** |
| **Test Loss** | 0.7274 | — |

#### Test Confusion Matrix ($N=590$)
| True \ Predicted | Predicted Real | Predicted Fake | Total |
|---|---|---|---|
| **True Real** | **206** (True Negatives) | 89 (False Positives) | 295 |
| **True Fake** | 86 (False Negatives) | **209** (True Positives) | 295 |

---

## 6. Observations on Heatmaps

1. **Artifact Localization**:
   - For manipulated faces (Deepfakes, FaceSwap, Face2Face, NeuralTextures), Grad-CAM heatmaps consistently concentrate high activation values along the blending boundary lines—notably the perioral (mouth/lips) region, nasal boundaries, and chin perimeter. These areas correspond precisely to where Poisson blending or warping masks leave subtle frequency discontinuities in FaceForensics++ C23 video frames.
2. **Authentic Real Faces**:
   - For real faces, the activation topology is noticeably more diffuse and centered on natural facial landmarks (the nasal bridge, eye pupils, and overall facial structure) rather than focused on sharp perimeter borders.
3. **Correlation vs. Proof**:
   - Grad-CAM heatmaps are post-hoc gradient attribution representations: they reveal which spatial features in the final convolutional stage (`conv_head`) maximally supported the model's classification score. They represent **feature correlation**, not forensic or causal proof of manipulation.

---

## 7. Limitations & Discussion

1. **Dataset Bias & Unseen Domain Generalization**:
   - The model was trained and validated strictly on FaceForensics++ (C23 compression). It has not yet been benchmarked against unseen datasets (e.g., Celeb-DF v2, DFDC, WildDeepfake) or modern generative diffusion models (e.g., Stable Diffusion, Midjourney, Flux). Cross-dataset generalization drops are well-documented across deepfake detection literature due to dataset-specific compression and color profiles.
2. **H.264 Compression Artifacts**:
   - FF++ C23 employs medium H.264 compression ($CRF=23$). High compression frequently attenuates fine-grained frequency artifacts, forcing the model to rely primarily on structural boundary mismatches.
3. **Absence of Temporal Modeling**:
   - This baseline operates on individual 2D face crops. Temporal inconsistencies (e.g., eye blinking rate anomalies, landmark jitter between consecutive frames) are not leveraged by frame-level 2D backbones.
4. **Face Detection Dependency**:
   - Haar cascade face detection requires reasonably frontal, well-lit portraits. Severe profile angles, heavy occlusions, or extreme lighting trigger the fallback mechanism to the resized full image with a visible user warning.

---

## 8. Completion Status Checklist

- [x] **Project Structure & Configs**: Master configuration `configs/config.yaml` established and modular structure implemented.
- [x] **Zero Identity Leakage Splitting**: Person-pair level splitting verified across 500 undirected pairs (0 person overlap between splits).
- [x] **Class Balancing**: Exact 1:1 real/fake ratio enforced for train (1,387/1,387), val (300/300), and test (295/295).
- [x] **Reproducibility**: Global seed (42) configured across Python, NumPy, PyTorch CPU & CUDA.
- [x] **Model Training**: EfficientNet-B0 trained for 5 epochs with AdamW, Cosine Annealing, and data augmentations.
- [x] **Model Selection**: Checkpoint selection performed strictly via validation ROC-AUC (`best_model.pt` at epoch 5 with Val AUC=0.8051).
- [x] **Test Evaluation**: Real test metrics generated once on unseen test split (Accuracy: 70.34%, ROC-AUC: 79.14%).
- [x] **Explainability Module**: Grad-CAM for CNNs and Attention Rollout for ViTs with unified `explain(image, class_idx)` interface, heatmap overlay utility, and fail-safe hook cleanup in `try ... finally`.
- [x] **Inference CLI**: Full CLI `python -m src.inference.predict` with `--image` and `--folder`, face detection with 0.25 margin, 224x224 crop, graceful no-face fallback, 3-panel cards, and `results.csv`.
- [x] **Web Application**: Gradio interface at `app.py` sharing the same inference engine, with styled verdict card, probability meters, explainability tabs, and forensic disclaimer.
- [x] **Unit & Regression Tests**: Pytest test suite with 6 passing tests covering heatmap dimensions, bounds $[0, 1]$, checkpoint reload consistency, seed reproducibility, ViT attention rollout, and full inference execution.
- [x] **Comprehensive Documentation**: Updated `README.md`, `PROGRESS.md`, and `docs/writeup.md`.
