# Project Progress & Status Tracker

## Stage Audit & Verification (Step 0)

| Verification Item | Status | Details |
|---|---|---|
| **2a. Project structure and configs** | **PASS** | `configs/config.yaml`, `src/`, `scripts/`, `tests/` structured and present. |
| **2b. Face crops dataset & splits** | **PASS** | Extracted and balanced 1:1 real/fake: <br>• `train`: 1,387 real / 1,387 fake (total 2,774)<br>• `val`: 300 real / 300 fake (total 600)<br>• `test`: 295 real / 295 fake (total 590)<br>• Zero identity leakage across splits (350 train persons, 76 val persons, 74 test persons; 0 overlap). |
| **2c. Trained checkpoint & logs** | **PASS** | Model trained for 5 epochs without interruption.<br>• `outputs/checkpoints/best_model.pt` exists (48.5 MB, epoch 5).<br>• `outputs/logs/training_history.csv` present.<br>• `outputs/logs/curves.png` & `confusion_matrix.png` present.<br>• `outputs/logs/test_metrics.json` present (Test Accuracy: 70.34%, Test ROC-AUC: 79.14%). |
| **Unit Tests Suite** | **PASS** | `pytest -v tests/` passed all 6 tests cleanly in 7.39s (including ViT Attention Rollout and convenience APIs). |

---

## Roadmap & Execution Checklist

- [x] **Step 0: Audit & Resume**
  - [x] Read project tree, configs, README, and logs
  - [x] Verify data split counts, balancing, and identity preservation
  - [x] Verify trained model checkpoint and evaluation outputs
  - [x] Execute existing test suite
  - [x] Initialize and maintain `PROGRESS.md`

- [x] **Step 1: Explainability Module**
  - [x] Verify Grad-CAM on CNN (last conv stage `conv_head`)
  - [x] Verify Attention Rollout for ViT
  - [x] Verify unified interface `explain(image, class_idx) -> heatmap (H, W) in [0, 1]`
  - [x] Verify overlay utility with colormaps
  - [x] Verify hook registration and cleanup in `try ... finally` blocks

- [x] **Step 2: Input-Driven Inference Pipeline & UI**
  - [x] Implement `src/inference/__init__.py` and `src/inference/predict.py`
    - [x] CLI: `--image <path>` and `--folder <path>` options with `--config` and `--checkpoint`
    - [x] Face detection using OpenCV Haar cascade with 0.25 margin, 224x224 crop
    - [x] Graceful fallback to full image with visible warning if no face is detected
    - [x] Compute $P(\text{real})$, $P(\text{fake})$, and final verdict
    - [x] Generate explainability heatmap on predicted class and create overlay
    - [x] Save output artifacts: original image, face crop, overlay, combined 3-panel image, and append row to `outputs/inference/results.csv`
    - [x] Multi-image / folder gallery generation (`outputs/inference/gallery.png`)
  - [x] Implement Gradio local web interface `app.py`
    - [x] Add `gradio>=4.0.0` to `requirements.txt`
    - [x] Display verdict badge, probabilities, face crop, heatmap overlay, combined 3-panel card, and forensic disclaimer
    - [x] Ensure shared codebase with CLI (no duplicated logic)

- [x] **Step 3: Comprehensive Testing & Verification**
  - [x] Add test for inference function returning verdict, probabilities, and heatmap (`test_inference_returns_verdict_probabilities_and_heatmap`)
  - [x] Run full pytest suite ensuring all 5 tests pass
  - [x] Run inference end-to-end on >=5 test split images (real & fake) + 1 image without a face
  - [x] Confirm no-face warning logged and handled gracefully in both CLI and 3-panel cards

- [x] **Step 4: Documentation & Final Deliverables**
  - [x] Update `README.md` with exact setup, CLI and web app commands, and full directory tree
  - [x] Update `docs/writeup.md` with real metrics, training log excerpt, heatmap observations, limitations, and checklist
  - [x] Review all generated assets and verify acceptance criteria
