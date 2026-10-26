"""Unit and regression tests verifying explainability, checkpointing, reproducibility, and inference."""

from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.explainer import CNNExplainer, ViTExplainer
from src.inference.predict import InferencePipeline, predict_single_image
from src.models.detector import DeepfakeDetector
from src.utils.seed import set_seed


def test_heatmap_shape_equals_input_size() -> None:
    """Test 1: Heatmap shape matches input image spatial dimensions."""
    set_seed(42)
    model = DeepfakeDetector(model_name="efficientnet_b0", pretrained=False, num_classes=2)
    model.eval()

    explainer = CNNExplainer(model=model)
    H, W = 224, 224
    dummy_input = torch.randn(1, 3, H, W)

    heatmap = explainer.explain(dummy_input, class_idx=1)
    assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy ndarray"
    assert heatmap.shape == (H, W), f"Expected heatmap shape ({H}, {W}), but got {heatmap.shape}"


def test_heatmap_values_within_zero_one() -> None:
    """Test 2: Heatmap values are strictly bounded in [0, 1] without NaN or Inf."""
    set_seed(42)
    model = DeepfakeDetector(model_name="efficientnet_b0", pretrained=False, num_classes=2)
    model.eval()

    explainer = CNNExplainer(model=model)
    dummy_input = torch.randn(1, 3, 224, 224)

    heatmap = explainer.explain(dummy_input, class_idx=0)
    assert np.isfinite(heatmap).all(), "Heatmap contains NaN or Inf values"
    assert heatmap.min() >= 0.0 - 1e-6, f"Heatmap min {heatmap.min()} < 0"
    assert heatmap.max() <= 1.0 + 1e-6, f"Heatmap max {heatmap.max()} > 1"


def test_checkpoint_reload_gives_identical_predictions() -> None:
    """Test 3: Reloading a model checkpoint produces identical predictions."""
    set_seed(42)
    model_orig = DeepfakeDetector(model_name="efficientnet_b0", pretrained=False, num_classes=2)
    model_orig.eval()

    test_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        orig_logits = model_orig(test_input)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        torch.save({"model_state_dict": model_orig.state_dict()}, tmp_path)

        # Initialize fresh model and load weights
        model_reloaded = DeepfakeDetector(model_name="efficientnet_b0", pretrained=False, num_classes=2)
        checkpoint = torch.load(tmp_path, map_location="cpu")
        model_reloaded.load_state_dict(checkpoint["model_state_dict"])
        model_reloaded.eval()

        with torch.no_grad():
            reloaded_logits = model_reloaded(test_input)

        assert torch.allclose(orig_logits, reloaded_logits, atol=1e-6), (
            "Logits after checkpoint reload differ from original model output!"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_seeding_is_reproducible() -> None:
    """Test 4: Random seeding guarantees reproducible behavior across python, numpy, and torch."""
    seed_val = 42

    # Run 1
    set_seed(seed_val)
    py_rand_1 = [random.random() for _ in range(5)]
    np_rand_1 = np.random.randn(5, 5)
    th_rand_1 = torch.randn(5, 5)

    # Run 2
    set_seed(seed_val)
    py_rand_2 = [random.random() for _ in range(5)]
    np_rand_2 = np.random.randn(5, 5)
    th_rand_2 = torch.randn(5, 5)

    assert py_rand_1 == py_rand_2, "Python random generator is not reproducible with set_seed"
    assert np.allclose(np_rand_1, np_rand_2), "Numpy random generator is not reproducible with set_seed"
    assert torch.allclose(th_rand_1, th_rand_2), "PyTorch random generator is not reproducible with set_seed"


def test_vit_explainer_attention_rollout() -> None:
    """Test 5: ViT Attention Rollout produces valid (224, 224) heatmaps bounded in [0, 1]."""
    set_seed(42)
    model = DeepfakeDetector(model_name="vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model.eval()

    explainer = ViTExplainer(model=model)
    dummy_input = torch.randn(1, 3, 224, 224)

    heatmap = explainer.explain(dummy_input)
    assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy ndarray"
    assert heatmap.shape == (224, 224), f"Expected heatmap shape (224, 224), got {heatmap.shape}"
    assert heatmap.min() >= 0.0 - 1e-6 and heatmap.max() <= 1.0 + 1e-6, "Heatmap values out of bounds [0, 1]"


def test_inference_returns_verdict_probabilities_and_heatmap() -> None:
    """Test 6: Inference function returns valid verdict, probabilities, and heatmap for sample images."""
    ckpt_path = PROJECT_ROOT / "outputs/checkpoints/best_model.pt"
    if not ckpt_path.exists():
        pytest.skip("Trained checkpoint not found, skipping inference test.")

    pipeline = InferencePipeline(
        config_path=str(PROJECT_ROOT / "configs/config.yaml"),
        checkpoint_path=str(ckpt_path),
        device="cpu",
    )

    # Test case A: Sample image with face
    sample_img_path = PROJECT_ROOT / "data/faces/test/real/test_real_031_f0048.png"
    if sample_img_path.exists():
        res = pipeline.predict(image_input=sample_img_path, save_outputs=False)
    else:
        synthetic_img = np.random.randint(50, 200, (250, 250, 3), dtype=np.uint8)
        res = pipeline.predict(image_input=synthetic_img, save_outputs=False)

    assert "verdict" in res, "Inference result missing 'verdict'"
    assert res["verdict"] in ("REAL", "FAKE"), f"Unexpected verdict value: {res['verdict']}"

    assert "probabilities" in res, "Inference result missing 'probabilities'"
    assert "real" in res["probabilities"] and "fake" in res["probabilities"], "Probabilities must have real and fake"
    assert 0.0 <= res["prob_real"] <= 1.0, f"P(real) out of bounds: {res['prob_real']}"
    assert 0.0 <= res["prob_fake"] <= 1.0, f"P(fake) out of bounds: {res['prob_fake']}"
    assert abs((res["prob_real"] + res["prob_fake"]) - 1.0) < 1e-4, "Probabilities do not sum to 1.0"

    assert "heatmap" in res, "Inference result missing 'heatmap'"
    heatmap = res["heatmap"]
    assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy ndarray"
    assert heatmap.shape == (224, 224), f"Heatmap shape expected (224, 224), got {heatmap.shape}"
    assert heatmap.min() >= 0.0 - 1e-5 and heatmap.max() <= 1.0 + 1e-5, "Heatmap values out of bounds [0, 1]"

    # Test case B: Image with no face (fallback behavior)
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    blank_res = pipeline.predict(image_input=blank_img, save_outputs=False)
    assert blank_res["face_detected"] is False, "Expected face_detected=False for blank image"
    assert blank_res["verdict"] in ("REAL", "FAKE")
    assert blank_res["heatmap"].shape == (224, 224)

    # Test case C: Convenience function predict_single_image
    res_convenience = predict_single_image(image_input=blank_img, pipeline=pipeline, save_outputs=False)
    assert res_convenience["verdict"] in ("REAL", "FAKE")
