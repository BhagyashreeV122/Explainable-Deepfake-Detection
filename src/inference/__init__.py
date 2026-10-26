"""Inference module for deepfake detection and explainability."""

from typing import Any

__all__ = ["InferencePipeline", "predict_single_image", "run_cli"]


def __getattr__(name: str) -> Any:
    if name in ("InferencePipeline", "predict_single_image", "run_cli"):
        import src.inference.predict as _predict
        return getattr(_predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
