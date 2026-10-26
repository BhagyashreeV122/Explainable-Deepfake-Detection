"""Evaluation metrics computation for binary deepfake detection."""

from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred_probs: Union[List[float], np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Union[float, list]]:
    """Compute binary classification metrics including Accuracy, AUC, F1, Precision, and Recall.

    Args:
        y_true: Ground-truth binary labels (0 = real, 1 = fake).
        y_pred_probs: Predicted probabilities for class 1 (fake).
        threshold: Classification decision threshold. Defaults to 0.5.

    Returns:
        Dictionary containing calculated metrics and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))

    # ROC AUC calculation (handle cases where single class present gracefully)
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_pred_probs))
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }
