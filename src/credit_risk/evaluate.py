"""Credit-risk evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between score CDFs of defaulters and non-defaulters."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.sort(np.unique(y_score))
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    cdf_pos = np.searchsorted(pos, thresholds, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, thresholds, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
