"""
Classification and trajectory-error metrics.

Classification metrics back the current interaction-law discovery task
(Task A). ``trajectory_mse`` is the basic error metric for the future
forecasting task (Task B); ADE/FDE live in ``ade_fde.py``.

All functions accept plain numpy arrays / lists so they are decoupled from the
training loop and from Torch.
"""

from __future__ import annotations

import numpy as np


def accuracy(preds, labels) -> float:
    """Top-1 classification accuracy."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float((preds == labels).mean())


def confusion_matrix(labels, preds, n_classes: int) -> np.ndarray:
    """Integer confusion matrix ``C[true, pred]``."""
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def per_class_accuracy(confusion: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Recall per class (diagonal / row sum), keyed by class name."""
    confusion = np.asarray(confusion)
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        denom = confusion[i].sum()
        out[name] = float(confusion[i, i] / denom) if denom else float("nan")
    return out


def trajectory_mse(pred, true, mask=None) -> float:
    """
    Mean squared error between predicted and true positions.

    pred, true : (..., 2) arrays of positions.
    mask       : optional boolean array broadcastable to the leading dims,
                 True where the entry is a real (non-padded) agent.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    sq = ((pred - true) ** 2).sum(axis=-1)      # squared L2 per point
    if mask is None:
        return float(sq.mean())
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float(sq[mask].mean())
