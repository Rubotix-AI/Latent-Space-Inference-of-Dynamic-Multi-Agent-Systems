"""
Average Displacement Error (ADE) and Final Displacement Error (FDE).

Standard trajectory-forecasting metrics for the future prediction task
(Task B). Ready to be consumed once a forecasting head is added.

Conventions
-----------
pred, true : (B, H, N, 2) predicted / ground-truth positions over an
             ``H``-step horizon for ``N`` agents.
mask       : optional (B, N) boolean, True for real (non-padded) agents.

- ADE = mean over agents and horizon of the L2 displacement error.
- FDE = mean over agents of the L2 error at the final horizon step.
"""

from __future__ import annotations

import numpy as np


def _l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """L2 distance over the last axis -> (B, H, N)."""
    return np.linalg.norm(pred - true, axis=-1)


def ade(pred, true, mask=None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = _l2(pred, true)                         # (B, H, N)
    if mask is None:
        return float(err.mean())
    mask = np.asarray(mask, dtype=bool)           # (B, N)
    m = np.broadcast_to(mask[:, None, :], err.shape)   # (B, H, N)
    if m.sum() == 0:
        return 0.0
    return float(err[m].mean())


def fde(pred, true, mask=None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err_final = _l2(pred[:, -1], true[:, -1])     # (B, N)
    if mask is None:
        return float(err_final.mean())
    mask = np.asarray(mask, dtype=bool)
    count = mask.sum() * 1.0
    return float((err_final * mask).sum() / count) if count else 0.0
