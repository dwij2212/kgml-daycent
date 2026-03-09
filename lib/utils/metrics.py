"""
Shared evaluation metrics for DayCent experiments.

Used by both the emulator evaluation (eval_emulator.py) and the inverse
model evaluation (eval_inverse.py).  All functions operate on plain NumPy
arrays so they can be called without any PyTorch dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core scalar metrics
# ---------------------------------------------------------------------------

def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_regression_metrics(
    true: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MSE / RMSE / MAE / R² for two flat arrays.

    Parameters
    ----------
    true, pred : np.ndarray
        1-D arrays of ground-truth and predicted values.

    Returns
    -------
    dict with keys: mse, rmse, mae, r2, n_samples
    """
    assert true.shape == pred.shape, "true and pred must have the same shape"
    n = len(true)
    if n == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0, "n_samples": 0}

    mse  = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = _r2(true, pred)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "n_samples": n}


def compute_masked_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics applying a boolean / 0-1 mask first.

    Parameters
    ----------
    true, pred : np.ndarray
        Arrays of any shape; flattened internally.
    mask : np.ndarray
        Same shape as *true*/*pred*.  Non-zero entries are treated as valid.

    Returns
    -------
    dict with keys: mse, rmse, mae, r2, n_samples
    """
    valid = mask.flatten().astype(bool)
    return compute_regression_metrics(true.flatten()[valid], pred.flatten()[valid])


# ---------------------------------------------------------------------------
# Per-channel metrics (used by inverse eval for driver-response channels
# and for static site-condition features)
# ---------------------------------------------------------------------------

def compute_per_channel_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    channel_names: Optional[List[str]] = None,
    filter_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Compute metrics for each channel (last axis) of (N, ..., C) arrays.

    Parameters
    ----------
    true, pred : np.ndarray
        Shape (N, C) or (N, T, C).  The last axis is treated as channels.
    channel_names : list[str], optional
        Names for each channel.  Falls back to ``ch_0``, ``ch_1``, …
    filter_names : list[str], optional
        If given, only compute metrics for channels whose name starts with
        any prefix in this list (case-insensitive).

    Returns
    -------
    list of dicts, one per (kept) channel:
        {channel_idx, channel_name, mse, rmse, mae, r2, corr}
    """
    n_channels = true.shape[-1]
    results = []

    for c in range(n_channels):
        name = channel_names[c] if channel_names else f"ch_{c}"

        if filter_names is not None:
            name_lower = name.lower()
            if not any(name_lower.startswith(p.lower()) for p in filter_names):
                continue

        t = true[..., c].flatten()
        p = pred[..., c].flatten()

        metrics = compute_regression_metrics(t, p)

        # Pearson correlation
        if len(t) > 1:
            c_mat = np.corrcoef(t, p)
            corr = float(c_mat[0, 1]) if not np.isnan(c_mat).any() else 0.0
        else:
            corr = 0.0

        results.append({
            "channel_idx": c,
            "channel_name": name,
            **metrics,
            "corr": corr,
        })

    return results


# ---------------------------------------------------------------------------
# Emulator-specific: dual-task (yield + SOMSC)
# ---------------------------------------------------------------------------

def compute_emulator_metrics(predictions: dict) -> Dict[str, Dict[str, float]]:
    """Compute yield and SOMSC metrics from the emulator predictions dict.

    Parameters
    ----------
    predictions : dict
        Must have keys: ``yield_pred``, ``yield_true``, ``yield_mask``,
        ``somsc_pred``, ``somsc_true``, ``somsc_mask``.

    Returns
    -------
    dict with keys ``"yield"`` and ``"somsc"``, each containing the
    regression metrics dict from :func:`compute_masked_metrics`.
    """
    yield_metrics = compute_masked_metrics(
        predictions["yield_true"],
        predictions["yield_pred"],
        predictions["yield_mask"],
    )
    somsc_metrics = compute_masked_metrics(
        predictions["somsc_true"],
        predictions["somsc_pred"],
        predictions["somsc_mask"],
    )
    return {"yield": yield_metrics, "somsc": somsc_metrics}
