"""
Shared evaluation metrics for DayCent experiments.

Used by both the emulator evaluation (eval_emulator.py) and the inverse
model evaluation (eval_inverse.py).  All functions operate on plain NumPy
arrays so they can be called without any PyTorch dependency.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence

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


def compute_grouped_masked_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    group_keys: Sequence[Hashable],
) -> Dict[str, float]:
    """Compute masked metrics per group, then average metrics across groups."""
    true_arr = np.asarray(true)
    pred_arr = np.asarray(pred)
    mask_arr = np.asarray(mask)

    assert true_arr.shape == pred_arr.shape, "true and pred must have the same shape"
    assert true_arr.shape == mask_arr.shape, "mask must have the same shape as true/pred"

    n_groups_expected = len(group_keys)
    if n_groups_expected == 0:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "n_groups": 0,
            "n_samples_total": 0,
            "mean_samples_per_group": 0.0,
        }

    if true_arr.ndim == 0 or true_arr.shape[0] != n_groups_expected:
        raise ValueError("group_keys must align with the first dimension of true/pred/mask")

    true_rows = true_arr.reshape(n_groups_expected, -1)
    pred_rows = pred_arr.reshape(n_groups_expected, -1)
    mask_rows = mask_arr.reshape(n_groups_expected, -1).astype(bool)

    grouped_indices: Dict[Hashable, List[int]] = {}
    for idx, key in enumerate(group_keys):
        grouped_indices.setdefault(key, []).append(idx)

    per_group_metrics = []
    samples_per_group = []

    for indices in grouped_indices.values():
        group_true = true_rows[indices].reshape(-1)
        group_pred = pred_rows[indices].reshape(-1)
        group_valid = mask_rows[indices].reshape(-1)
        n_valid = int(group_valid.sum())

        if n_valid == 0:
            continue

        per_group_metrics.append(
            compute_regression_metrics(group_true[group_valid], group_pred[group_valid])
        )
        samples_per_group.append(n_valid)

    if not per_group_metrics:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "n_groups": 0,
            "n_samples_total": 0,
            "mean_samples_per_group": 0.0,
        }

    return {
        "mse": float(np.mean([m["mse"] for m in per_group_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in per_group_metrics])),
        "mae": float(np.mean([m["mae"] for m in per_group_metrics])),
        "r2": float(np.mean([m["r2"] for m in per_group_metrics])),
        "n_groups": len(per_group_metrics),
        "n_samples_total": int(np.sum(samples_per_group)),
        "mean_samples_per_group": float(np.mean(samples_per_group)),
    }


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
