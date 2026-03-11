"""
Stratified point selection strategy.

Stratification can be done on:
  - Initial site conditions (soil, SOM, climate, …)
  - Spatial location (lat/lon)
  - Elevation
  - Or any combination via PCA on the full feature set.

The idea: cluster the pool into ``n_points`` groups using K-Means on the
selected feature space, then pick the point closest to each centroid.
This ensures the selected subset covers the feature diversity of the pool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseStrategy, SelectionResult
from .registry import register_strategy


# Pre-defined feature groups that can be mixed-and-matched
FEATURE_GROUPS = {
    "spatial": ["SITLAT", "SITLNG"],
    "elevation": ["ELEV"],
    "climate": [
        "PRECIP(1)", "PRECIP(2)", "PRECIP(3)", "PRECIP(4)",
        "PRECIP(5)", "PRECIP(6)", "PRECIP(7)", "PRECIP(8)",
        "PRECIP(9)", "PRECIP(10)", "PRECIP(11)", "PRECIP(12)",
        "TMN2M(1)", "TMN2M(2)", "TMN2M(3)", "TMN2M(4)",
        "TMN2M(5)", "TMN2M(6)", "TMN2M(7)", "TMN2M(8)",
        "TMN2M(9)", "TMN2M(10)", "TMN2M(11)", "TMN2M(12)",
        "TMX2M(1)", "TMX2M(2)", "TMX2M(3)", "TMX2M(4)",
        "TMX2M(5)", "TMX2M(6)", "TMX2M(7)", "TMX2M(8)",
        "TMX2M(9)", "TMX2M(10)", "TMX2M(11)", "TMX2M(12)",
    ],
    "soil": [
        "SLSAND(1)", "SLCLAY(1)", "SLPH(1)", "SLBLKD(1)",
        "SLFLDC(1)", "SLWLTP(1)",
    ],
    "som": [
        "SOM1CI(1,1)", "SOM1CI(2,1)", "SOM2CI(1,1)",
        "SOM2CI(2,1)", "SOM3CI(1)",
    ],
}


def _resolve_features(feature_groups: List[str]) -> List[str]:
    """Expand a list of group names to the union of their feature columns."""
    features = []
    for g in feature_groups:
        if g in FEATURE_GROUPS:
            features.extend(FEATURE_GROUPS[g])
        else:
            # treat as a literal column name
            features.append(g)
    # preserve order, remove duplicates
    seen = set()
    out = []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


@register_strategy("stratified")
class StratifiedStrategy(BaseStrategy):
    """K-Means-based stratified selection.

    Parameters
    ----------
    n_points : int
        Number of points to select (= number of clusters).
    seed : int
        Random seed for K-Means and tie-breaking.
    feature_groups : list[str]
        Which feature groups to use for stratification.
        Valid names: ``"spatial"``, ``"elevation"``, ``"climate"``,
        ``"soil"``, ``"som"``, or any column name from the initial
        conditions table.
        Default: ``["spatial", "elevation", "climate", "soil"]``.
    """

    def __init__(
        self,
        n_points: int,
        seed: int = 42,
        feature_groups: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(n_points, seed, **kwargs)
        self.feature_groups = feature_groups or [
            "spatial", "elevation", "climate", "soil",
        ]

    def select(
        self,
        pool_points: List[str],
        selected_points: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        self._validate_n(pool_points)

        if metadata is None or "init_cond_df" not in metadata:
            raise ValueError(
                "StratifiedStrategy requires metadata['init_cond_df'] "
                "(the initial site conditions DataFrame)."
            )

        init_df: pd.DataFrame = metadata["init_cond_df"].copy()
        init_df["id"] = init_df["id"].astype(str)
        pool_df = init_df[init_df["id"].isin(pool_points)].copy()

        if len(pool_df) < self.n_points:
            raise ValueError(
                f"Only {len(pool_df)} pool points have initial-condition data "
                f"(need {self.n_points})."
            )

        feature_cols = _resolve_features(self.feature_groups)
        # keep only cols that exist in the df
        feature_cols = [c for c in feature_cols if c in pool_df.columns]
        if not feature_cols:
            raise ValueError(
                f"No matching feature columns found for groups "
                f"{self.feature_groups}."
            )

        # Drop rows with NaN in the selected features (rare edge cases)
        pool_df = pool_df.dropna(subset=feature_cols)

        X = pool_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(
            n_clusters=self.n_points,
            random_state=self.seed,
            n_init=10,
        )
        labels = kmeans.fit_predict(X_scaled)
        pool_df = pool_df.copy()
        pool_df["_cluster"] = labels

        # For each cluster, pick the point closest to the centroid
        selected_ids: List[str] = []
        cluster_map: Dict[int, str] = {}
        for k in range(self.n_points):
            mask = pool_df["_cluster"] == k
            cluster_X = X_scaled[mask]
            centroid = kmeans.cluster_centers_[k]
            dists = np.linalg.norm(cluster_X - centroid, axis=1)
            closest_idx = np.argmin(dists)
            pid = pool_df.loc[mask, "id"].iloc[closest_idx]
            selected_ids.append(str(pid))
            cluster_map[int(k)] = str(pid)

        return SelectionResult(
            selected_points=sorted(selected_ids),
            strategy_name=self.name,
            strategy_params={
                "n_points": self.n_points,
                "seed": self.seed,
                "feature_groups": self.feature_groups,
                "feature_cols_used": feature_cols,
            },
            metadata={
                "cluster_assignments": {
                    str(row["id"]): int(row["_cluster"])
                    for _, row in pool_df.iterrows()
                },
                "cluster_representative": cluster_map,
            },
        )
