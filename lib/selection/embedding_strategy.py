"""
Embedding-based point selection strategies (Holzmüller-style).

Implements two greedy strategies that operate on pre-computed latent
embeddings from the DayCent inverse model:

    * **MaxDist** – greedy diversity: always pick the pool point whose
      minimum distance to the already-selected set is largest.
    * **LCMD** (Largest Cluster Maximum Distance) – treat selected points
      as cluster centres, find the cluster with the highest total residual
      (sum of squared distances), and pick the farthest point in *that*
      cluster.

Both strategies operate in **TP (Train + Pool) mode**: an initial warm-start
set is chosen at random, then the greedy loop fills the remaining budget.

Phase 1 – Embedding Preparation (helper functions)
---------------------------------------------------
1. **Aggregation**: average per-PID embeddings across years / scenarios to
   get one static D-dimensional vector per site.
2. **Global Scaling**: divide the full embedding matrix by the mean
   Euclidean norm across all pool points so that average ||z||² ≈ 1.

Both strategies expect ``metadata["embedding_path"]`` pointing to the
output directory of ``eval_inverse.py`` which contains:
    * ``avg_codes.npy``      – (P, D) array of per-PID mean embeddings
    * ``avg_codes_pids.npy`` – (P,)   string array of corresponding PIDs
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStrategy, SelectionResult
from .registry import register_strategy


# =========================================================================== #
#  Embedding helpers
# =========================================================================== #

def load_embeddings(
    embedding_path: str,
    pool_points: List[str],
) -> tuple[np.ndarray, List[str]]:
    """Load per-PID average embeddings and align them with *pool_points*.

    Parameters
    ----------
    embedding_path : str
        Directory containing ``avg_codes.npy`` and ``avg_codes_pids.npy``
        (produced by ``eval_inverse.py``).
    pool_points : list[str]
        Full list of candidate point IDs in the selection pool.

    Returns
    -------
    embeddings : ndarray, shape (N_pool, D)
        Rows are in the **same order** as *pool_points*.
    aligned_pids : list[str]
        Same as *pool_points* (only the subset for which embeddings exist).
    """
    codes = np.load(os.path.join(embedding_path, "avg_codes.npy"))       # (P, D)
    pids = np.load(os.path.join(embedding_path, "avg_codes_pids.npy"))   # (P,)
    pids = [str(p) for p in pids]

    pid_to_idx = {p: i for i, p in enumerate(pids)}

    aligned_indices: List[int] = []
    aligned_pids: List[str] = []
    for p in pool_points:
        if p in pid_to_idx:
            aligned_indices.append(pid_to_idx[p])
            aligned_pids.append(p)

    if not aligned_pids:
        raise ValueError(
            f"No pool points found in embeddings at {embedding_path}. "
            f"First few pool IDs: {pool_points[:5]}, "
            f"first few embedding PIDs: {pids[:5]}."
        )

    embeddings = codes[aligned_indices]  # (N_aligned, D)
    return embeddings, aligned_pids


def global_scale(embeddings: np.ndarray) -> np.ndarray:
    """Apply global scaling so that the mean squared norm equals 1.

    Computes the mean Euclidean norm across all rows and divides the
    entire matrix by that scalar.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)

    Returns
    -------
    scaled : ndarray, shape (N, D)
    """
    norms = np.linalg.norm(embeddings, axis=1)          # (N,)
    mean_norm = norms.mean()
    if mean_norm < 1e-12:
        return embeddings  # avoid division by zero (all-zero embeddings)
    return embeddings / mean_norm


def _sq_distances_to_point(
    embeddings: np.ndarray,
    point_embedding: np.ndarray,
) -> np.ndarray:
    """Squared Euclidean distances from every row of *embeddings* to a
    single *point_embedding*.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
    point_embedding : ndarray, shape (D,)

    Returns
    -------
    dists_sq : ndarray, shape (N,)
    """
    diff = embeddings - point_embedding[np.newaxis, :]
    return np.sum(diff ** 2, axis=1)


# =========================================================================== #
#  MaxDist Strategy
# =========================================================================== #

@register_strategy("maxdist")
class MaxDistStrategy(BaseStrategy):
    """Greedy diversity selection in embedding space (MaxDist / P-greedy).

    Picks the pool point that is **farthest** from its nearest
    already-selected neighbour at each step.

    In incremental mode (when ``selected_points`` is passed to ``select``),
    the greedy distance array is initialised from the already-selected set
    so new picks are diverse w.r.t. *all* prior selections.  A random
    warm-start is only used on the very first step when no prior selection
    exists.

    Parameters
    ----------
    n_points : int
        Number of **new** points to select in this step.
    seed : int
        Random seed for the warm-start initialisation.
    n_warm : int
        Number of initial random points before greedy selection begins
        (only used when ``selected_points`` is empty).  Default 1.
    embedding_path : str
        Path to the directory with ``avg_codes.npy`` /
        ``avg_codes_pids.npy``.
    """

    def __init__(
        self,
        n_points: int,
        seed: int = 42,
        n_warm: int = -1,
        embedding_path: str = "",
        **kwargs,
    ):
        super().__init__(n_points, seed, **kwargs)
        self.n_warm = max(n_points, n_warm)
        self.embedding_path = embedding_path

    def select(
        self,
        pool_points: List[str],
        selected_points: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        self._validate_n(pool_points)
        if selected_points is None:
            selected_points = []

        # ---- resolve embedding path from metadata if not set directly ----
        emb_path = self.embedding_path
        if not emb_path and metadata and "embedding_path" in metadata:
            emb_path = metadata["embedding_path"]
        if not emb_path:
            raise ValueError(
                "MaxDistStrategy requires an embedding_path (either via "
                "constructor kwarg or metadata['embedding_path'])."
            )

        # ---- load & scale embeddings for the UNION of pool + selected ----
        all_points = list(pool_points) + [
            p for p in selected_points if p not in set(pool_points)
        ]
        embeddings_raw, aligned_pids = load_embeddings(emb_path, all_points)
        embeddings = global_scale(embeddings_raw)

        pid_to_emb_idx = {p: i for i, p in enumerate(aligned_pids)}
        aligned_pool_set = set(aligned_pids)

        # Map pool_points to their embedding indices (skip those w/o embeddings)
        pool_pids = [p for p in pool_points if p in aligned_pool_set]
        pool_emb_indices = [pid_to_emb_idx[p] for p in pool_pids]
        n_pool = len(pool_pids)

        if self.n_points > n_pool:
            raise ValueError(
                f"Requested {self.n_points} points but only "
                f"{n_pool} pool points have embeddings."
            )

        pool_emb = embeddings[pool_emb_indices]  # (n_pool, D)

        # ---- Initialise distance array from already-selected points ----
        min_dists = np.full(n_pool, np.inf)
        remaining_mask = np.ones(n_pool, dtype=bool)


        for pid in selected_points:
            if pid in pid_to_emb_idx:
                d = _sq_distances_to_point(pool_emb, embeddings[pid_to_emb_idx[pid]])
                min_dists = np.minimum(min_dists, d)

        # ---- warm-start (only if no prior selection) ----
        selected_new_local: List[int] = []
        if not selected_points:
            rng = random.Random(self.seed)
            n_warm_actual = min(self.n_warm, self.n_points)
            warm_local = rng.sample(range(n_pool), n_warm_actual)
            for lidx in warm_local:
                selected_new_local.append(lidx)
                remaining_mask[lidx] = False
                d = _sq_distances_to_point(pool_emb, pool_emb[lidx])
                min_dists = np.minimum(min_dists, d)

        # ---- greedy loop ----
        n_greedy = self.n_points - len(selected_new_local)
        for _ in range(n_greedy):
            masked_dists = np.where(remaining_mask, min_dists, -1.0)
            next_local = int(np.argmax(masked_dists))

            selected_new_local.append(next_local)
            remaining_mask[next_local] = False

            d = _sq_distances_to_point(pool_emb, pool_emb[next_local])
            min_dists = np.minimum(min_dists, d)

        selected_pids = [pool_pids[i] for i in selected_new_local]

        return SelectionResult(
            selected_points=sorted(selected_pids),
            strategy_name=self.name,
            strategy_params={
                "n_points": self.n_points,
                "seed": self.seed,
                "n_warm": self.n_warm,
                "embedding_path": emb_path,
            },
            metadata={
                "selection_order": selected_pids,
                "n_prior_selected": len(selected_points),
            },
        )


# =========================================================================== #
#  LCMD Strategy (Largest Cluster Maximum Distance)
# =========================================================================== #

@register_strategy("lcmd")
class LCMDStrategy(BaseStrategy):
    """Largest Cluster Maximum Distance selection in embedding space.

    Treats selected points as cluster centres.  At each step it:
      1. Computes the *cluster size* of every centre (sum of squared
         distances of pool points assigned to it).
      2. Identifies the centre with the **largest cluster size**.
      3. Within that cluster, picks the point **farthest** from its
         centre.

    This balances **diversity** (picks far-away points) with
    **representativeness** (picks from dense / under-served regions).

    In incremental mode (when ``selected_points`` is passed), the
    distance and cluster-assignment arrays are initialised from the
    already-selected set.  A random warm-start is only used on the
    very first step.

    Parameters
    ----------
    n_points : int
        Number of **new** points to select in this step.
    seed : int
        Random seed for the warm-start initialisation.
    n_warm : int
        Number of initial random points before greedy selection begins
        (only used when ``selected_points`` is empty).  Default 1.
    embedding_path : str
        Path to the directory with ``avg_codes.npy`` /
        ``avg_codes_pids.npy``.
    """

    def __init__(
        self,
        n_points: int,
        seed: int = 42,
        n_warm: int = 1,
        embedding_path: str = "",
        **kwargs,
    ):
        super().__init__(n_points, seed, **kwargs)
        self.n_warm = max(1, n_warm)
        self.embedding_path = embedding_path

    def select(
        self,
        pool_points: List[str],
        selected_points: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        self._validate_n(pool_points)
        if selected_points is None:
            selected_points = []

        # ---- resolve embedding path ----
        emb_path = self.embedding_path
        if not emb_path and metadata and "embedding_path" in metadata:
            emb_path = metadata["embedding_path"]
        if not emb_path:
            raise ValueError(
                "LCMDStrategy requires an embedding_path (either via "
                "constructor kwarg or metadata['embedding_path'])."
            )

        # ---- load & scale embeddings for UNION of pool + selected ----
        all_points = list(pool_points) + [
            p for p in selected_points if p not in set(pool_points)
        ]
        embeddings_raw, aligned_pids = load_embeddings(emb_path, all_points)
        embeddings = global_scale(embeddings_raw)

        pid_to_emb_idx = {p: i for i, p in enumerate(aligned_pids)}
        aligned_pool_set = set(aligned_pids)

        pool_pids = [p for p in pool_points if p in aligned_pool_set]
        pool_emb_indices = [pid_to_emb_idx[p] for p in pool_pids]
        n_pool = len(pool_pids)

        if self.n_points > n_pool:
            raise ValueError(
                f"Requested {self.n_points} points but only "
                f"{n_pool} pool points have embeddings."
            )

        pool_emb = embeddings[pool_emb_indices]  # (n_pool, D)

        # ---- kernel diagonal (squared norms after scaling) ----
        k_diag = np.sum(pool_emb ** 2, axis=1)  # (n_pool,)

        # ---- Initialise distance / cluster arrays from prior selection ----
        min_dists = np.full(n_pool, np.inf)
        remaining_mask = np.ones(n_pool, dtype=bool)
        # cluster_centre[i] = a sentinel index identifying the nearest
        # selected centre for pool point i.  For prior-selected points we
        # use negative IDs (-1, -2, …) to distinguish them from newly-
        # selected local indices.
        cluster_centre = np.full(n_pool, -1, dtype=int)

        has_centres = False
        for sentinel, pid in enumerate(selected_points):
            if pid in pid_to_emb_idx:
                d = _sq_distances_to_point(pool_emb, embeddings[pid_to_emb_idx[pid]])
                closer_mask = d < min_dists
                cluster_centre[closer_mask] = -(sentinel + 1)  # negative sentinel
                min_dists = np.minimum(min_dists, d)
                has_centres = True

        # ---- warm-start (only if no prior selection) ----
        selected_new_local: List[int] = []
        if not selected_points:
            rng = random.Random(self.seed)
            n_warm_actual = min(self.n_warm, self.n_points)
            warm_local = rng.sample(range(n_pool), n_warm_actual)
            for lidx in warm_local:
                selected_new_local.append(lidx)
                remaining_mask[lidx] = False
                d = _sq_distances_to_point(pool_emb, pool_emb[lidx])
                closer_mask = d < min_dists
                cluster_centre[closer_mask] = lidx
                min_dists = np.minimum(min_dists, d)
                has_centres = True

        # ---- greedy LCMD loop ----
        n_greedy = self.n_points - len(selected_new_local)
        for _ in range(n_greedy):
            if not has_centres:
                # Edge case: no centres at all — pick max kernel diagonal
                masked_kdiag = np.where(remaining_mask, k_diag, -1.0)
                next_local = int(np.argmax(masked_kdiag))
            else:
                # 1. Compute cluster sizes (sum of sq distances)
                cluster_sizes: Dict[int, float] = {}
                remaining_indices = np.where(remaining_mask)[0]

                for i in remaining_indices:
                    c = int(cluster_centre[i])
                    cluster_sizes[c] = cluster_sizes.get(c, 0.0) + float(min_dists[i])

                # 2. Largest cluster
                largest_centre = max(cluster_sizes, key=cluster_sizes.get)

                # 3. Farthest point in that cluster
                candidates = [
                    i for i in remaining_indices
                    if cluster_centre[i] == largest_centre
                ]
                next_local = int(max(candidates, key=lambda i: min_dists[i]))

            # ---- Add new point ----
            selected_new_local.append(next_local)
            remaining_mask[next_local] = False
            has_centres = True

            # Update distances and cluster assignments
            d = _sq_distances_to_point(pool_emb, pool_emb[next_local])
            closer_mask = d < min_dists
            cluster_centre[closer_mask] = next_local
            min_dists = np.minimum(min_dists, d)

        selected_pids = [pool_pids[i] for i in selected_new_local]

        return SelectionResult(
            selected_points=sorted(selected_pids),
            strategy_name=self.name,
            strategy_params={
                "n_points": self.n_points,
                "seed": self.seed,
                "n_warm": self.n_warm,
                "embedding_path": emb_path,
            },
            metadata={
                "selection_order": selected_pids,
                "n_prior_selected": len(selected_points),
            },
        )

