"""
Bayesian Optimization over combinatorial subset space (GraphComBO-style).

Adapts the GraphComBO framework (Liang, Wan & Dong, NeurIPS 2024) to the
DayCent data-selection problem.  Each k-subset of pool locations is a node
in a *combo-graph*; edges connect subsets differing by one location swap.
A GP with a diffusion kernel defined on local combo-subgraphs drives
Expected Improvement acquisition.

Unlike the greedy strategies, BOGraphStrategy requires an **ask-tell**
outer loop::

    strategy = get_strategy("bo_graph", n_points=50, seed=42,
                            embedding_path="/path/to/embeddings")
    for i in range(n_iterations):
        result = strategy.select(pool_points, metadata=metadata)
        score  = train_and_evaluate(result.selected_points)   # expensive
        strategy.tell(result.selected_points, score)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve, LinAlgError
from scipy.optimize import minimize
from scipy.stats import norm

from .base import BaseStrategy, SelectionResult
from .registry import register_strategy
from .embedding_strategy import load_embeddings, global_scale


# =========================================================================== #
#  Base graph construction
# =========================================================================== #

def _build_base_graph(
    embeddings: np.ndarray,
    epsilon_factor: float = 0.3,
) -> nx.Graph:
    """Build an epsilon-ball graph from embedding vectors.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
        Globally-scaled embedding vectors, one per pool location.
    epsilon_factor : float
        Threshold multiplier: ``epsilon = mean_pairwise_dist * factor``.

    Returns
    -------
    nx.Graph with integer node indices ``0 .. N-1``.
    """
    dists = squareform(pdist(embeddings, metric="euclidean"))
    n = len(embeddings)

    # Compute threshold from upper-triangle mean
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = dists[triu_mask].mean()
    epsilon = mean_dist * epsilon_factor

    G = nx.Graph()
    G.add_nodes_from(range(n))

    rows, cols = np.where((dists < epsilon) & triu_mask)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))

    # Ensure connectivity by linking disconnected components
    components = list(nx.connected_components(G))
    while len(components) > 1:
        comp0 = sorted(components[0])
        rest = sorted(set().union(*components[1:]))
        sub_dists = dists[np.ix_(comp0, rest)]
        mi = np.unravel_index(np.argmin(sub_dists), sub_dists.shape)
        G.add_edge(comp0[mi[0]], rest[mi[1]])
        components = list(nx.connected_components(G))

    return G


# =========================================================================== #
#  Combo-graph construction (ported from GraphComBO)
# =========================================================================== #

def _find_combo_neighbors(graph: nx.Graph, combo_node: tuple) -> List[tuple]:
    """Enumerate all 1-swap neighbours of *combo_node* in the base graph.

    A 1-swap neighbour is obtained by replacing one element of the
    combo-node with one of its graph-neighbours that is not already in the
    combo-node.
    """
    combo_set = set(combo_node)
    neighbors: List[tuple] = []
    for idx, node in enumerate(combo_node):
        node_nbrs = set(graph.neighbors(node)) - combo_set
        for nbr in node_nbrs:
            new_combo = list(combo_node)
            new_combo[idx] = nbr
            neighbors.append(tuple(sorted(new_combo)))
    return neighbors


def _build_combo_subgraph(
    graph: nx.Graph,
    center: tuple,
    Q: int = 200,
    l_max: int = 100,
    rng: np.random.RandomState | None = None,
) -> Tuple[nx.Graph, int]:
    """Build a local combo-subgraph of size ~Q via iterative multi-hop BFS.

    Ported from ``GraphComBO/search/self_combograph.py`` but uses an
    iterative loop instead of recursion for stack safety.

    Returns
    -------
    combo_subgraph : nx.Graph
        Nodes are sorted int-tuples of length k.
    n_hop : int
        Number of BFS hops actually used.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    combo_sub = nx.Graph()
    combo_sub.add_node(center)
    all_nodes: set = {center}
    frontier = [center]
    n_hop = 0

    for l in range(1, l_max + 1):
        new_nodes: List[tuple] = []
        for combo_node in frontier:
            for nbr in _find_combo_neighbors(graph, combo_node):
                combo_sub.add_edge(combo_node, nbr)   # idempotent for nx
                if nbr not in all_nodes:
                    all_nodes.add(nbr)
                    new_nodes.append(nbr)
            if combo_sub.number_of_nodes() >= Q:
                break

        if not new_nodes:
            n_hop = l
            break

        rng.shuffle(new_nodes)
        n_hop = l

        if combo_sub.number_of_nodes() >= Q:
            excess = combo_sub.number_of_nodes() - Q
            if excess > 0 and new_nodes:
                rm_count = min(excess, len(new_nodes))
                rm_idx = rng.choice(len(new_nodes), size=rm_count, replace=False)
                rm_nodes = [new_nodes[i] for i in rm_idx]
                combo_sub.remove_nodes_from(rm_nodes)
                for nd in rm_nodes:
                    all_nodes.discard(nd)
            break

        frontier = new_nodes

    # --- Vectorised intra-subgraph edge completion (from GraphComBO) -------
    node_list = list(combo_sub.nodes())
    n_nodes = len(node_list)
    if n_nodes > 1:
        k = len(center)
        A = np.asarray(nx.adjacency_matrix(graph).todense())
        X = np.array(node_list, dtype=np.int32)          # (Q, k)

        X1 = X[:, np.newaxis, :]                          # (Q, 1, k)
        X2 = X[np.newaxis, :, :]                          # (1, Q, k)
        X_concat = np.concatenate(
            [np.repeat(X1, n_nodes, axis=1),
             np.repeat(X2, n_nodes, axis=0)],
            axis=2,
        )                                                  # (Q, Q, 2k)
        X_concat.sort(axis=2)
        raw_adj = (np.sum(np.diff(X_concat, axis=2) != 0, axis=2) == k)
        raw_edges_idx = np.argwhere(np.triu(raw_adj, k=1))

        for ei, ej in raw_edges_idx:
            diff = list(set(node_list[ei]).symmetric_difference(set(node_list[ej])))
            if len(diff) == 2 and A[diff[0], diff[1]]:
                combo_sub.add_edge(node_list[ei], node_list[ej])

    print(f"[BOGraph] Combo-subgraph: {combo_sub.number_of_nodes()} nodes, "
          f"{combo_sub.number_of_edges()} edges, hops={n_hop}")
    return combo_sub, n_hop


# =========================================================================== #
#  Laplacian eigendecomposition
# =========================================================================== #

def _eigendecompose_laplacian(
    combo_subgraph: nx.Graph,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecompose the normalised Laplacian of *combo_subgraph*.

    Returns eigenvalues in [0, 1] and an (N, N) eigenvector matrix.
    """
    L = np.asarray(
        nx.normalized_laplacian_matrix(combo_subgraph).todense(),
        dtype=np.float64,
    )
    L /= 2.0                              # normalise eigenvalues to [0, 1]
    eigenvalues, eigenvecs = np.linalg.eigh(L)
    return eigenvalues, eigenvecs


# =========================================================================== #
#  Diffusion kernel
# =========================================================================== #

def _diffusion_kernel_matrix(
    eigenvalues: np.ndarray,
    eigenvecs: np.ndarray,
    lengthscale: float,
    idx1: np.ndarray,
    idx2: np.ndarray,
) -> np.ndarray:
    r"""Diffusion kernel: :math:`K_{ij}=\sum_m w_m\,\varphi_m(i)\,\varphi_m(j)`.

    where :math:`w_m = \exp(-\lambda_m \cdot \ell)\cdot M/\sum_m \exp(\dots)`.
    """
    weights = np.exp(-eigenvalues * lengthscale)
    w_sum = weights.sum()
    if w_sum > 1e-12:
        weights *= len(weights) / w_sum

    V1 = eigenvecs[idx1]              # (n1, M)
    V2 = eigenvecs[idx2]              # (n2, M)
    return (V1 * weights) @ V2.T     # (n1, n2)


def _kernel_diag(
    eigenvalues: np.ndarray,
    eigenvecs: np.ndarray,
    lengthscale: float,
    indices: np.ndarray,
) -> np.ndarray:
    """Diagonal of the diffusion kernel at *indices* (avoids full matrix)."""
    weights = np.exp(-eigenvalues * lengthscale)
    w_sum = weights.sum()
    if w_sum > 1e-12:
        weights *= len(weights) / w_sum
    V = eigenvecs[indices]
    return np.sum(V ** 2 * weights, axis=1)


# =========================================================================== #
#  GP posterior (numpy-only, Cholesky-based)
# =========================================================================== #

def _gp_posterior(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_cross: np.ndarray,
    K_diag: np.ndarray,
    noise_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form GP posterior mean and variance."""
    n = K_train.shape[0]
    Ky = K_train + noise_var * np.eye(n)
    try:
        L = cholesky(Ky, lower=True)
    except LinAlgError:
        Ky += 1e-6 * np.eye(n)
        L = cholesky(Ky, lower=True)

    alpha = cho_solve((L, True), y_train)
    mu = K_cross @ alpha

    v = cho_solve((L, True), K_cross.T)
    var = K_diag - np.einsum("ij,ji->i", K_cross, v)
    var = np.maximum(var, 1e-10)
    return mu, var


# =========================================================================== #
#  Expected Improvement
# =========================================================================== #

def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
) -> np.ndarray:
    """Closed-form EI: ``E[max(f(x) - f_best, 0)]``."""
    ei = np.zeros_like(mu)
    valid = sigma > 1e-10
    if not np.any(valid):
        return ei
    z = (mu[valid] - f_best) / sigma[valid]
    ei[valid] = (mu[valid] - f_best) * norm.cdf(z) + sigma[valid] * norm.pdf(z)
    return np.maximum(ei, 0.0)


# =========================================================================== #
#  GP hyperparameter optimisation
# =========================================================================== #

def _neg_log_marginal_likelihood(
    log_params: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvecs: np.ndarray,
    train_indices: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative log marginal likelihood (for scipy.optimize)."""
    ls, noise = np.exp(log_params)
    n = len(y)
    K = _diffusion_kernel_matrix(eigenvalues, eigenvecs, ls,
                                 train_indices, train_indices)
    Ky = K + noise * np.eye(n)
    try:
        L = cholesky(Ky, lower=True)
    except LinAlgError:
        return 1e10
    alpha = cho_solve((L, True), y)
    nll = (0.5 * y @ alpha
           + np.sum(np.log(np.diag(L)))
           + 0.5 * n * np.log(2.0 * np.pi))
    return float(nll)


def _optimize_hyperparams(
    eigenvalues: np.ndarray,
    eigenvecs: np.ndarray,
    train_indices: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[float, float]:
    """Optimise GP lengthscale and noise via marginal likelihood."""
    y_mean = y_train.mean()
    y_std = max(float(y_train.std()), 1e-8)
    y = (y_train - y_mean) / y_std

    result = minimize(
        _neg_log_marginal_likelihood,
        x0=np.array([0.0, -4.0]),
        args=(eigenvalues, eigenvecs, train_indices, y),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-8.0, 0.0)],
    )
    ls, noise = np.exp(result.x)
    return float(ls), float(noise)


# =========================================================================== #
#  Trust region (ported from GraphComBO)
# =========================================================================== #

@dataclass
class _TrustRegionState:
    """Tracks success/failure streaks and triggers shrink/restart."""

    n_nodes: int = 200
    n_nodes_min: int = 10
    n_nodes_max: int = 200
    failure_counter: int = 0
    shrink_counter: int = 0
    success_counter: int = 0
    fail_tol: int = 20
    succ_tol: int = 10
    shrink_tol: int = 5
    best_value: float = -float("inf")
    restart_triggered: bool = False
    shrink_triggered: bool = False
    trust_region_multiplier: float = 1.5


def _update_trust_region(state: _TrustRegionState, y_new: float) -> None:
    """Update counters after a new observation (mutates *state*)."""
    if y_new > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.shrink_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.failure_counter != 0 and state.failure_counter % state.shrink_tol == 0:
        state.shrink_triggered = True
        state.shrink_counter += 1

    if state.success_counter == state.succ_tol:
        state.n_nodes = int(min(
            state.trust_region_multiplier * state.n_nodes,
            state.n_nodes_max,
        ))
        state.success_counter = 0
    elif state.failure_counter == state.fail_tol:
        state.restart_triggered = True
        state.failure_counter = 0
        state.shrink_counter = 0

    state.best_value = max(state.best_value, y_new)
    if state.n_nodes < state.n_nodes_min:
        state.restart_triggered = True


# =========================================================================== #
#  BOGraphStrategy
# =========================================================================== #

@register_strategy("bo_graph")
class BOGraphStrategy(BaseStrategy):
    """Bayesian Optimisation over combinatorial subset space.

    Each call to :meth:`select` proposes a **complete** k-subset.
    After evaluating it, call :meth:`tell` with the resulting score
    before calling :meth:`select` again.

    Parameters
    ----------
    n_points : int
        Subset size *k*.
    seed : int
        Random seed.
    embedding_path : str
        Directory with ``avg_codes.npy`` / ``avg_codes_pids.npy``.
    epsilon_factor : float
        Base-graph threshold = ``mean_pairwise_dist * factor``.
    Q : int
        Combo-subgraph size (max candidates per BO step).
    max_radius : int
        Max BFS hops for combo-subgraph construction.
    kernel : str
        Kernel type (``"diffusion"`` only for now).
    fail_tol, succ_tol, shrink_tol : int
        Trust-region tolerances (see GraphComBO paper).
    """

    def __init__(
        self,
        n_points: int,
        seed: int = 42,
        embedding_path: str = "",
        epsilon_factor: float = 0.3,
        Q: int = 200,
        max_radius: int = 3,
        kernel: str = "diffusion",
        fail_tol: int = 20,
        succ_tol: int = 10,
        shrink_tol: int = 5,
        **kwargs,
    ):
        super().__init__(n_points, seed, **kwargs)
        self.embedding_path = embedding_path
        self.epsilon_factor = epsilon_factor
        self.Q = Q
        self.max_radius = max_radius
        self.kernel = kernel
        self.fail_tol = fail_tol
        self.succ_tol = succ_tol
        self.shrink_tol = shrink_tol

        # ---- internal state (persists across select/tell calls) ----
        self._rng = np.random.RandomState(seed)
        self._observations: List[Tuple[tuple, float]] = []
        self._trust_state: _TrustRegionState | None = None
        self._base_graph: nx.Graph | None = None
        self._combo_subgraph: nx.Graph | None = None
        self._cached_eigenbasis: Tuple[np.ndarray, np.ndarray] | None = None
        self._pid_to_idx: Dict[str, int] = {}
        self._idx_to_pid: Dict[int, str] = {}
        self._pool_pids: List[str] = []
        self._center: tuple | None = None
        self._n_hop: int | None = None
        self._initialized: bool = False
        self._bo_iteration: int = 0
        self._n_restarts: int = 0
        self._best_combo: tuple | None = None
        self._best_score: float = -float("inf")
        self._center_changed: bool = False

    # ------------------------------------------------------------------ #
    #  Initialisation helpers
    # ------------------------------------------------------------------ #

    def _lazy_init(
        self,
        pool_points: List[str],
        metadata: Dict[str, Any] | None,
    ) -> None:
        """Build base graph from embeddings (first call only)."""
        if self._initialized:
            return

        emb_path = self.embedding_path
        if not emb_path and metadata and "embedding_path" in metadata:
            emb_path = metadata["embedding_path"]
        if not emb_path:
            raise ValueError(
                "BOGraphStrategy requires an embedding_path (via constructor "
                "kwarg or metadata['embedding_path'])."
            )

        embeddings_raw, aligned_pids = load_embeddings(emb_path, pool_points)
        embeddings = global_scale(embeddings_raw)

        self._pid_to_idx = {p: i for i, p in enumerate(aligned_pids)}
        self._idx_to_pid = {i: p for i, p in enumerate(aligned_pids)}
        self._pool_pids = aligned_pids

        self._base_graph = _build_base_graph(embeddings, self.epsilon_factor)

        self._trust_state = _TrustRegionState(
            n_nodes=self.Q,
            n_nodes_max=self.Q,
            fail_tol=self.fail_tol,
            succ_tol=self.succ_tol,
            shrink_tol=self.shrink_tol,
        )

        self._initialized = True
        n_edges = self._base_graph.number_of_edges()
        n_nodes = self._base_graph.number_of_nodes()
        print(
            f"[BOGraph] Base graph: {n_nodes} nodes, {n_edges} edges, "
            f"mean degree {2 * n_edges / max(n_nodes, 1):.1f}"
        )

    def _rebuild_combo_subgraph(self) -> None:
        """(Re-)build combo-subgraph around ``self._center``."""
        self._combo_subgraph, self._n_hop = _build_combo_subgraph(
            self._base_graph,
            self._center,
            Q=self._trust_state.n_nodes,
            l_max=self.max_radius,
            rng=self._rng,
        )
        self._cached_eigenbasis = None
        self._center_changed = False

    def _random_combo_node(self, exclude: set | None = None) -> tuple:
        """Sample a random k-subset as a sorted int-tuple."""
        if exclude is None:
            exclude = set()
        n_pool = len(self._pool_pids)
        for _ in range(1000):
            indices = self._rng.choice(n_pool, self.n_points, replace=False)
            combo = tuple(sorted(indices.tolist()))
            if combo not in exclude:
                return combo
        raise RuntimeError(
            "Could not sample a unique random combo-node after 1000 tries."
        )

    def _combo_to_pids(self, combo_node: tuple) -> List[str]:
        return [self._idx_to_pid[i] for i in combo_node]

    def _pids_to_combo(self, pids: List[str]) -> tuple:
        return tuple(sorted(self._pid_to_idx[p] for p in pids))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def tell(self, subset: List[str], score: float) -> None:
        """Feed back an evaluation result.  Must be called after each
        :meth:`select` and before the next one.
        """
        if not self._initialized:
            raise RuntimeError("Must call select() before tell().")

        combo = self._pids_to_combo(subset)
        self._observations.append((combo, score))

        _update_trust_region(self._trust_state, score)

        if score > self._best_score:
            self._best_score = score
            prev_best = self._best_combo
            self._best_combo = combo
            if prev_best is not None and prev_best != combo:
                self._center_changed = True

    def select(
        self,
        pool_points: List[str],
        selected_points: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> SelectionResult:
        """Propose the next k-subset to evaluate."""
        self._lazy_init(pool_points, metadata)
        self._bo_iteration += 1
        observed_combos = {obs[0] for obs in self._observations}

        # --- (1) First call: random initialisation ----------------------
        if not self._observations:
            combo = self._random_combo_node()
            self._center = combo
            self._rebuild_combo_subgraph()
            return self._make_result(combo)

        # --- (2) Restart ------------------------------------------------
        if self._trust_state.restart_triggered:
            self._n_restarts += 1
            print(f"[BOGraph] Restart #{self._n_restarts}")
            combo = self._random_combo_node(exclude=observed_combos)
            self._center = combo
            self._trust_state = _TrustRegionState(
                n_nodes=self.Q,
                n_nodes_max=self.Q,
                fail_tol=self.fail_tol,
                succ_tol=self.succ_tol,
                shrink_tol=self.shrink_tol,
                best_value=self._best_score,
            )
            self._rebuild_combo_subgraph()
            return self._make_result(combo)

        # --- (3) Center moved → rebuild subgraph -----------------------
        if self._center_changed:
            self._center = self._best_combo
            self._rebuild_combo_subgraph()

        # --- (4) Shrink -------------------------------------------------
        if self._trust_state.shrink_triggered:
            if self._n_hop is not None and self._n_hop > 1:
                self._n_hop -= 1
                self._combo_subgraph = nx.ego_graph(
                    self._combo_subgraph, self._center, self._n_hop,
                )
                self._cached_eigenbasis = None
                print(
                    f"[BOGraph] Shrink → radius={self._n_hop}, "
                    f"nodes={self._combo_subgraph.number_of_nodes()}"
                )
            self._trust_state.shrink_triggered = False

        # --- (5) Ensure subgraph exists ---------------------------------
        if self._combo_subgraph is None:
            self._center = self._best_combo
            self._rebuild_combo_subgraph()

        # --- (6) Eigenbasis (cached) ------------------------------------
        if self._cached_eigenbasis is None:
            if not nx.is_connected(self._combo_subgraph):
                largest_cc = max(
                    nx.connected_components(self._combo_subgraph), key=len,
                )
                self._combo_subgraph = self._combo_subgraph.subgraph(
                    largest_cc
                ).copy()
            self._cached_eigenbasis = _eigendecompose_laplacian(
                self._combo_subgraph,
            )

        eigenvalues, eigenvecs = self._cached_eigenbasis

        # --- (7) Index mapping for current subgraph ---------------------
        node_list = list(self._combo_subgraph.nodes())
        combo_to_local = {n: i for i, n in enumerate(node_list)}
        local_to_combo = {i: n for i, n in enumerate(node_list)}

        # --- (8) Prune observations to local subgraph -------------------
        local_train_idx: List[int] = []
        local_y: List[float] = []
        for combo, score in self._observations:
            if combo in combo_to_local:
                local_train_idx.append(combo_to_local[combo])
                local_y.append(score)

        train_idx = np.array(local_train_idx, dtype=int)
        y_arr = np.array(local_y, dtype=np.float64)

        # --- (9) Fall back if too few local observations ----------------
        if len(y_arr) < 2:
            unqueried = [n for n in node_list if n not in observed_combos]
            if not unqueried:
                self._trust_state.restart_triggered = True
                return self._make_result(
                    self._random_combo_node(exclude=observed_combos),
                )
            combo = unqueried[self._rng.randint(len(unqueried))]
            return self._make_result(combo)

        # --- (10) Standardise y -----------------------------------------
        y_mean = y_arr.mean()
        y_std = max(float(y_arr.std()), 1e-8)
        y_standardised = (y_arr - y_mean) / y_std

        # --- (11) Optimise GP hyperparameters ---------------------------
        ls, noise = _optimize_hyperparams(
            eigenvalues, eigenvecs, train_idx, y_arr,
        )

        # --- (12) Candidate (unqueried) nodes ---------------------------
        cand_local = np.array(
            [i for i, n in enumerate(node_list) if n not in observed_combos],
            dtype=int,
        )
        if len(cand_local) == 0:
            self._trust_state.restart_triggered = True
            print("[BOGraph] All subgraph nodes queried → restart next iter")
            return self._make_result(
                self._random_combo_node(exclude=observed_combos),
            )

        # --- (13) GP posterior ------------------------------------------
        K_train = _diffusion_kernel_matrix(
            eigenvalues, eigenvecs, ls, train_idx, train_idx,
        )
        K_cross = _diffusion_kernel_matrix(
            eigenvalues, eigenvecs, ls, cand_local, train_idx,
        )
        K_diag = _kernel_diag(eigenvalues, eigenvecs, ls, cand_local)

        mu, var = _gp_posterior(K_train, y_standardised, K_cross, K_diag, noise)
        sigma = np.sqrt(var)

        # --- (14) Expected Improvement ----------------------------------
        f_best = float(y_standardised.max())
        ei = _expected_improvement(mu, sigma, f_best)

        # --- (15) Select argmax EI --------------------------------------
        best_cand_pos = int(np.argmax(ei))
        best_local = cand_local[best_cand_pos]
        best_combo = local_to_combo[best_local]
        best_ei = float(ei[best_cand_pos])

        print(
            f"[BOGraph] Iter {self._bo_iteration}: EI={best_ei:.6f}, "
            f"ls={ls:.4f}, noise={noise:.6f}, "
            f"local_obs={len(y_arr)}, candidates={len(cand_local)}"
        )

        return self._make_result(
            best_combo, ei_value=best_ei, gp_ls=ls, gp_noise=noise,
        )

    # ------------------------------------------------------------------ #
    #  Result builder
    # ------------------------------------------------------------------ #

    def _make_result(
        self,
        combo: tuple,
        ei_value: float | None = None,
        gp_ls: float | None = None,
        gp_noise: float | None = None,
    ) -> SelectionResult:
        pids = self._combo_to_pids(combo)
        n_local = 0
        if self._combo_subgraph is not None:
            sg_nodes = set(self._combo_subgraph.nodes())
            n_local = sum(1 for o, _ in self._observations if o in sg_nodes)

        return SelectionResult(
            selected_points=sorted(pids),
            strategy_name=self.name,
            strategy_params={
                "n_points": self.n_points,
                "seed": self.seed,
                "embedding_path": self.embedding_path,
                "epsilon_factor": self.epsilon_factor,
                "Q": self.Q,
                "max_radius": self.max_radius,
                "kernel": self.kernel,
                "fail_tol": self.fail_tol,
            },
            metadata={
                "bo_iteration": self._bo_iteration,
                "ei_value": ei_value,
                "gp_lengthscale": gp_ls,
                "gp_noise_var": gp_noise,
                "best_observed_score": (
                    self._best_score
                    if self._best_score > -float("inf")
                    else None
                ),
                "n_local_observations": n_local,
                "subgraph_size": (
                    self._combo_subgraph.number_of_nodes()
                    if self._combo_subgraph
                    else 0
                ),
                "n_restarts": self._n_restarts,
                "n_total_observations": len(self._observations),
            },
        )


# =========================================================================== #
#  Standalone demo with REAL embeddings + mock objective
# =========================================================================== #

if __name__ == "__main__":
    """
    Runs the full BO-Graph pipeline on your real DayCent embeddings with a
    mock objective (no LSTM training) so you can visualise what the algorithm
    does.

    Usage:
      cd /users/6/mehta423/projects/daycent/lib
      conda activate wstatt
      python -m selection.bo_graph --T 30 --k 50 --out-dir bo_graph_demo
    """
    import argparse
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.spatial.distance import pdist as _pdist
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # ------------------------------------------------------------------
    #  CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Demo BO-Graph on real DayCent embeddings + mock objective.",
    )
    parser.add_argument(
        "--embedding-path", type=str,
        default="/users/6/mehta423/projects/daycent/output/inverse_1/eval",
    )
    parser.add_argument(
        "--pool-csv", type=str,
        default="/users/6/mehta423/projects/daycent/lib/configs/selection/"
                "point_sets/pool_exp5_400.csv",
    )
    parser.add_argument("--k", type=int, default=50,
                        help="Subset size (default: 50).")
    parser.add_argument("--T", type=int, default=30,
                        help="BO iterations (default: 30).")
    parser.add_argument("--Q", type=int, default=200,
                        help="Combo-subgraph size (default: 200).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon-factor", type=float, default=0.3)
    parser.add_argument("--max-radius", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="KMeans clusters for mock objective + colouring.")
    parser.add_argument("--out-dir", type=str, default="bo_graph_demo")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ------------------------------------------------------------------
    #  1. Load real embeddings + pool points
    # ------------------------------------------------------------------
    codes_all = np.load(os.path.join(args.embedding_path, "avg_codes.npy"))
    pids_all = np.load(os.path.join(args.embedding_path, "avg_codes_pids.npy"))
    pids_all = [str(p) for p in pids_all]
    pid_to_emb_idx = {p: i for i, p in enumerate(pids_all)}

    with open(args.pool_csv) as f:
        reader = csv.DictReader(f)
        pool_pids_raw = [row["point_id"] for row in reader]

    pool_pids = [p for p in pool_pids_raw if p in pid_to_emb_idx]
    pool_indices = [pid_to_emb_idx[p] for p in pool_pids]
    embeddings_raw = codes_all[pool_indices]          # (N_pool, 32)

    norms = np.linalg.norm(embeddings_raw, axis=1)
    embeddings = embeddings_raw / max(norms.mean(), 1e-8)

    N = len(pool_pids)
    print(f"Loaded {N} pool points with {embeddings.shape[1]}-D embeddings")
    print(f"  ({len(pids_all)} total embedding PIDs, "
          f"{len(pool_pids_raw)} in pool CSV)")

    # PCA → 2-D for plotting
    pca = PCA(n_components=2, random_state=args.seed)
    emb_2d = pca.fit_transform(embeddings_raw)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # KMeans clustering (mock objective + colouring)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings_raw)

    pid_to_idx = {p: i for i, p in enumerate(pool_pids)}
    idx_to_pid = {i: p for i, p in enumerate(pool_pids)}

    # ------------------------------------------------------------------
    #  2. Build base graph
    # ------------------------------------------------------------------
    base_graph = _build_base_graph(embeddings, args.epsilon_factor)
    n_edges = base_graph.number_of_edges()
    n_nodes_bg = base_graph.number_of_nodes()
    print(f"Base graph: {n_nodes_bg} nodes, {n_edges} edges, "
          f"mean degree {2 * n_edges / max(n_nodes_bg, 1):.1f}")

    # ------------------------------------------------------------------
    #  3. Mock objective
    #
    #  score(S) = 0.4 * cluster_coverage + 0.6 * diversity + noise
    #    cluster_coverage = fraction of K clusters "hit" by the subset
    #    diversity = mean pairwise distance in embedding space (normalised)
    # ------------------------------------------------------------------
    _emb_std = max(float(embeddings_raw.std()), 1e-8)

    def mock_objective(indices: tuple, noise_std: float = 0.02) -> float:
        idx = list(indices)
        clusters_hit = len(set(cluster_labels[idx]))
        coverage = clusters_hit / args.n_clusters
        if len(idx) > 1:
            sub = idx if len(idx) <= 30 else rng.choice(idx, 30, replace=False).tolist()
            diversity = float(_pdist(embeddings_raw[sub]).mean()) / _emb_std
        else:
            diversity = 0.0
        return float(0.4 * coverage + 0.6 * diversity + rng.randn() * noise_std)

    # ------------------------------------------------------------------
    #  4. Run ask-tell BO loop using building blocks directly
    # ------------------------------------------------------------------
    trust_state = _TrustRegionState(
        n_nodes=args.Q, n_nodes_max=args.Q,
        fail_tol=15, succ_tol=8, shrink_tol=4,
    )
    observations: List[Tuple[tuple, float]] = []
    bo_rng = np.random.RandomState(args.seed + 1)

    # initial random subset
    init_idx = bo_rng.choice(N, args.k, replace=False)
    center = tuple(sorted(init_idx.tolist()))
    score = mock_objective(center)
    observations.append((center, score))
    _update_trust_region(trust_state, score)
    best_combo, best_score = center, score

    combo_sub, n_hop = _build_combo_subgraph(
        base_graph, center, Q=args.Q, l_max=args.max_radius, rng=bo_rng,
    )
    eigenvalues, eigenvecs = _eigendecompose_laplacian(combo_sub)

    scores_per_iter = [score]
    best_per_iter   = [best_score]
    ei_per_iter     = [None]
    ls_per_iter     = [None]
    noise_per_iter  = [None]

    for t in range(1, args.T):
        # restart?
        if trust_state.restart_triggered:
            center_idx = bo_rng.choice(N, args.k, replace=False)
            center = tuple(sorted(center_idx.tolist()))
            trust_state = _TrustRegionState(
                n_nodes=args.Q, n_nodes_max=args.Q,
                fail_tol=15, succ_tol=8, shrink_tol=4,
                best_value=best_score,
            )
            combo_sub, n_hop = _build_combo_subgraph(
                base_graph, center, Q=args.Q, l_max=args.max_radius, rng=bo_rng,
            )
            eigenvalues, eigenvecs = _eigendecompose_laplacian(combo_sub)
            print(f"  [Restart at iter {t}]")

        # shrink?
        if trust_state.shrink_triggered:
            if n_hop > 1:
                n_hop -= 1
                combo_sub = nx.ego_graph(combo_sub, center, n_hop)
                eigenvalues, eigenvecs = _eigendecompose_laplacian(combo_sub)
            trust_state.shrink_triggered = False

        node_list = list(combo_sub.nodes())
        combo_to_local = {n: i for i, n in enumerate(node_list)}
        observed_set = {o[0] for o in observations}

        local_idx, local_y = [], []
        for combo, sc in observations:
            if combo in combo_to_local:
                local_idx.append(combo_to_local[combo])
                local_y.append(sc)

        if len(local_y) < 2:
            unqueried = [n for n in node_list if n not in observed_set]
            if unqueried:
                pick = unqueried[bo_rng.randint(len(unqueried))]
            else:
                pick_idx = bo_rng.choice(N, args.k, replace=False)
                pick = tuple(sorted(pick_idx.tolist()))
            sc = mock_objective(pick)
            observations.append((pick, sc))
            _update_trust_region(trust_state, sc)
            if sc > best_score:
                best_score, best_combo = sc, pick
            scores_per_iter.append(sc)
            best_per_iter.append(best_score)
            ei_per_iter.append(None)
            ls_per_iter.append(None)
            noise_per_iter.append(None)
            continue

        train_idx = np.array(local_idx, dtype=int)
        y_arr = np.array(local_y)
        y_mean = y_arr.mean()
        y_std  = max(float(y_arr.std()), 1e-8)
        y_std_arr = (y_arr - y_mean) / y_std

        ls, noise_var = _optimize_hyperparams(eigenvalues, eigenvecs, train_idx, y_arr)

        cand_local = np.array(
            [i for i, n in enumerate(node_list) if n not in observed_set],
            dtype=int,
        )
        if len(cand_local) == 0:
            trust_state.restart_triggered = True
            scores_per_iter.append(best_score)
            best_per_iter.append(best_score)
            ei_per_iter.append(0.0)
            ls_per_iter.append(ls)
            noise_per_iter.append(noise_var)
            continue

        K_train = _diffusion_kernel_matrix(eigenvalues, eigenvecs, ls, train_idx, train_idx)
        K_cross = _diffusion_kernel_matrix(eigenvalues, eigenvecs, ls, cand_local, train_idx)
        K_diag  = _kernel_diag(eigenvalues, eigenvecs, ls, cand_local)
        mu, var = _gp_posterior(K_train, y_std_arr, K_cross, K_diag, noise_var)
        sigma   = np.sqrt(var)

        f_best  = float(y_std_arr.max())
        ei      = _expected_improvement(mu, sigma, f_best)

        best_pos  = int(np.argmax(ei))
        pick      = node_list[cand_local[best_pos]]
        best_ei   = float(ei[best_pos])

        sc = mock_objective(pick)
        observations.append((pick, sc))
        _update_trust_region(trust_state, sc)
        if sc > best_score:
            best_score, best_combo = sc, pick
            center = pick
            combo_sub, n_hop = _build_combo_subgraph(
                base_graph, center, Q=trust_state.n_nodes,
                l_max=args.max_radius, rng=bo_rng,
            )
            eigenvalues, eigenvecs = _eigendecompose_laplacian(combo_sub)

        scores_per_iter.append(sc)
        best_per_iter.append(best_score)
        ei_per_iter.append(best_ei)
        ls_per_iter.append(ls)
        noise_per_iter.append(noise_var)

        if t % 5 == 0 or t == args.T - 1:
            print(f"  iter {t:3d}: score={sc:.4f}  best={best_score:.4f}  "
                  f"EI={best_ei:.5f}  ls={ls:.3f}")

    print(f"\nDone. Best score = {best_score:.4f}")
    print(f"Best subset PIDs (first 5): "
          f"{[idx_to_pid[i] for i in list(best_combo)[:5]]}")

    # ------------------------------------------------------------------
    #  5. Visualisations
    # ------------------------------------------------------------------

    # ---- (a) Base graph in PCA-2D, nodes coloured by cluster ----
    fig, ax = plt.subplots(figsize=(10, 10))
    edge_lines = [[emb_2d[u], emb_2d[v]] for u, v in base_graph.edges()]
    lc = LineCollection(edge_lines, colors="lightgrey", linewidths=0.4,
                        alpha=0.5, zorder=1)
    ax.add_collection(lc)
    scatter = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=cluster_labels, cmap="tab10", s=50, edgecolors="k",
        linewidths=0.4, zorder=2,
    )
    plt.colorbar(scatter, ax=ax, label="KMeans cluster", shrink=0.8)
    best_pts_2d = emb_2d[list(best_combo)]
    ax.scatter(best_pts_2d[:, 0], best_pts_2d[:, 1], s=160,
               facecolors="none", edgecolors="red", linewidths=2,
               zorder=4, label=f"Best subset (score={best_score:.3f})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"Base graph: {N} locations, {n_edges} edges "
                 f"(eps_factor={args.epsilon_factor})\n"
                 f"Red circles = best {args.k}-subset")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "1_base_graph.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out_dir}/1_base_graph.png")

    # ---- (b) Combo-subgraph (spring layout) ----
    fig, ax = plt.subplots(figsize=(10, 8))
    combo_pos = nx.spring_layout(combo_sub, seed=args.seed, k=0.8)
    queried_set = {o[0] for o in observations}
    node_colors = []
    for n in combo_sub.nodes():
        if n == best_combo:
            node_colors.append("red")
        elif n in queried_set:
            node_colors.append("dodgerblue")
        else:
            node_colors.append("lightgrey")
    nx.draw_networkx_edges(combo_sub, combo_pos, alpha=0.12, ax=ax)
    nx.draw_networkx_nodes(combo_sub, combo_pos, node_color=node_colors,
                           node_size=25, ax=ax)
    ax.set_title(
        f"Combo-subgraph: {combo_sub.number_of_nodes()} nodes, "
        f"{combo_sub.number_of_edges()} edges\n"
        f"Blue = queried, Red = best, Grey = unqueried"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "2_combo_subgraph.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out_dir}/2_combo_subgraph.png")

    # ---- (c) Convergence + GP diagnostics (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iters = np.arange(len(scores_per_iter))

    axes[0, 0].plot(iters, scores_per_iter, "o-", color="dodgerblue",
                    markersize=3, alpha=0.6, label="Per-iteration")
    axes[0, 0].plot(iters, best_per_iter, "s-", color="crimson",
                    linewidth=2, markersize=4, label="Best so far")
    axes[0, 0].set_xlabel("BO Iteration")
    axes[0, 0].set_ylabel("Mock Objective")
    axes[0, 0].set_title("BO Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    ei_vals = [(i, v) for i, v in enumerate(ei_per_iter) if v is not None]
    if ei_vals:
        ei_x, ei_y = zip(*ei_vals)
        axes[0, 1].plot(ei_x, ei_y, "o-", color="forestgreen", markersize=3)
    axes[0, 1].set_xlabel("BO Iteration")
    axes[0, 1].set_ylabel("Expected Improvement")
    axes[0, 1].set_title("Acquisition (EI)")
    axes[0, 1].grid(True, alpha=0.3)

    ls_vals = [(i, v) for i, v in enumerate(ls_per_iter) if v is not None]
    if ls_vals:
        ls_x, ls_y = zip(*ls_vals)
        axes[1, 0].plot(ls_x, ls_y, "o-", color="darkorange", markersize=3)
    axes[1, 0].set_xlabel("BO Iteration")
    axes[1, 0].set_ylabel("Lengthscale")
    axes[1, 0].set_title("GP Lengthscale")
    axes[1, 0].grid(True, alpha=0.3)

    nv = [(i, v) for i, v in enumerate(noise_per_iter) if v is not None]
    if nv:
        nv_x, nv_y = zip(*nv)
        axes[1, 1].plot(nv_x, nv_y, "o-", color="purple", markersize=3)
    axes[1, 1].set_xlabel("BO Iteration")
    axes[1, 1].set_ylabel("Noise Variance")
    axes[1, 1].set_title("GP Noise Variance")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"BO-Graph on Real Embeddings  (N={N}, k={args.k}, T={args.T})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.out_dir, "3_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out_dir}/3_convergence.png")

    # ---- (d) GP posterior mean + uncertainty on combo-subgraph ----
    node_list_final = list(combo_sub.nodes())
    combo_to_local_f = {n: i for i, n in enumerate(node_list_final)}
    all_idx = np.arange(len(node_list_final), dtype=int)

    lidx, ly = [], []
    for combo, sc in observations:
        if combo in combo_to_local_f:
            lidx.append(combo_to_local_f[combo])
            ly.append(sc)

    if len(ly) >= 2:
        tr_idx = np.array(lidx, dtype=int)
        y_a = np.array(ly)
        ym, ys = y_a.mean(), max(float(y_a.std()), 1e-8)
        y_s = (y_a - ym) / ys
        ls_f, nv_f = _optimize_hyperparams(eigenvalues, eigenvecs, tr_idx, y_a)
        Kt = _diffusion_kernel_matrix(eigenvalues, eigenvecs, ls_f, tr_idx, tr_idx)
        Kc = _diffusion_kernel_matrix(eigenvalues, eigenvecs, ls_f, all_idx, tr_idx)
        Kd = _kernel_diag(eigenvalues, eigenvecs, ls_f, all_idx)
        mu_f, var_f = _gp_posterior(Kt, y_s, Kc, Kd, nv_f)
        sigma_f = np.sqrt(var_f)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        nx.draw_networkx_edges(combo_sub, combo_pos, alpha=0.08, ax=axes[0])
        sc0 = axes[0].scatter(
            [combo_pos[n][0] for n in node_list_final],
            [combo_pos[n][1] for n in node_list_final],
            c=mu_f, cmap="RdYlGn", s=35, edgecolors="k", linewidths=0.2,
        )
        plt.colorbar(sc0, ax=axes[0], shrink=0.8)
        axes[0].set_title("GP Posterior Mean (standardised)")

        nx.draw_networkx_edges(combo_sub, combo_pos, alpha=0.08, ax=axes[1])
        sc1 = axes[1].scatter(
            [combo_pos[n][0] for n in node_list_final],
            [combo_pos[n][1] for n in node_list_final],
            c=sigma_f, cmap="YlOrRd", s=35, edgecolors="k", linewidths=0.2,
        )
        plt.colorbar(sc1, ax=axes[1], shrink=0.8)
        axes[1].set_title("GP Posterior Std Dev (uncertainty)")

        fig.suptitle("GP Surrogate over Combo-Subgraph",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(args.out_dir, "4_gp_surface.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: {args.out_dir}/4_gp_surface.png")

    # ---- (e) Location selection frequency heatmap ----
    fig, ax = plt.subplots(figsize=(10, 10))
    location_freq = np.zeros(N)
    for combo, _ in observations:
        for idx in combo:
            if idx < N:
                location_freq[idx] += 1
    edge_lines = [[emb_2d[u], emb_2d[v]] for u, v in base_graph.edges()]
    lc = LineCollection(edge_lines, colors="lightgrey", linewidths=0.4,
                        alpha=0.4, zorder=1)
    ax.add_collection(lc)
    sc = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=location_freq, cmap="hot_r", s=70, edgecolors="k",
        linewidths=0.4, zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="Times selected across BO iterations")
    best_pts_2d = emb_2d[list(best_combo)]
    ax.scatter(best_pts_2d[:, 0], best_pts_2d[:, 1], s=160,
               facecolors="none", edgecolors="lime", linewidths=2,
               zorder=4, label="Best subset")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Location Selection Frequency across BO Iterations")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "5_location_frequency.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out_dir}/5_location_frequency.png")

    # ---- (f) Base graph degree distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    degrees = [d for _, d in base_graph.degree()]
    axes[0].hist(degrees, bins=30, color="steelblue", edgecolor="k", alpha=0.8)
    axes[0].axvline(np.mean(degrees), color="red", linestyle="--",
                    label=f"mean={np.mean(degrees):.1f}")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Base Graph Degree Distribution")
    axes[0].legend()
    sc2 = axes[1].scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=degrees, cmap="viridis", s=50, edgecolors="k", linewidths=0.3,
    )
    plt.colorbar(sc2, ax=axes[1], label="Node degree")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Degree by Location (PCA space)")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "6_degree_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out_dir}/6_degree_distribution.png")

    print(f"\nAll 6 plots saved to: {args.out_dir}/")
