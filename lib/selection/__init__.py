"""
Point selection strategies and active acquisition functions for data-efficient
DayCent emulation.

One-shot strategies
-------------------
    from selection import get_strategy

    strategy = get_strategy("random", n_points=50, seed=42)
    selected = strategy.select(pool_points, metadata)

Active acquisition functions (for iterative active loops)
----------------------------------------------------------
    from selection import get_acquisition

    acq = get_acquisition("random", seed=42)
    scores = acq.score(model_path, candidate_points, current_train, metadata)
    next_batch = acq.select_top(scores, n=25)
"""

from .base import BaseStrategy, SelectionResult
from .random_strategy import RandomStrategy
from .stratified_strategy import StratifiedStrategy
from .registry import STRATEGY_REGISTRY, get_strategy, register_strategy
from .acquisition import (
    BaseAcquisition,
    ACQUISITION_REGISTRY,
    get_acquisition,
    register_acquisition,
    RandomAcquisition,
    MCDropoutAcquisition,
    LatentDiversityAcquisition,
)

__all__ = [
    # One-shot strategies
    "BaseStrategy",
    "SelectionResult",
    "RandomStrategy",
    "StratifiedStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "register_strategy",
    # Active acquisition
    "BaseAcquisition",
    "ACQUISITION_REGISTRY",
    "get_acquisition",
    "register_acquisition",
    "RandomAcquisition",
    "MCDropoutAcquisition",
    "LatentDiversityAcquisition",
]
