"""
Point selection strategies for data-efficient DayCent emulation.

Example:
    from selection import get_strategy

    strategy = get_strategy("random", n_points=50, seed=42)
    result = strategy.select(pool_points, metadata=metadata)
"""

from .base import BaseStrategy, SelectionResult
from .random_strategy import RandomStrategy
from .stratified_strategy import StratifiedStrategy
from .embedding_strategy import MaxDistStrategy, LCMDStrategy
from .bo_graph import BOGraphStrategy
from .registry import STRATEGY_REGISTRY, get_strategy, register_strategy

__all__ = [
    # One-shot strategies
    "BaseStrategy",
    "SelectionResult",
    "RandomStrategy",
    "StratifiedStrategy",
    "MaxDistStrategy",
    "LCMDStrategy",
    "BOGraphStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "register_strategy",
]
