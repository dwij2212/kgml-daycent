"""
Point selection strategies for data-efficient DayCent emulation.

This module provides a framework for experimenting with different spatial
point selection strategies. Each strategy takes a pool of candidate points
and selects a subset for training, aiming to maximise model generalisability
with minimal data.

Usage:
    from selection import get_strategy

    strategy = get_strategy("random", n_points=50, seed=42)
    selected = strategy.select(pool_points, metadata)
"""

from .base import BaseStrategy, SelectionResult
from .random_strategy import RandomStrategy
from .stratified_strategy import StratifiedStrategy
from .registry import STRATEGY_REGISTRY, get_strategy, register_strategy

__all__ = [
    "BaseStrategy",
    "SelectionResult",
    "RandomStrategy",
    "StratifiedStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "register_strategy",
]
