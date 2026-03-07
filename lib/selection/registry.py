"""
Strategy registry – maps string names to strategy classes.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(name: str):
    """Decorator that registers a strategy class under *name*."""
    def decorator(cls: Type[BaseStrategy]):
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Instantiate a strategy by name.

    Parameters
    ----------
    name : str
        One of the registered strategy names (e.g. ``"random"``,
        ``"stratified"``).
    **kwargs
        Forwarded to the strategy constructor (must include ``n_points``).
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)
