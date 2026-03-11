"""
Base class and data structures for point selection strategies.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    """Container for the output of a selection strategy.

    Attributes:
        selected_points: List of point IDs chosen for training.
        strategy_name: Name of the strategy that produced this result.
        strategy_params: Dict of hyper-parameters used by the strategy.
        metadata: Any extra info the strategy wants to record (e.g. cluster
            assignments, feature importances, acquisition values, …).
    """
    selected_points: List[str]
    strategy_name: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.selected_points)

    # ---- serialisation helpers ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SelectionResult":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)


class BaseStrategy(ABC):
    """Abstract base class for all point selection strategies.

    Subclasses must implement ``select()``.  They receive:
      * *pool_points* – the list of **remaining** candidate point IDs
        (strings).  This shrinks at each selection step.
      * *selected_points* – the list of point IDs already selected in
        previous steps.  Strategies that are TP-aware (e.g. MaxDist,
        LCMD) use this to initialise distances / cluster centres.
      * *metadata* – a dict with at least:
          - ``"lookup_df"`` (pd.DataFrame) from the Midwest lookup table
          - ``"init_cond_df"`` (pd.DataFrame) initial site conditions
        Strategies that don't need metadata can ignore it.

    Convention: every strategy is instantiated with ``n_points`` (how many
    to pick **in this step**) plus any strategy-specific kwargs.
    """

    name: str = "base"  # override in subclass

    def __init__(self, n_points: int, seed: int = 42, **kwargs):
        self.n_points = n_points
        self.seed = seed
        self.kwargs = kwargs

    @abstractmethod
    def select(
        self,
        pool_points: List[str],
        selected_points: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Return a SelectionResult with ``n_points`` chosen from *pool_points*.

        Parameters
        ----------
        pool_points : list[str]
            Remaining candidate point IDs (excludes already-selected).
        selected_points : list[str] or None
            Points already chosen in prior selection steps.  Strategies
            that are TP-aware use these to seed distances / clusters.
        metadata : dict or None
            Auxiliary data (lookup tables, embeddings, etc.).
        """
        ...

    def _validate_n(self, pool_points: List[str]) -> None:
        if self.n_points > len(pool_points):
            raise ValueError(
                f"Requested {self.n_points} points but pool only has "
                f"{len(pool_points)}"
            )
