"""
Random point selection strategy.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import BaseStrategy, SelectionResult
from .registry import register_strategy


@register_strategy("random")
class RandomStrategy(BaseStrategy):
    """Uniformly random subset selection (baseline).

    Parameters
    ----------
    n_points : int
        Number of points to select.
    seed : int
        Random seed for reproducibility.
    """

    def select(
        self,
        pool_points: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        self._validate_n(pool_points)

        rng = random.Random(self.seed)
        selected = rng.sample(pool_points, self.n_points)

        return SelectionResult(
            selected_points=sorted(selected),
            strategy_name=self.name,
            strategy_params={
                "n_points": self.n_points,
                "seed": self.seed,
            },
        )
