"""
Acquisition function framework for active point selection.

Acquisition functions score unlabeled pool points using a trained model,
guiding which points to label next in an active learning loop.

Usage
-----
Implement a custom acquisition by subclassing BaseAcquisition and
decorating with @register_acquisition:

    @register_acquisition("my_strategy")
    class MyAcquisition(BaseAcquisition):
        def score(self, model_path, candidate_points, current_train, metadata):
            ...

Then retrieve it with get_acquisition("my_strategy").

Available acquisitions
----------------------
- "random"           : Random scores (baseline / placeholder for testing the loop).

Future acquisitions (not yet implemented)
------------------------------------------
- "mc_dropout"       : MCDropoutAcquisition — run N stochastic forward passes with
                       dropout active; score = mean predictive variance over yield + SOMSC.
- "latent_diversity" : LatentDiversityAcquisition — farthest-first traversal in model
                       latent space; score = distance to nearest current training point.
- "hybrid"           : Weighted sum of normalised mc_dropout + latent_diversity scores.
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type


# =========================================================================== #
#  Registry
# =========================================================================== #

ACQUISITION_REGISTRY: Dict[str, Type["BaseAcquisition"]] = {}


def register_acquisition(name: str):
    """Class decorator that registers an acquisition function by name."""
    def decorator(cls: Type["BaseAcquisition"]) -> Type["BaseAcquisition"]:
        ACQUISITION_REGISTRY[name] = cls
        return cls
    return decorator


def get_acquisition(name: str, **kwargs) -> "BaseAcquisition":
    """Instantiate an acquisition function by registry name."""
    if name not in ACQUISITION_REGISTRY:
        available = sorted(ACQUISITION_REGISTRY.keys())
        raise ValueError(
            f"Unknown acquisition '{name}'. Available: {available}"
        )
    return ACQUISITION_REGISTRY[name](**kwargs)


# =========================================================================== #
#  Base class
# =========================================================================== #

class BaseAcquisition(ABC):
    """Abstract base for acquisition functions used in active selection loops.

    An acquisition function scores each point in the unlabeled pool, with
    *higher scores indicating more informative points to label next*.

    Subclasses must implement :meth:`score`.
    """

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed

    @abstractmethod
    def score(
        self,
        model_path: str,
        candidate_points: List[str],
        current_train: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, float]:
        """Score each candidate point.

        Parameters
        ----------
        model_path : str
            Path to the trained model checkpoint (.pth file).
        candidate_points : list[str]
            Point IDs in the unlabeled pool to score.
        current_train : list[str]
            Point IDs already in the training set.
        metadata : dict
            Lookup table and initial conditions DataFrames.

        Returns
        -------
        dict[str, float]
            ``{point_id: score}`` for every point in *candidate_points*.
            Higher score = more informative = higher selection priority.
        """
        ...

    def select_top(
        self,
        scores: Dict[str, float],
        n: int,
    ) -> List[str]:
        """Return the top-*n* point IDs by descending score."""
        ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
        return ranked[:n]


# =========================================================================== #
#  Built-in: Random (placeholder / baseline)
# =========================================================================== #

@register_acquisition("random")
class RandomAcquisition(BaseAcquisition):
    """Assigns random scores to candidate points.

    This serves as a baseline and as a placeholder for testing the active
    selection loop before real acquisition functions are implemented.
    The resulting selection is equivalent to random sampling at each round.
    """

    def score(
        self,
        model_path: str,
        candidate_points: List[str],
        current_train: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, float]:
        rng = random.Random(self.seed)
        return {pt: rng.random() for pt in candidate_points}


# =========================================================================== #
#  Future stubs (not yet implemented — raise NotImplementedError)
# =========================================================================== #

@register_acquisition("mc_dropout")
class MCDropoutAcquisition(BaseAcquisition):
    """MC Dropout uncertainty acquisition (not yet implemented).

    Planned behaviour
    -----------------
    1. Load model from *model_path* and enable training mode (dropout active).
    2. Run *n_mc_passes* stochastic forward passes for each candidate point.
    3. Compute predictive variance over yield and SOMSC across passes.
    4. Return mean variance as the acquisition score.

    Parameters
    ----------
    n_mc_passes : int
        Number of Monte Carlo dropout forward passes (default 20).
    seed : int
        Random seed.
    """

    def __init__(self, n_mc_passes: int = 20, seed: int = 42, **kwargs):
        super().__init__(seed=seed)
        self.n_mc_passes = n_mc_passes

    def score(
        self,
        model_path: str,
        candidate_points: List[str],
        current_train: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "MCDropoutAcquisition is not yet implemented. "
            "Use --acquisition random for testing the active loop."
        )


@register_acquisition("latent_diversity")
class LatentDiversityAcquisition(BaseAcquisition):
    """Farthest-first diversity acquisition in model latent space (not yet implemented).

    Planned behaviour
    -----------------
    1. Extract latent embeddings for all candidate + current_train points
       using the encoder of the trained model at *model_path*.
    2. For each candidate, compute the distance to its nearest current_train
       neighbour in the embedding space.
    3. Return that distance as the acquisition score (farthest = most diverse).

    Parameters
    ----------
    seed : int
        Random seed for tie-breaking.
    """

    def score(
        self,
        model_path: str,
        candidate_points: List[str],
        current_train: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "LatentDiversityAcquisition is not yet implemented. "
            "Use --acquisition random for testing the active loop."
        )
