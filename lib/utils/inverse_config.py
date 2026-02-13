"""
Configuration for DayCent inverse modelling experiments.

Supports percentage-based spatial (point) splits:
    data:
      points:
        source: csv            # 'csv' reads unique points from the driver-response file
        train_ratio: 0.7
        val_ratio:   0.15
        test_ratio:  0.15
"""
import os
import random
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class InverseDataConfig:
    """Paths and split specification for inverse modelling data."""

    # Pre-processed CSV paths (from data_prep_inv_modelling.ipynb)
    driver_response_path: str = ""
    management_path: str = ""
    init_cond_path: str = ""

    # Scenario selection (shared across splits or per-split)
    scenario_range: Optional[Dict[str, int]] = None
    scenarios: Optional[List[int]] = None

    # ---------- Point split configuration ----------
    # Option A: percentage-based  (source: csv / lookup)
    # Option B: explicit lists     (points_train / points_val / points_test)
    points: Optional[Dict[str, Any]] = None  # {source, train_ratio, val_ratio, test_ratio}

    # Explicit per-split overrides (take precedence over percentage split)
    points_train: Optional[List[str]] = None
    points_val: Optional[List[str]] = None
    points_test: Optional[List[str]] = None

    # Year ranges per split
    train_year_range: Optional[Dict[str, int]] = None
    val_year_range: Optional[Dict[str, int]] = None
    test_year_range: Optional[Dict[str, int]] = None

    # ---------- resolved at runtime ----------
    _train_points: List[str] = field(default_factory=list, init=False, repr=False)
    _val_points: List[str] = field(default_factory=list, init=False, repr=False)
    _test_points: List[str] = field(default_factory=list, init=False, repr=False)

    # -----------------------------------------------------------------
    def get_scenario_ids(self) -> List[str]:
        if self.scenarios:
            return [str(s) for s in self.scenarios]
        elif self.scenario_range:
            return [str(i) for i in range(self.scenario_range['start'],
                                          self.scenario_range['end'] + 1)]
        return []

    @staticmethod
    def _years_from_range(yr: Optional[Dict[str, int]]) -> List[int]:
        if yr is None:
            return []
        return list(range(yr['start'], yr['end'] + 1))

    # -----------------------------------------------------------------
    def resolve_points(self, seed: int = 42):
        """
        Populate _train_points, _val_points, _test_points.
        Call this once after loading the config.
        """
        # Explicit lists take precedence
        if self.points_train and self.points_val and self.points_test:
            self._train_points = [str(p) for p in self.points_train]
            self._val_points = [str(p) for p in self.points_val]
            self._test_points = [str(p) for p in self.points_test]
            return

        if self.points is None:
            raise ValueError(
                "Either specify explicit point lists (points_train/val/test) "
                "or a 'points' dict with source + ratios."
            )

        source = self.points.get('source', 'csv')
        train_r = self.points.get('train_ratio', 0.7)
        val_r = self.points.get('val_ratio', 0.15)
        test_r = self.points.get('test_ratio', 0.15)

        assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_r + val_r + test_r}"

        # Discover all unique points
        if source == 'csv':
            df = pd.read_csv(self.driver_response_path, usecols=['point_id'])
            all_points = sorted(df['point_id'].astype(str).unique().tolist())
        elif source == 'management':
            df = pd.read_csv(self.management_path, usecols=['id'])
            all_points = sorted(df['id'].astype(str).unique().tolist())
        else:
            raise ValueError(f"Unknown points source: {source}")

        # Optional: subsample total points
        max_points = self.points.get('max_points', None)
        if max_points and max_points < len(all_points):
            rng = random.Random(seed)
            all_points = sorted(rng.sample(all_points, max_points))

        # Deterministic shuffle + split
        rng = random.Random(seed)
        rng.shuffle(all_points)

        n = len(all_points)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        # remainder goes to test
        self._train_points = sorted(all_points[:n_train])
        self._val_points = sorted(all_points[n_train:n_train + n_val])
        self._test_points = sorted(all_points[n_train + n_val:])

        print(f"  Point split ({n} total): "
              f"train={len(self._train_points)}, "
              f"val={len(self._val_points)}, "
              f"test={len(self._test_points)}")

    # -----------------------------------------------------------------
    def get_split_config(self, split: str) -> Dict[str, Any]:
        """Return {scenarios, points, years} for a split name."""
        scenarios = self.get_scenario_ids()

        if split == 'train':
            points = self._train_points
            years = self._years_from_range(self.train_year_range)
        elif split == 'val':
            points = self._val_points
            years = self._years_from_range(self.val_year_range)
        elif split == 'test':
            points = self._test_points
            years = self._years_from_range(self.test_year_range)
        else:
            raise ValueError(f"Unknown split: {split}")

        if not points:
            return None
        return {'scenarios': scenarios, 'points': points, 'years': years}

    def get_all_point_ids(self) -> List[str]:
        return sorted(set(self._train_points + self._val_points + self._test_points))


@dataclass
class InverseTrainingConfig:
    """Training hyper-parameters for inverse modelling."""
    batch_size: int = 128
    epochs: int = 200
    device: str = "cuda:0"

    # Model architecture
    code_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    architecture: str = "ATT_NL"  # "ATT_NL" or "LAST"

    # Optimiser
    learning_rate: float = 0.003
    weight_decay: float = 0.0

    # Loss weights
    recon_weight: float = 1.0
    static_weight: float = 1.0
    contrastive_weight: float = 1.0
    temperature: float = 1.0

    # Misc
    grad_clip_norm: float = 5.0
    random_seed: int = 42
    num_workers: int = 2
    year_emb_dim: int = 16


@dataclass
class InverseExperimentConfig:
    """Top-level config for an inverse modelling experiment."""
    experiment_id: str
    description: str = ""

    data: InverseDataConfig = field(default_factory=InverseDataConfig)
    training: InverseTrainingConfig = field(default_factory=InverseTrainingConfig)

    output_dir: str = ""
    model_dir: str = field(init=False)
    result_dir: str = field(init=False)

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"/users/6/mehta423/daycent/output/{self.experiment_id}"
        self.model_dir = os.path.join(self.output_dir, "models")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def get_model_path(self, name: str = "best") -> str:
        return os.path.join(self.model_dir, f"{name}.pt")

    # -----------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InverseExperimentConfig':
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        data_cfg = InverseDataConfig(**cfg.get('data', {}))
        training_cfg = InverseTrainingConfig(**cfg.get('training', {}))

        config = cls(
            experiment_id=cfg['experiment_id'],
            description=cfg.get('description', ''),
            data=data_cfg,
            training=training_cfg,
            output_dir=cfg.get('output_dir', ''),
        )

        # Resolve point splits
        config.data.resolve_points(seed=training_cfg.random_seed)

        return config
