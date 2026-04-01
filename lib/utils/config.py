"""
Experiment configuration management for DayCent modeling.

Contains configuration dataclasses for all three research projects:
  - Emulator / selection:  ExperimentConfig (and its sub-configs)
  - Inverse modelling:     InverseExperimentConfig (and its sub-configs)
"""
import os
import random
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union


@dataclass
class SplitConfig:
    """Configuration for a single data split (train/val/test)."""
    # Scenario selection
    scenarios: Optional[List[str]] = None  # Explicit list of scenario IDs
    scenario_range: Optional[Dict[str, int]] = None  # e.g., {'start': 1, 'end': 100}
    
    # Point selection (one of these should be specified)
    quadrants: Optional[List[str]] = None  # e.g., ['Q1 (SW)', 'Q2 (SE)']
    points: Optional[List[str]] = None  # Explicit point IDs (takes precedence)
    
    # Year selection
    years: Optional[List[int]] = None  # Explicit list
    year_range: Optional[Dict[str, int]] = None  # e.g., {'start': 2000, 'end': 2020}
    
    def get_scenario_ids(self) -> List[str]:
        """Get list of scenario IDs from either explicit list or range."""
        if self.scenarios:
            return [str(s) for s in self.scenarios]
        elif self.scenario_range:
            start = self.scenario_range['start']
            end = self.scenario_range['end']
            return [str(i) for i in range(start, end + 1)]
        else:
            return []
    
    def get_years(self) -> List[int]:
        """Get list of years from either explicit list or range."""
        if self.years:
            return self.years
        elif self.year_range:
            start = self.year_range['start']
            end = self.year_range['end']
            return list(range(start, end + 1))
        else:
            return []
    
    def get_point_ids(self, points_lookup_path: str) -> List[str]:
        """
        Get list of point IDs.
        If explicit points are specified, use those.
        Otherwise, use quadrants to determine points.
        """
        if self.points:
            return [str(p) for p in self.points]
        elif self.quadrants:
            return self._get_points_from_quadrants(points_lookup_path, self.quadrants)
        else:
            return []
    
    def _get_points_from_quadrants(self, points_lookup_path: str, quadrants: List[str]) -> List[str]:
        """Get point IDs from quadrant names."""

        try:
            df = pd.read_excel(points_lookup_path)
        except Exception:
            df = pd.read_csv(points_lookup_path)
        
        # Calculate medians for splitting
        median_x = df['POINT_X'].median()
        median_y = df['POINT_Y'].median()
        
        # Create quadrants
        df['quadrant'] = 'Q1'
        df.loc[(df['POINT_X'] <= median_x) & (df['POINT_Y'] <= median_y), 'quadrant'] = 'Q1 (SW)'
        df.loc[(df['POINT_X'] > median_x) & (df['POINT_Y'] <= median_y), 'quadrant'] = 'Q2 (SE)'
        df.loc[(df['POINT_X'] <= median_x) & (df['POINT_Y'] > median_y), 'quadrant'] = 'Q3 (NW)'
        df.loc[(df['POINT_X'] > median_x) & (df['POINT_Y'] > median_y), 'quadrant'] = 'Q4 (NE)'
        
        # Filter by quadrants
        filtered = df[df['quadrant'].isin(quadrants)]
        return filtered['id'].astype(str).tolist()


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    base_dir: str = "/projects/standard/kumarv/shared/dwij/daycent/data/SAS_KGML_090925"
    input_dir: str = field(init=False)
    output_dir: str = field(init=False)
    weather_dir: str = field(init=False)
    points_lookup: str = field(init=False)
    init_cond_file: str = field(init=False)
    scenarios_file: str = field(init=False)
    experiment_number: int = 1
    
    # Legacy fields (kept for backward compatibility)
    scenario_ids: Optional[List[str]] = None
    train_quadrants: Optional[List[str]] = None
    test_quadrants: Optional[List[str]] = None
    
    # New split configurations
    train: Optional[SplitConfig] = None
    val: Optional[SplitConfig] = None
    test: Optional[SplitConfig] = None
    
    # Processing options
    max_workers: int = 10
    use_synthetic: bool = True  # True for Synthetic_10000, False for Realistic_8
    legacy_dir_structure: bool = True  # Whether to use legacy directory structure
    
    def __post_init__(self):
        if self.legacy_dir_structure:
            self.input_dir = os.path.join(self.base_dir, "InputData")
            output_subdir = "OutputData_Synthetic_10000" if self.use_synthetic else "OutputData_Realistic_8"
            self.output_dir = os.path.join(self.base_dir, output_subdir)
            self.weather_dir = os.path.join(self.input_dir, "WeatherData")
            self.points_lookup = os.path.join(self.base_dir, "SAS_points_lookup.csv")
            self.init_cond_file = os.path.join(self.input_dir, "initial_site_conditions.xlsx")
            scenarios_suffix = "Synthetic_10000" if self.use_synthetic else "Realistic_8"
            self.scenarios_file = os.path.join(self.input_dir, f"schedule_scenarios_all_{scenarios_suffix}.csv")

        else:
            self.input_dir = os.path.join(self.base_dir, "inputs")
            self.output_dir = os.path.join(self.base_dir, "outputs")
            self.weather_dir = os.path.join(self.input_dir, "Weather")
            self.points_lookup = os.path.join(self.input_dir, "Midwest_lookupTable.xlsx")
            self.init_cond_file = os.path.join(self.input_dir, "initial_site_conditions.xlsx")
            self.scenarios_file = os.path.join(self.input_dir, f"consolidated_management_scenarios_experiment_{self.experiment_number}.csv")
        
        self.scenario_ids = self.get_all_scenario_ids()
    
    def get_all_scenario_ids(self) -> List[str]:
        """Get all unique scenario IDs across all splits."""
        all_scenarios = set()
        
        # Add from legacy scenario_ids
        if self.scenario_ids:
            all_scenarios.update([str(s) for s in self.scenario_ids])
        
        # Add from splits
        for split in [self.train, self.val, self.test]:
            if split:
                all_scenarios.update(split.get_scenario_ids())
        
        return sorted(list(all_scenarios))
    
    def get_all_point_ids(self) -> List[str]:
        """Get all unique point IDs across all splits."""
        all_points = set()
        
        for split in [self.train, self.val, self.test]:
            if split:
                points = split.get_point_ids(self.points_lookup)
                all_points.update(points)
        
        return sorted(list(all_points))
    
    def get_train_config(self) -> Dict[str, Any]:
        """Get training configuration in format expected by DayCentDatasetV2."""
        if not self.train:
            return None
        
        return {
            'scenarios': self.train.get_scenario_ids(),
            'points': self.train.get_point_ids(self.points_lookup),
            'years': self.train.get_years()
        }
    
    def get_val_config(self) -> Dict[str, Any]:
        """Get validation configuration in format expected by DayCentDatasetV2."""
        if not self.val:
            return None
        
        return {
            'scenarios': self.val.get_scenario_ids(),
            'points': self.val.get_point_ids(self.points_lookup),
            'years': self.val.get_years()
        }
    
    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration in format expected by DayCentDatasetV2."""
        if not self.test:
            return None
        
        return {
            'scenarios': self.test.get_scenario_ids(),
            'points': self.test.get_point_ids(self.points_lookup),
            'years': self.test.get_years()
        }


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "transformer"  # Options: 'daycent', 'transformer'

    # Transformer-specific hyperparameters
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512

    # Will be inferred from data
    input_dim: Optional[int] = None
    init_dim: Optional[int] = None
    year_dim: Optional[int] = None
    
    # daycent-specific hyperparameters
    hidden_dim: int = 128
    latent_dim: int = 32
    lstm_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 2048
    epochs: int = 100
    device: str = "cuda:0"
    patience: int = 25  # For early stopping

    # Loss weights
    somsc_loss_weight: float = 1.0
    yield_loss_weight: float = 1.0
    somsc_abs_loss_weight: float = 1.0
    somsc_delta_loss_weight: float = 2.0
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Optimizer configuration (dict-based for flexibility)
    # Example: {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        'name': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999]
    })
    
    # Scheduler configuration (dict-based for flexibility)
    # Example: {'name': 'warmup_cosine', 'warmup_ratio': 0.1}
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        'name': 'warmup_linear',
        'warmup_ratio': 0.1
    })
    
    # Legacy fields (kept for backward compatibility)
    learning_rate: float = 1e-3  # Fallback if optimizer.lr not specified
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Data split (legacy)
    num_scenarios: int = 50
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Reproducibility
    random_seed: int = 42
    num_workers: int = 4
    
    def __post_init__(self):
        """Ensure optimizer has lr field for backward compatibility."""
        if isinstance(self.optimizer, dict) and 'lr' not in self.optimizer:
            self.optimizer['lr'] = self.learning_rate
        # Ensure optimizer is a dict (handle YAML loading)
        if not isinstance(self.optimizer, dict):
            self.optimizer = {'name': 'adamw', 'lr': self.learning_rate}
        if not isinstance(self.scheduler, dict):
            self.scheduler = {'name': 'warmup_linear', 'warmup_ratio': 0.1}


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "daycent"
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str
    description: str = ""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Optional override: if set, all ensemble members sharing the same
    # training data will reuse one preprocessed cache instead of each
    # member creating its own under data/{experiment_id}.
    shared_processed_dir: str | None = field(default=None)

    # Paths (auto-generated)
    processed_dir: str = field(init=False)
    output_dir: str = field(init=False)
    plots_dir: str = field(init=False)

    def __post_init__(self):
        if self.shared_processed_dir:
            self.processed_dir = self.shared_processed_dir
        else:
            self.processed_dir = f"/projects/standard/kumarv/shared/dwij/daycent/data/{self.experiment_id}"
        self.output_dir = f"/projects/standard/kumarv/shared/dwij/daycent/output/{self.experiment_id}"
        self.plots_dir = os.path.join(self.output_dir, "plots")

        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        data_dict = config_dict.get('data', {})
        
        # Parse split configs if they exist
        if 'train' in data_dict:
            data_dict['train'] = SplitConfig(**data_dict['train'])
        if 'val' in data_dict:
            data_dict['val'] = SplitConfig(**data_dict['val'])
        if 'test' in data_dict:
            data_dict['test'] = SplitConfig(**data_dict['test'])
        
        data_config = DataConfig(**data_dict)
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        wandb_config = WandbConfig(**config_dict.get('wandb', {}))
        
        return cls(
            experiment_id=config_dict['experiment_id'],
            description=config_dict.get('description', ''),
            data=data_config,
            model=model_config,
            training=training_config,
            wandb=wandb_config
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'experiment_id': self.experiment_id,
            'description': self.description,
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'wandb': asdict(self.wandb)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_train_npy_path(self) -> str:
        """Get path to training data .npy file."""
        return os.path.join(self.processed_dir, "train_X.npy")
    
    def get_test_npy_path(self) -> str:
        """Get path to test data .npy file."""
        return os.path.join(self.processed_dir, "test_X.npy")
    
    def get_train_output_path(self) -> str:
        """Get path to training output .npy file."""
        return os.path.join(self.processed_dir, "train_Y.npy")
    
    def get_test_output_path(self) -> str:
        """Get path to test output .npy file."""
        return os.path.join(self.processed_dir, "test_Y.npy")
    
    def get_scaler_path(self) -> str:
        """Get path to scaler pickle file."""
        return os.path.join(self.processed_dir, "scaler_Y.pkl")
    
    def get_model_path(self) -> str:
        """Get path to saved model."""
        return os.path.join(self.output_dir, "best_model.pth")


# ==========================================================================
# Inverse modelling configuration
# ==========================================================================

@dataclass
class InverseDataConfig:
    """Paths and split specification for inverse modelling data.

    Point splits can be specified two ways:
      Option A  —  percentage-based:  ``points = {source, train_ratio, val_ratio, test_ratio}``
      Option B  —  explicit lists:    ``points_train`` / ``points_val`` / ``points_test``
    """

    # Pre-processed CSV paths (from data_prep_inv_modelling.ipynb)
    driver_response_path: str = ""
    management_path: str = ""
    init_cond_path: str = ""

    # Scenario selection (shared across splits)
    scenario_range: Optional[Dict[str, int]] = None
    scenarios: Optional[List[int]] = None

    # Point split configuration — percentage-based
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
    _val_points:   List[str] = field(default_factory=list, init=False, repr=False)
    _test_points:  List[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def resolve_points(self, seed: int = 42) -> None:
        """Populate ``_train_points``, ``_val_points``, ``_test_points``.

        Must be called once after loading the config (done automatically by
        ``InverseExperimentConfig.from_yaml``).
        """
        # Explicit lists take precedence
        if self.points_train and self.points_val and self.points_test:
            self._train_points = [str(p) for p in self.points_train]
            self._val_points   = [str(p) for p in self.points_val]
            self._test_points  = [str(p) for p in self.points_test]
            return

        if self.points is None:
            raise ValueError(
                "Either specify explicit point lists (points_train/val/test) "
                "or a 'points' dict with source + ratios."
            )

        source  = self.points.get('source', 'csv')
        train_r = self.points.get('train_ratio', 0.7)
        val_r   = self.points.get('val_ratio', 0.15)
        test_r  = self.points.get('test_ratio', 0.15)

        assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_r + val_r + test_r}"

        if source == 'csv':
            df = pd.read_csv(self.driver_response_path, usecols=['point_id'])
            all_points = sorted(df['point_id'].astype(str).unique().tolist())
        elif source == 'management':
            df = pd.read_csv(self.management_path, usecols=['id'])
            all_points = sorted(df['id'].astype(str).unique().tolist())
        else:
            raise ValueError(f"Unknown points source: {source!r}")

        max_points = self.points.get('max_points', None)
        if max_points and max_points < len(all_points):
            rng = random.Random(seed)
            all_points = sorted(rng.sample(all_points, max_points))

        rng = random.Random(seed)
        rng.shuffle(all_points)
        n       = len(all_points)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        self._train_points = sorted(all_points[:n_train])
        self._val_points   = sorted(all_points[n_train:n_train + n_val])
        self._test_points  = sorted(all_points[n_train + n_val:])

        print(f"  Point split ({n} total): "
              f"train={len(self._train_points)}, "
              f"val={len(self._val_points)}, "
              f"test={len(self._test_points)}")

    # ------------------------------------------------------------------
    def get_split_config(self, split: str) -> Optional[Dict[str, Any]]:
        """Return ``{scenarios, points, years}`` for a split name."""
        scenarios = self.get_scenario_ids()
        if split == 'train':
            points = self._train_points
            years  = self._years_from_range(self.train_year_range)
        elif split == 'val':
            points = self._val_points
            years  = self._years_from_range(self.val_year_range)
        elif split == 'test':
            points = self._test_points
            years  = self._years_from_range(self.test_year_range)
        else:
            raise ValueError(f"Unknown split: {split!r}")

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
    architecture: str = "ATT_NL"   # "ATT_NL" or "LAST"

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
    model_dir:  str = field(init=False)
    result_dir: str = field(init=False)

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"/projects/standard/kumarv/shared/dwij/daycent/output/{self.experiment_id}"
        self.model_dir  = os.path.join(self.output_dir, "models")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.model_dir,  exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def get_model_path(self, name: str = "best") -> str:
        return os.path.join(self.model_dir, f"{name}.pt")

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InverseExperimentConfig':
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        data_cfg     = InverseDataConfig(**cfg.get('data', {}))
        training_cfg = InverseTrainingConfig(**cfg.get('training', {}))

        config = cls(
            experiment_id=cfg['experiment_id'],
            description=cfg.get('description', ''),
            data=data_cfg,
            training=training_cfg,
            output_dir=cfg.get('output_dir', ''),
        )
        config.data.resolve_points(seed=training_cfg.random_seed)
        return config
