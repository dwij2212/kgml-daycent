"""
Experiment configuration management for DayCent modeling.
"""
import os
import yaml
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
    base_dir: str = "/users/6/mehta423/daycent/data/SAS_KGML_090925"
    input_dir: str = field(init=False)
    output_dir: str = field(init=False)
    weather_dir: str = field(init=False)
    points_lookup: str = field(init=False)
    init_cond_file: str = field(init=False)
    scenarios_file: str = field(init=False)
    
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
    
    def __post_init__(self):
        self.input_dir = os.path.join(self.base_dir, "InputData")
        output_subdir = "OutputData_Synthetic_10000" if self.use_synthetic else "OutputData_Realistic_8"
        self.output_dir = os.path.join(self.base_dir, output_subdir)
        self.weather_dir = os.path.join(self.input_dir, "WeatherData")
        self.points_lookup = os.path.join(self.base_dir, "SAS_points_lookup.csv")
        self.init_cond_file = os.path.join(self.input_dir, "initial_site_conditions.xlsx")
        scenarios_suffix = "Synthetic_10000" if self.use_synthetic else "Realistic_8"
        self.scenarios_file = os.path.join(self.input_dir, f"schedule_scenarios_all_{scenarios_suffix}.csv")
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
    # Will be inferred from data
    input_dim: Optional[int] = None
    init_dim: Optional[int] = None
    year_dim: Optional[int] = None
    
    # Model hyperparameters (can be customized)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 2048
    epochs: int = 100
    learning_rate: float = 1e-2
    device: str = "cuda:2"
    
    # Loss weights
    somsc_loss_weight: float = 1.0
    yield_loss_weight: float = 1.0
    
    # Scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Data split
    num_scenarios: int = 50
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Reproducibility
    random_seed: int = 42
    num_workers: int = 4


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
    
    # Paths (auto-generated)
    processed_dir: str = field(init=False)
    output_dir: str = field(init=False)
    plots_dir: str = field(init=False)
    
    def __post_init__(self):
        self.processed_dir = f"/users/6/mehta423/daycent/data/{self.experiment_id}"
        self.output_dir = f"/users/6/mehta423/daycent/output/{self.experiment_id}"
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
