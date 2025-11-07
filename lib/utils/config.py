"""
Experiment configuration management for DayCent modeling.
"""
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


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
    
    # Data selection
    scenario_ids: List[str] = field(default_factory=lambda: ['1'])
    train_quadrants: List[str] = field(default_factory=lambda: ['Q1 (SW)'])
    test_quadrants: List[str] = field(default_factory=lambda: ['Q2 (SE)', 'Q3 (NW)', 'Q4 (NE)'])
    
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
        data_config = DataConfig(**config_dict.get('data', {}))
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
