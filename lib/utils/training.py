"""
Training utilities for DayCent experiments.

This module contains common training utilities that are shared between
different training scripts and experiments.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm


def setup_reproducibility(seed: int):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA (if available)
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get torch device, falling back to CPU if CUDA unavailable.
    
    Args:
        device_str: Device string (e.g., "cuda", "cuda:0", "cpu")
        
    Returns:
        torch.device
    """
    if "cuda" in device_str and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move a batch dictionary to the specified device.
    
    Only moves torch.Tensor values; other values are left unchanged.
    
    Args:
        batch: Dictionary containing batch data
        device: Target device
        
    Returns:
        Batch with tensors on target device
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in batch.items()
    }


def compute_masked_mse(pred: torch.Tensor, target: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked MSE loss.
    
    Args:
        pred: Predictions
        target: Ground truth
        mask: Binary mask (1 = valid, 0 = ignore)
        
    Returns:
        Scalar MSE loss over valid elements
    """
    mask_sum = mask.sum()
    if mask_sum.item() > 0:
        return ((pred - target) ** 2 * mask).sum() / mask_sum
    return torch.tensor(0.0, device=pred.device)


def compute_losses(outputs: Dict[str, torch.Tensor], 
                   batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute SOMSC and yield losses from model outputs.
    
    Args:
        outputs: Model output dictionary with 'somsc_pred' and 'yield_pred'
        batch: Batch dictionary with targets and masks
        
    Returns:
        Dictionary with 'somsc_loss', 'yield_loss' tensors
    """
    # SOMSC loss
    if "somsc_delta_pred" in outputs and "somsc_deltas" in batch:
        
        # Primary Objective: Match the Rate of Change (Deltas)
        somsc_loss = compute_masked_mse(
            outputs["somsc_delta_pred"], 
            batch["somsc_deltas"], 
            batch["somsc_delta_mask"]
        )
        
    else:
        # Fallback for legacy models (predicting absolute only)
        somsc_loss = compute_masked_mse(
            outputs["somsc_pred"], 
            batch["somsc"], 
            batch["somsc_mask"]
        )
    
    # Yield loss
    yield_pred = outputs["yield_pred"]
    yield_target = batch["yield"]
    yield_mask = batch["yield_mask"]
    yield_loss = compute_masked_mse(yield_pred, yield_target, yield_mask)
    
    return {
        'somsc_loss': somsc_loss,
        'yield_loss': yield_loss
    }


class GradientClipper:
    """Utility class for gradient clipping."""
    
    def __init__(self, max_norm: float = 1.0, clip_type: str = "norm"):
        """
        Args:
            max_norm: Maximum gradient norm/value
            clip_type: "norm" for clip_grad_norm_, "value" for clip_grad_value_
        """
        self.max_norm = max_norm
        self.clip_type = clip_type
    
    def __call__(self, parameters):
        """Clip gradients of parameters."""
        if self.clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.max_norm)


class CheckpointManager:
    """Manages model checkpointing during training."""
    
    def __init__(self, save_dir: str, model: nn.Module, 
                 mode: str = 'min', save_best_only: bool = True):
        """
        Args:
            save_dir: Directory to save checkpoints
            model: Model to checkpoint
            mode: 'min' or 'max' for determining best
            save_best_only: If True, only keep best model
        """
        self.save_dir = save_dir
        self.model = model
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)
    
    def is_better(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.mode == 'min':
            return value < self.best_value
        return value > self.best_value
    
    def save(self, value: float, epoch: int, 
             extra_state: Optional[Dict] = None) -> bool:
        """
        Save checkpoint if value is better.
        
        Args:
            value: Metric value to compare
            epoch: Current epoch number
            extra_state: Additional state to save (optimizer, scheduler, etc.)
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        if not self.is_better(value):
            return False
        
        self.best_value = value
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_value': self.best_value,
        }
        
        if extra_state:
            checkpoint.update(extra_state)
        
        # Save best model
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        
        # Optionally save epoch checkpoint
        if not self.save_best_only:
            epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
        
        return True
    
    def load_best(self) -> Dict:
        """Load the best checkpoint."""
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No checkpoint found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


class WandbLogger:
    """
    Wrapper for Weights & Biases logging.
    
    Provides a consistent interface that gracefully handles W&B being disabled.
    """
    
    def __init__(self, config, enabled: bool = True):
        """
        Args:
            config: ExperimentConfig instance
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.run = None
        
        if not enabled:
            return
            
        try:
            import wandb
            self.wandb = wandb
            
            wandb_config = config.wandb
            self.run = wandb.init(
                project=wandb_config.project,
                name=wandb_config.name or config.experiment_id,
                notes=wandb_config.notes or config.description,
                tags=wandb_config.tags,
                config={
                    "experiment_id": config.experiment_id,
                    "model_type": config.model.model_type,
                    "learning_rate": config.training.optimizer.get('lr', config.training.learning_rate),
                    "epochs": config.training.epochs,
                    "batch_size": config.training.batch_size,
                }
            )
            print("Weights & Biases logging enabled")
            
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def finish(self):
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with 'total' and 'trainable' parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable
    }


def print_training_summary(config, model: nn.Module, train_loader, val_loader=None):
    """Print a summary of the training configuration."""
    params = count_parameters(model)
    
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nExperiment: {config.experiment_id}")
    print(f"Description: {config.description}")
    
    print(f"\nModel:")
    print(f"  Type: {config.model.model_type}")
    print(f"  Total params: {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    
    print(f"\nData:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Device: {config.training.device}")
    
    opt_config = getattr(config.training, 'optimizer', {})
    sched_config = getattr(config.training, 'scheduler', {})
    print(f"  Optimizer: {opt_config.get('name', 'adamw')}")
    print(f"  Scheduler: {sched_config.get('name', 'warmup_linear')}")
    print(f"  Learning rate: {opt_config.get('lr', config.training.learning_rate)}")
    
    print(f"\n{'='*80}\n")
