"""
Optimizer and scheduler factories for DayCent experiments.

This module provides flexible creation of optimizers and learning rate schedulers
from configuration, making it easy to experiment with different training strategies.

Example YAML config:
    training:
      optimizer:
        name: adamw
        lr: 0.001
        weight_decay: 0.01
        betas: [0.9, 0.999]
      
      scheduler:
        name: warmup_cosine
        warmup_ratio: 0.1
        min_lr: 1e-6
"""
from typing import Dict, Any, List, Optional, Callable
import math
import torch
from torch import optim
from torch.optim.lr_scheduler import (
    LambdaLR, 
    StepLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)


# ============================================================================
# Optimizer Registry
# ============================================================================

_OPTIMIZER_REGISTRY: Dict[str, Callable] = {}


def register_optimizer(name: str) -> Callable:
    """Decorator to register an optimizer builder function."""
    def decorator(fn: Callable) -> Callable:
        _OPTIMIZER_REGISTRY[name] = fn
        return fn
    return decorator


def list_optimizers() -> List[str]:
    """Get list of available optimizer names."""
    return list(_OPTIMIZER_REGISTRY.keys())


@register_optimizer("adam")
def build_adam(params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), 
               weight_decay: float = 0.0, **kwargs) -> optim.Optimizer:
    """Adam optimizer."""
    return optim.Adam(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


@register_optimizer("adamw")
def build_adamw(params, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                weight_decay: float = 0.01, **kwargs) -> optim.Optimizer:
    """AdamW optimizer with decoupled weight decay."""
    return optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


@register_optimizer("sgd")
def build_sgd(params, lr: float = 1e-2, momentum: float = 0.9,
              weight_decay: float = 0.0, nesterov: bool = True, **kwargs) -> optim.Optimizer:
    """SGD with momentum."""
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


@register_optimizer("rmsprop")
def build_rmsprop(params, lr: float = 1e-3, alpha: float = 0.99,
                  weight_decay: float = 0.0, momentum: float = 0.0, **kwargs) -> optim.Optimizer:
    """RMSprop optimizer."""
    return optim.RMSprop(params, lr=lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum)


def build_optimizer(params, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """
    Build an optimizer from configuration.
    
    Args:
        params: Model parameters (or list of param groups)
        optimizer_config: Dict with 'name' and optimizer-specific kwargs
        
    Returns:
        Configured optimizer
        
    Example:
        optimizer = build_optimizer(model.parameters(), {
            'name': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01
        })
    """
    config = optimizer_config.copy()
    name = config.pop('name', 'adamw')
    
    if name not in _OPTIMIZER_REGISTRY:
        available = list(_OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Optimizer '{name}' not found. Available: {available}")
    
    builder = _OPTIMIZER_REGISTRY[name]
    return builder(params, **config)


# ============================================================================
# Scheduler Registry  
# ============================================================================

_SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_scheduler(name: str) -> Callable:
    """Decorator to register a scheduler builder function."""
    def decorator(fn: Callable) -> Callable:
        _SCHEDULER_REGISTRY[name] = fn
        return fn
    return decorator


def list_schedulers() -> List[str]:
    """Get list of available scheduler names."""
    return list(_SCHEDULER_REGISTRY.keys())


@register_scheduler("none")
def build_none_scheduler(optimizer, total_steps: int, **kwargs):
    """No scheduling - constant learning rate."""
    return LambdaLR(optimizer, lambda step: 1.0)


@register_scheduler("warmup_linear")
def build_warmup_linear(optimizer, total_steps: int, 
                        warmup_ratio: float = 0.1, 
                        warmup_steps: Optional[int] = None,
                        **kwargs) -> LambdaLR:
    """
    Linear warmup followed by linear decay to 0.
    
    Args:
        optimizer: The optimizer
        total_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup (default: 0.1)
        warmup_steps: Explicit number of warmup steps (overrides warmup_ratio)
    """
    if warmup_steps is None:
        warmup_steps = int(warmup_ratio * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


@register_scheduler("warmup_cosine")
def build_warmup_cosine(optimizer, total_steps: int,
                        warmup_ratio: float = 0.1,
                        warmup_steps: Optional[int] = None,
                        min_lr_ratio: float = 0.0,
                        **kwargs) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer
        total_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        warmup_steps: Explicit warmup steps (overrides ratio)
        min_lr_ratio: Minimum LR as fraction of initial (default: 0 = decay to 0)
    """
    if warmup_steps is None:
        warmup_steps = int(warmup_ratio * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


@register_scheduler("warmup_constant")
def build_warmup_constant(optimizer, total_steps: int,
                          warmup_ratio: float = 0.1,
                          warmup_steps: Optional[int] = None,
                          **kwargs) -> LambdaLR:
    """
    Linear warmup followed by constant learning rate.
    
    Useful for fine-tuning or when you want to manually control decay.
    """
    if warmup_steps is None:
        warmup_steps = int(warmup_ratio * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


@register_scheduler("step")
def build_step_scheduler(optimizer, total_steps: int,
                         step_size: int = 10,
                         gamma: float = 0.1,
                         **kwargs) -> StepLR:
    """
    Step decay scheduler.
    
    Note: step_size is in EPOCHS, not steps. Use with caution.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("cosine")
def build_cosine_scheduler(optimizer, total_steps: int,
                           eta_min: float = 0.0,
                           **kwargs) -> CosineAnnealingLR:
    """
    Cosine annealing without warmup.
    
    Args:
        total_steps: Total steps (T_max for cosine)
        eta_min: Minimum learning rate
    """
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)


@register_scheduler("reduce_on_plateau")
def build_plateau_scheduler(optimizer, total_steps: int,
                            mode: str = 'min',
                            factor: float = 0.5,
                            patience: int = 5,
                            threshold: float = 1e-4,
                            **kwargs) -> ReduceLROnPlateau:
    """
    Reduce LR when a metric has stopped improving.
    
    Note: Requires calling scheduler.step(metric) instead of scheduler.step()
    """
    return ReduceLROnPlateau(
        optimizer, 
        mode=mode, 
        factor=factor, 
        patience=patience,
        threshold=threshold
    )


@register_scheduler("one_cycle")
def build_one_cycle(optimizer, total_steps: int,
                    max_lr: float = 0.01,
                    pct_start: float = 0.3,
                    div_factor: float = 25.0,
                    final_div_factor: float = 1e4,
                    **kwargs) -> OneCycleLR:
    """
    1Cycle learning rate policy.
    
    Increases LR from initial to max_lr, then decreases to very small value.
    Often gives good results with fewer epochs.
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )


@register_scheduler("cosine_restarts")
def build_cosine_restarts(optimizer, total_steps: int,
                          T_0: int = 10,
                          T_mult: int = 2,
                          eta_min: float = 0.0,
                          **kwargs) -> CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts.
    
    Args:
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase T_i after each restart
        eta_min: Minimum learning rate
    """
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )


def build_scheduler(optimizer, scheduler_config: Dict[str, Any], 
                    total_steps: int) -> Any:
    """
    Build a learning rate scheduler from configuration.
    
    Args:
        optimizer: The optimizer
        scheduler_config: Dict with 'name' and scheduler-specific kwargs
        total_steps: Total number of training steps
        
    Returns:
        Configured scheduler
        
    Example:
        scheduler = build_scheduler(optimizer, {
            'name': 'warmup_cosine',
            'warmup_ratio': 0.1,
            'min_lr_ratio': 0.01
        }, total_steps=10000)
    """
    config = scheduler_config.copy()
    name = config.pop('name', 'warmup_linear')
    
    if name not in _SCHEDULER_REGISTRY:
        available = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Scheduler '{name}' not found. Available: {available}")
    
    builder = _SCHEDULER_REGISTRY[name]
    return builder(optimizer, total_steps=total_steps, **config)


# ============================================================================
# High-level helper
# ============================================================================

def create_optimizer_and_scheduler(
    params,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    total_steps: int,
    verbose: bool = True
) -> tuple:
    """
    Create optimizer and scheduler together.
    
    This is the main entry point for creating optimization components.
    
    Args:
        params: Model parameters
        optimizer_config: Optimizer configuration dict
        scheduler_config: Scheduler configuration dict  
        total_steps: Total number of training steps
        verbose: Print configuration info
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = build_optimizer(params, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config, total_steps)
    
    if verbose:
        opt_name = optimizer_config.get('name', 'adamw')
        sched_name = scheduler_config.get('name', 'warmup_linear')
        lr = optimizer_config.get('lr', 1e-3)
        
        print(f"\nOptimization:")
        print(f"  Optimizer: {opt_name}")
        print(f"  Initial LR: {lr}")
        print(f"  Scheduler: {sched_name}")
        print(f"  Total steps: {total_steps}")
    
    return optimizer, scheduler
