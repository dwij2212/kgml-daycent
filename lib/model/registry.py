"""
Model registry and factory for DayCent experiments.

This module provides a centralized registry for models, making it easy to:
1. Add new models by decorating the class with @register_model
2. Instantiate models by name from config
3. Get available models and their configurations

Example:
    # Registering a new model (in your model file):
    @register_model("my_new_model")
    class MyNewModel(nn.Module):
        def __init__(self, input_dim, init_dim, year_dim, **kwargs):
            ...
    
    # Building a model from config (in train script):
    model = build_model(config.model, sample_data)
"""
from typing import Dict, Type, Callable, Any, Optional
import torch.nn as nn


# Global registry for models
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class.
    
    Args:
        name: The name to register the model under (used in config YAML)
    
    Returns:
        Decorator function
    
    Example:
        @register_model("transformer")
        class DayCentTransformer(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered!")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type[nn.Module]:
    """
    Get a model class by name.
    
    Args:
        name: Registered model name
    
    Returns:
        Model class
    
    Raises:
        ValueError: If model is not registered
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list:
    """Get list of all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def build_model(model_config, sample_data: dict, verbose: bool = True) -> nn.Module:
    """
    Build a model from configuration and sample data.
    
    This function:
    1. Infers input dimensions from sample data
    2. Looks up the model class by name
    3. Instantiates the model with appropriate parameters
    
    Args:
        model_config: ModelConfig dataclass with model hyperparameters
        sample_data: Sample batch from dataset for dimension inference
        verbose: Whether to print model information
    
    Returns:
        Instantiated model (not yet moved to device)
    """
    # Infer dimensions from sample
    seq_feat_dim = sample_data["sequence"].shape[1]
    init_dim = sample_data["init_cond"].shape[0]
    year_dim = sample_data["year_enc"].shape[0]
    
    if verbose:
        print(f"\nModel dimensions:")
        print(f"  Input features: {seq_feat_dim}")
        print(f"  Init cond dim:  {init_dim}")
        print(f"  Year enc dim:   {year_dim}")
    
    # Update config with inferred dimensions
    model_config.input_dim = seq_feat_dim
    model_config.init_dim = init_dim
    model_config.year_dim = year_dim
    
    # Get model class
    model_name = model_config.model_type
    model_cls = get_model_class(model_name)
    
    if verbose:
        print(f"  Model type: {model_name}")
    
    # Build kwargs from config
    # Common kwargs for all models
    kwargs = {
        'input_dim': seq_feat_dim,
        'init_dim': init_dim,
        'year_dim': year_dim,
    }
    
    # Add model-specific kwargs based on what the config has
    optional_kwargs = [
        'hidden_dim', 'latent_dim', 'lstm_layers',  # LSTM models
        'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',  # Transformer
        
    ]
    
    for key in optional_kwargs:
        if hasattr(model_config, key):
            value = getattr(model_config, key)
            if value is not None:
                kwargs[key] = value
    
    # Instantiate model
    model = model_cls(**kwargs)
    
    if verbose:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    return model
