"""
DayCent models package.

Models are registered via the @register_model decorator.
Use build_model() to instantiate models from config.
"""
from .registry import register_model, build_model, list_models, get_model_class

# Import models to trigger registration
from .daycent import DayCentModel, MultiTaskLoss
from .transformer import DayCentTransformer

__all__ = [
    # Registry functions
    'build_model', 

    'DayCentModel',
    'DayCentTransformer',

    'MultiTaskLoss',
]