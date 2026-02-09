"""
DayCent models package.

Models are registered via the @register_model decorator.
Use build_model() to instantiate models from config.
"""
from .registry import register_model, build_model, list_models, get_model_class

# Import models to trigger registration
from .daycent import DayCentModel, MultiTaskLoss
from .daycent_v2 import DayCentModelV2
from .transformer import DayCentTransformer
from .nlinear import NLinearSimple

__all__ = [
    # Registry functions
    'build_model', 

    'DayCentModel',
    'DayCentModelV2',
    'DayCentTransformer',
    'NLinearSimple',

    'MultiTaskLoss',
]