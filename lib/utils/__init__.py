"""
Utilities package for DayCent experiments.
"""
from .eval import evaluate
from .config import (
    # Emulator / selection configs
    SplitConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
    ExperimentConfig,
    # Inverse modelling configs
    InverseDataConfig,
    InverseTrainingConfig,
    InverseExperimentConfig,
)

# Shared evaluation metrics
from .metrics import (
    compute_regression_metrics,
    compute_masked_metrics,
    compute_per_channel_metrics,
    compute_emulator_metrics,
)

# Shared plotting utilities
from .plotting import (
    plot_timeseries,
    plot_dual_timeseries,
    plot_scatter,
    plot_scatter_grid,
    plot_bar_h,
    plot_budget_curve,
    plot_distribution_comparison,
    plot_spatial_points,
)

# Optimizer and scheduler factories
from .optim import (
    build_optimizer,
    build_scheduler,
    create_optimizer_and_scheduler,
    list_optimizers,
    list_schedulers,
)

# Training utilities
from .training import (
    setup_reproducibility,
    get_device,
    move_batch_to_device,
    compute_masked_mse,
    compute_losses,
    GradientClipper,
    CheckpointManager,
    WandbLogger,
    count_parameters,
    print_training_summary,
)
