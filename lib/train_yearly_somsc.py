"""
Training script for the yearly December SOMSC state model.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from data.yearly import SOC_STATE_COLS, create_yearly_data_loaders, prepare_yearly_data
from model import build_model
from utils.config import ExperimentConfig
from utils.training import (
    CheckpointManager,
    GradientClipper,
    WandbLogger,
    compute_masked_mse,
    get_device,
    move_batch_to_device,
    print_training_summary,
    setup_reproducibility,
)
from utils.optim import create_optimizer_and_scheduler


MIN_DELTA_STD = 1e-6
VALID_TARGET_MODES = {"pool_deltas", "somsc_delta"}
SOMSC_POOL_COLS = SOC_STATE_COLS[:3]


def get_target_mode(model_config) -> str:
    target_mode = getattr(model_config, "target_mode", "pool_deltas")
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"model.target_mode must be one of {sorted(VALID_TARGET_MODES)}, "
            f"got {target_mode!r}."
        )
    return target_mode


def get_target_pool_cols(model_config) -> list:
    pool_cols = getattr(model_config, "target_pool_cols", None)
    if pool_cols is None:
        return list(SOC_STATE_COLS)

    unknown = [col for col in pool_cols if col not in SOC_STATE_COLS]
    if unknown:
        raise ValueError(f"Unknown target pool columns: {unknown}")
    if not pool_cols:
        raise ValueError("model.target_pool_cols cannot be empty.")
    return list(pool_cols)


def get_target_pool_indices(model_config) -> list:
    return [SOC_STATE_COLS.index(col) for col in get_target_pool_cols(model_config)]


def compute_train_yearly_scales(dataset, model_config) -> dict:
    """Estimate raw-unit pool and aggregate SOMSC scales from training samples."""
    target_mode = get_target_mode(model_config)
    target_pool_cols = get_target_pool_cols(model_config)
    target_pool_indices = get_target_pool_indices(model_config)

    pool_deltas = []
    somsc_deltas = []
    somsc_levels = []

    for scenario_id, pid, year in dataset.samples:
        prev_key = (scenario_id, pid, year - 1)
        target_key = (scenario_id, pid, year)
        prev_state = dataset.december_soc_state_raw[prev_key]
        target_state = dataset.december_soc_state_raw[target_key]

        pool_delta = target_state[target_pool_indices] - prev_state[target_pool_indices]
        prev_somsc = prev_state[:3].sum()
        target_somsc = target_state[:3].sum()

        pool_deltas.append(pool_delta)
        somsc_deltas.append(target_somsc - prev_somsc)
        somsc_levels.append(target_somsc)

    if not somsc_deltas:
        raise ValueError("Cannot compute yearly scales from an empty training dataset.")

    pool_deltas = np.asarray(pool_deltas, dtype=np.float32)
    somsc_deltas = np.asarray(somsc_deltas, dtype=np.float32)
    somsc_levels = np.asarray(somsc_levels, dtype=np.float32)

    loss_context = {
        "target_mode": target_mode,
        "target_pool_cols": target_pool_cols,
        "target_pool_indices": target_pool_indices,
        "predicts_complete_somsc": all(
            col in target_pool_cols for col in SOMSC_POOL_COLS
        ),
        "somsc_delta_std": max(float(somsc_deltas.std()), MIN_DELTA_STD),
        "somsc_level_std": max(float(somsc_levels.std()), MIN_DELTA_STD),
    }
    if target_mode == "pool_deltas":
        loss_context["pool_delta_std"] = np.maximum(
            pool_deltas.std(axis=0),
            MIN_DELTA_STD,
        ).astype(np.float32)
    return loss_context


def compute_yearly_loss(
    outputs,
    batch,
    loss_context: dict,
):
    somsc_pred = outputs["somsc_pred"].reshape(-1)
    somsc_true = batch["somsc"].reshape(-1)
    prev_somsc = batch["prev_somsc_state"].reshape(-1)
    somsc_mask = batch["somsc_mask"].reshape(-1)

    somsc_delta_scale = torch.as_tensor(
        loss_context["somsc_delta_std"],
        dtype=somsc_pred.dtype,
        device=somsc_pred.device,
    )
    somsc_level_scale = torch.as_tensor(
        loss_context["somsc_level_std"],
        dtype=somsc_pred.dtype,
        device=somsc_pred.device,
    )

    somsc_delta_loss = compute_masked_mse(
        (somsc_pred - prev_somsc) / somsc_delta_scale,
        (somsc_true - prev_somsc) / somsc_delta_scale,
        somsc_mask,
    )
    somsc_abs_anchor_loss = compute_masked_mse(
        somsc_pred / somsc_level_scale,
        somsc_true / somsc_level_scale,
        somsc_mask,
    )

    if loss_context["target_mode"] == "somsc_delta":
        return somsc_delta_loss + 0.05 * somsc_abs_anchor_loss

    target_pool_indices = loss_context["target_pool_indices"]
    pool_delta_pred = outputs["soc_state_delta_pred"][:, target_pool_indices]
    pool_delta_true = batch["soc_state_delta_raw"][:, target_pool_indices]
    mask = batch["somsc_mask"].reshape(-1, 1).expand_as(pool_delta_true)
    pool_scale = torch.as_tensor(
        loss_context["pool_delta_std"],
        dtype=pool_delta_pred.dtype,
        device=pool_delta_pred.device,
    ).reshape(1, -1)

    pool_delta_loss = compute_masked_mse(
        pool_delta_pred / pool_scale,
        pool_delta_true / pool_scale,
        mask,
    )

    # A subset-target model does not predict all components of SOMSC. Penalizing
    # its partial sum against total SOMSC change would corrupt the pool target.
    if not loss_context["predicts_complete_somsc"]:
        return pool_delta_loss

    return pool_delta_loss + 0.5 * somsc_delta_loss + 0.05 * somsc_abs_anchor_loss


def compute_loss_from_context(outputs, batch, loss_context: dict):
    return compute_yearly_loss(outputs, batch, loss_context)


def serialize_loss_context(loss_context: dict) -> dict:
    """Convert loss scales to checkpoint-friendly Python types."""
    serialized = {}
    for key, value in loss_context.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def evaluate_yearly(model, loader, device, loss_context: dict):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss = compute_loss_from_context(outputs, batch, loss_context)

            batch_size = batch["sequence"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    grad_clipper,
    loss_context,
    step_scheduler_per_batch=True,
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_loss_from_context(outputs, batch, loss_context)
        loss.backward()
        grad_clipper(model.parameters())
        optimizer.step()

        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch["sequence"].size(0)

    return total_loss / len(train_loader.dataset)


def train(config: ExperimentConfig, prepared_data: dict = None):
    print(f"\n{'=' * 80}")
    print(f"Training Yearly SOMSC Experiment: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"{'=' * 80}\n")

    if config.model.model_type != "yearly_somsc_state":
        raise ValueError(
            "The yearly training pipeline now only supports "
            "model.model_type='yearly_somsc_state'."
        )

    print("Step 1: Preparing yearly data...")
    if prepared_data is None:
        prepared_data = prepare_yearly_data(config)
    else:
        print("  (using pre-loaded yearly data, skipping disk I/O)")

    print("\nStep 2: Setting up reproducibility...")
    setup_reproducibility(config.training.random_seed)

    print("\nStep 3: Creating yearly data loaders...")
    train_loader, val_loader, test_loader, dataset = create_yearly_data_loaders(
        config, prepared_data
    )
    if len(dataset) == 0:
        raise ValueError("Yearly training dataset is empty after filtering invalid samples.")

    loss_context = compute_train_yearly_scales(dataset, config.model)
    print(f"  Target mode: {loss_context['target_mode']}")
    print(f"  Previous-state context: {config.model.prev_state_context}")
    if loss_context["target_mode"] == "pool_deltas":
        print("  Training SOC state delta stds (raw units):")
        for name, scale in zip(
            loss_context["target_pool_cols"],
            loss_context["pool_delta_std"],
        ):
            print(f"    {name}: {float(scale):.6f}")
        if not loss_context["predicts_complete_somsc"]:
            print("  Aggregate SOMSC loss disabled: target does not cover all soil pools.")
    print(f"  Training SOMSC delta std (raw units): {loss_context['somsc_delta_std']:.6f}")
    print(f"  Training SOMSC level std (raw units): {loss_context['somsc_level_std']:.6f}")

    print("\nStep 4: Initializing yearly model...")
    sample = dataset[0]
    model = build_model(config.model, sample, verbose=True)

    device = get_device(config.training.device)
    model.to(device)
    print(f"  Device: {device}")

    print("\nStep 5: Initializing optimizer and scheduler...")
    params = list(model.parameters())
    steps_per_epoch = len(train_loader)
    step_scheduler_per_batch = config.training.scheduler["name"] in [
        "warmup_cosine",
        "warmup_linear",
    ]
    total_steps = (
        steps_per_epoch * config.training.epochs
        if step_scheduler_per_batch
        else config.training.epochs
    )

    optimizer, scheduler = create_optimizer_and_scheduler(
        params=params,
        optimizer_config=config.training.optimizer,
        scheduler_config=config.training.scheduler,
        total_steps=total_steps,
        verbose=True,
    )
    grad_clipper = GradientClipper(max_norm=config.training.grad_clip_norm)

    print("\nStep 6: Initializing logging...")
    wandb_logger = WandbLogger(config, enabled=config.wandb.enabled)
    checkpoint_manager = CheckpointManager(
        save_dir=config.output_dir,
        model=model,
        mode="min",
        save_best_only=True,
    )

    print_training_summary(config, model, train_loader, val_loader)

    print(f"Step 7: Training for {config.training.epochs} epochs...")
    print(f"{'=' * 80}\n")

    patience_counter = 0

    for epoch in range(config.training.epochs):
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clipper=grad_clipper,
            loss_context=loss_context,
            step_scheduler_per_batch=step_scheduler_per_batch,
        )

        if val_loader:
            val_loss = evaluate_yearly(model, val_loader, device, loss_context)
        else:
            val_loss = 0.0

        if scheduler and not step_scheduler_per_batch:
            scheduler.step(val_loss if val_loader else train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config.training.epochs} | LR: {current_lr:.2e}")
        print(f"  Train yearly state loss: {train_loss:.4f}")
        if val_loader:
            print(f"  Val yearly state loss:   {val_loss:.4f}")

        if val_loader:
            saved = checkpoint_manager.save(
                value=val_loss,
                epoch=epoch,
                extra_state={
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "yearly_loss_context": serialize_loss_context(loss_context),
                },
            )
            if saved:
                patience_counter = 0
                print(f"  Saved best model (val_yearly_state_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

        log_dict = {
            "epoch": epoch + 1,
            "train_yearly_state_loss": train_loss,
            "learning_rate": current_lr,
        }
        if val_loader:
            log_dict["val_yearly_state_loss"] = val_loss
        wandb_logger.log(log_dict)

        if val_loader and patience_counter >= config.training.patience:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    if test_loader:
        print(f"\n{'=' * 80}")
        print("Step 8: Final evaluation on yearly test set...")
        if val_loader and checkpoint_manager.best_value < float("inf"):
            checkpoint_manager.load_best()

        test_loss = evaluate_yearly(model, test_loader, device, loss_context)
        print(f"  Test yearly state loss: {test_loss:.4f}")

    wandb_logger.finish()

    print(f"\n{'=' * 80}")
    print("Yearly SOMSC state training complete!")
    print(f"Best model saved to: {config.get_model_path()}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train yearly December SOMSC state model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
