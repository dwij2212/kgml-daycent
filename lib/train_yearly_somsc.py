"""
Training script for the yearly December SOMSC model.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from data.yearly import create_yearly_data_loaders, prepare_yearly_data
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


YEARLY_ABS_ANCHOR_WEIGHT = 0.05
MIN_DELTA_STD = 1e-6


def compute_train_delta_std(dataset) -> float:
    """
    Estimate the December-to-December SOMSC delta scale from the training split.

    The yearly objective is MSE on normalized deltas, so only the delta
    standard deviation matters here. Centering would cancel out in a symmetric
    MSE term because it would be applied to both prediction and target.
    """
    deltas = np.asarray(
        [
            dataset.december_somsc[(scenario_id, pid, year)]
            - dataset.december_somsc[(scenario_id, pid, year - 1)]
            for scenario_id, pid, year in dataset.samples
        ],
        dtype=np.float32,
    )
    if deltas.size == 0:
        raise ValueError("Cannot compute yearly delta scale from an empty training dataset.")

    delta_std = float(deltas.std())
    return max(delta_std, MIN_DELTA_STD)


def compute_yearly_somsc_loss(
    outputs,
    batch,
    delta_std: float,
    abs_weight: float = YEARLY_ABS_ANCHOR_WEIGHT,
):
    pred = outputs["somsc_pred"].reshape(-1)
    target = batch["somsc"].reshape(-1)
    prev = batch["prev_somsc_state"].reshape(-1)
    mask = batch["somsc_mask"].reshape(-1)

    delta_pred = pred - prev
    delta_true = target - prev
    delta_scale = torch.as_tensor(delta_std, dtype=pred.dtype, device=pred.device)

    loss_delta = compute_masked_mse(delta_pred / delta_scale, delta_true / delta_scale, mask)
    loss_abs = compute_masked_mse(pred, target, mask)
    return loss_delta + abs_weight * loss_abs


def evaluate_yearly(model, loader, device, delta_std, abs_weight=YEARLY_ABS_ANCHOR_WEIGHT):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss = compute_yearly_somsc_loss(
                outputs,
                batch,
                delta_std=delta_std,
                abs_weight=abs_weight,
            )

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
    delta_std,
    abs_weight=YEARLY_ABS_ANCHOR_WEIGHT,
    step_scheduler_per_batch=True,
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_yearly_somsc_loss(
            outputs,
            batch,
            delta_std=delta_std,
            abs_weight=abs_weight,
        )
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
    delta_std = compute_train_delta_std(dataset)
    print(f"  Training delta std (normalized SOMSC space): {delta_std:.6f}")

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
            delta_std=delta_std,
            step_scheduler_per_batch=step_scheduler_per_batch,
        )

        if val_loader:
            val_somsc_loss = evaluate_yearly(model, val_loader, device, delta_std=delta_std)
        else:
            val_somsc_loss = 0.0

        if scheduler and not step_scheduler_per_batch:
            scheduler.step(val_somsc_loss if val_loader else train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config.training.epochs} | LR: {current_lr:.2e}")
        print(f"  Train SOMSC Loss: {train_loss:.4f}")
        if val_loader:
            print(f"  Val SOMSC Loss:   {val_somsc_loss:.4f}")

        if val_loader:
            saved = checkpoint_manager.save(
                value=val_somsc_loss,
                epoch=epoch,
                extra_state={
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                },
            )
            if saved:
                patience_counter = 0
                print(f"  Saved best model (val_somsc_loss: {val_somsc_loss:.4f})")
            else:
                patience_counter += 1

        log_dict = {
            "epoch": epoch + 1,
            "train_somsc_loss": train_loss,
            "learning_rate": current_lr,
        }
        if val_loader:
            log_dict["val_somsc_loss"] = val_somsc_loss
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

        test_somsc_loss = evaluate_yearly(model, test_loader, device, delta_std=delta_std)
        print(f"  Test December SOMSC Loss: {test_somsc_loss:.4f}")

    wandb_logger.finish()

    print(f"\n{'=' * 80}")
    print("Yearly SOMSC training complete!")
    print(f"Best model saved to: {config.get_model_path()}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train yearly December SOMSC model")
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
