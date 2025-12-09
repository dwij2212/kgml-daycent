"""
Main training script for DayCent experiments.

Usage:
    python train_experiment.py --config configs/experiment.yaml
    python train_experiment.py --config configs/experiment.yaml --skip-data-prep

This script uses modular components from the utils package:
- Model building via model registry
- Optimizer/scheduler via factories
- Common training utilities
"""
import argparse
import os
import sys
import torch
from tqdm import tqdm

from utils.config import ExperimentConfig
from data.preprocessing import prepare_data_for_datasetv2
from data.loader import create_data_loaders, create_dataset
from model import build_model, MultiTaskLoss
from utils import (
    # Training utilities
    setup_reproducibility,
    get_device,
    move_batch_to_device,
    compute_losses,
    GradientClipper,
    CheckpointManager,
    WandbLogger,
    print_training_summary,
    # Optimizer/scheduler
    create_optimizer_and_scheduler,
    # Evaluation
    evaluate,
)


def train_epoch(model, train_loader, optimizer, scheduler, device, config, 
                mtl_loss, grad_clipper, step_scheduler_per_batch=True):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Torch device
        config: ExperimentConfig
        mtl_loss: Multi-task loss module (or None for weighted sum)
        grad_clipper: GradientClipper instance
        step_scheduler_per_batch: If True, step scheduler after each batch
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Compute losses
        losses = compute_losses(outputs, batch)
        
        # Combine losses
        if mtl_loss is not None:
            loss = mtl_loss(losses['somsc_loss'], losses['yield_loss'])
        else:
            alpha = config.training.somsc_loss_weight
            beta = config.training.yield_loss_weight
            loss = alpha * losses['somsc_loss'] + beta * losses['yield_loss']
        
        loss.backward()
        
        grad_clipper(model.parameters())
        optimizer.step()
        
        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * batch["sequence"].size(0)
    
    return total_loss / len(train_loader.dataset)

def train(config: ExperimentConfig, skip_data_prep: bool = False):
    """
    Main training function.
    
    Args:
        config: ExperimentConfig instance
        skip_data_prep: Whether to skip data preparation (use cached data)
    """
    print(f"\n{'='*80}")
    print(f"Training Experiment: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"{'='*80}\n")
    
    # Step 1: Data preparation
    print("Step 1: Preparing data...")
    prepared_data = prepare_data_for_datasetv2(config)
    
    # Step 2: Setup reproducibility
    print("\nStep 2: Setting up reproducibility...")
    setup_reproducibility(config.training.random_seed)
    
    # Step 3: Create data loaders
    print("\nStep 3: Creating data loaders...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(config, prepared_data)
    
    # Step 4: Initialize model using registry
    print("\nStep 4: Initializing model...")
    sample = dataset[0]
    model = build_model(config.model, sample, verbose=True)
    
    device = get_device(config.training.device)
    model.to(device)
    print(f"  Device: {device}")
    
    # Step 5: Initialize optimizer and scheduler
    print("\nStep 5: Initializing optimizer and scheduler...")
    
    # Setup multi-task loss if using learned weights
    mtl_loss = None  # Set to MultiTaskLoss().to(device) if using learned weights
    
    # Get all trainable parameters
    params = list(model.parameters())
    if mtl_loss is not None:
        params += list(mtl_loss.parameters())
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.training.epochs
    
    # Build optimizer and scheduler using factories
    optimizer, scheduler = create_optimizer_and_scheduler(
        params=params,
        optimizer_config=config.training.optimizer,
        scheduler_config=config.training.scheduler,
        total_steps=total_steps,
        verbose=True
    )
    
    # Gradient clipper
    grad_clipper = GradientClipper(max_norm=config.training.grad_clip_norm)
    
    # Step 6: Initialize logging and checkpointing
    print("\nStep 6: Initializing logging...")
    wandb_logger = WandbLogger(config, enabled=config.wandb.enabled)
    checkpoint_manager = CheckpointManager(
        save_dir=config.output_dir,
        model=model,
        mode='min',
        save_best_only=True
    )
    
    # Print training summary
    print_training_summary(config, model, train_loader, val_loader)
    
    # Step 7: Training loop
    print(f"Step 7: Training for {config.training.epochs} epochs...")
    print(f"{'='*80}\n")
        
    for epoch in range(config.training.epochs):
        # Train

        # stepscheduler per batch depends on the scheduler type
        if scheduler:
            step_scheduler_per_batch = config.training.scheduler['name'] in ['warmup_cosine', 'warmup_linear']
        else:
            step_scheduler_per_batch = False
        
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            mtl_loss=mtl_loss,
            grad_clipper=grad_clipper,
            step_scheduler_per_batch=step_scheduler_per_batch
        )
        
        # Validate
        if val_loader:
            val_somsc_loss, val_yield_loss = evaluate(model, val_loader, device)
            alpha = config.training.somsc_loss_weight
            beta = config.training.yield_loss_weight
            val_total_loss = (alpha * val_somsc_loss) + (beta * val_yield_loss)
        else:
            val_somsc_loss = val_yield_loss = val_total_loss = 0.0
        
        if scheduler and not step_scheduler_per_batch:
            scheduler.step(val_total_loss if val_loader else train_loss)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config.training.epochs} | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loader:
            print(f"  Val Loss:   {val_total_loss:.4f} (SOMSC: {val_somsc_loss:.4f}, Yield: {val_yield_loss:.4f})")
        
        # Save best model
        if val_loader:
            saved = checkpoint_manager.save(
                value=val_total_loss,
                epoch=epoch,
                extra_state={
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
            )
            if saved:
                print(f"  ✓ Saved best model (val_total_loss: {val_total_loss:.4f})")

        # Log to W&B
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": current_lr,
        }
        if val_loader:
            log_dict.update({
                "val_somsc_loss": val_somsc_loss,
                "val_yield_loss": val_yield_loss,
                "val_total_loss": val_total_loss,
            })
        wandb_logger.log(log_dict)
    
    # Step 8: Final evaluation on test set
    if test_loader:
        print(f"\n{'='*80}")
        print("Step 8: Final evaluation on test set...")
        
        # Load best model for final evaluation
        checkpoint_manager.load_best()
        
        test_somsc_loss, test_yield_loss = evaluate(model, test_loader, device)
        print(f"  Test SOMSC Loss: {test_somsc_loss:.4f}")
        print(f"  Test Yield Loss: {test_yield_loss:.4f}")
        print(f"  Test Total Loss: {test_somsc_loss + test_yield_loss:.4f}")
    
    wandb_logger.finish()
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best model saved to: {config.get_model_path()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train DayCent experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment YAML config file')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation and use existing cached data')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = ExperimentConfig.from_yaml(args.config)
    
    # Run training
    train(config, skip_data_prep=args.skip_data_prep)


if __name__ == "__main__":
    main()
