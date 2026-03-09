"""
Main training script for DayCent emulator experiments.

Usage:
    python train_emulator.py --config configs/emulator/experiment_1.yaml
    python train_emulator.py --config configs/emulator/experiment_1.yaml --skip-data-prep

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
    flag = False
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        if flag:
            print("\n--- DEBUG: BATCH NORMALIZATION CHECK ---")
            
            # 1. Check Input Sequence (Batch, Time, Feat)
            seq = batch['sequence']
            print(f"Sequence Shape: {seq.shape}")
            print(f"  > Entire Seq  | Mean: {seq.mean().item():.3f} | Std: {seq.std().item():.3f} | Min: {seq.min().item():.3f} | Max: {seq.max().item():.3f}")
            
            # Check specific columns (assuming col 0 is Raw DOY, col 1-3 are Weather)
            print(f"  > Col 0 (DOY) | Mean: {seq[:,:,0].mean().item():.3f} | Max: {seq[:,:,0].max().item():.3f} (If > 1.0, huge scale mismatch with sin/cos!)")
            print(f"  > Col 1 (Tmax)| Mean: {seq[:,:,1].mean().item():.3f} | Min: {seq[:,:,1].std().item():.3f} | Max: {seq[:,:,1].max().item():.3f}")
            print(f"  > Col 2 (Tmin)| Mean: {seq[:,:,2].mean().item():.3f} | Min: {seq[:,:,2].std().item():.3f} | Max: {seq[:,:,2].max().item():.3f}")
            print(f"  > Col 3 (Precip)| Mean: {seq[:,:,3].mean().item():.3f} | Min: {seq[:,:,3].std().item():.3f} | Max: {seq[:,:,3].max().item():.3f}")

            # 2. Check Targets (Yield)
            y = batch['yield']
            print(f"Yield Shape:    {y.shape}")
            print(f"  > Yield Stats | Mean: {y.mean().item():.3f} | Std: {y.std().item():.3f} | Min: {y.min().item():.3f} | Max: {y.max().item():.3f}")
            if y.max().item() > 100:
                print("  ⚠️  WARNING: Yield values are very large. Loss gradients may explode.")

            # 3. Check Targets (SOMSC)
            som = batch['somsc']
            mask = batch['somsc_mask']
            valid_som = som[mask == 1] # Only check valid values
            if valid_som.numel() > 0:
                print(f"SOMSC Stats     | Mean: {valid_som.mean().item():.3f} | Std: {valid_som.std().item():.3f} | Max: {valid_som.max().item():.3f}")
            
            print("-" * 40)
            flag = False

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
        
        # Gradient clipping
        # grad_clipper(model.parameters())
        
        optimizer.step()
        
        # Step scheduler per batch (for warmup schedules)
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

    # stepscheduler per batch depends on the scheduler type
    step_scheduler_per_batch = config.training.scheduler['name'] in ['warmup_cosine', 'warmup_linear']
    if step_scheduler_per_batch:
        total_steps = steps_per_epoch * config.training.epochs
    else:
        total_steps = config.training.epochs  # For epoch-based schedulers

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

    counter = 0
        
    for epoch in range(config.training.epochs):
        # Train

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
            val_total_loss = config.training.somsc_loss_weight * val_somsc_loss + config.training.yield_loss_weight * val_yield_loss
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
                value=val_yield_loss,
                epoch=epoch,
                extra_state={
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
            )
            if saved:
                counter = 0
                print(f"  ✓ Saved best model (val_yield_loss: {val_yield_loss:.4f})")
            else:
                counter += 1

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

        if counter >= config.training.patience:
            print(f"Early stopping triggered after {counter} epochs without improvement.")
            break
    
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
