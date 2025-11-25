"""
Main training script for DayCent experiments.

Usage:
    python train_experiment.py --config configs/experiment11.yaml
    python train_experiment.py --config configs/experiment11.yaml --skip-data-prep
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm
import joblib

from utils.config import ExperimentConfig
from data.preprocessing import prepare_experiment_data, prepare_data_for_datasetv2
from data import DayCentDataset, DayCentDatasetV2
from model import DayCentModel, MultiTaskLoss
from utils import evaluate


def setup_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(config: ExperimentConfig, prepared_data: dict):
    """
    Create train/val/test data loaders using DayCentDatasetV2.
    
    Creates separate dataset instances for each split since DataLoader holds 
    references (not copies) to dataset objects.
    
    Args:
        config: ExperimentConfig instance
        prepared_data: Dict with 'weather_df', 'management_df', 'output_df' from prepare_data_for_datasetv2
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset)
    """
    # Get split configurations
    train_config = config.data.get_train_config()
    val_config = config.data.get_val_config()
    test_config = config.data.get_test_config()
    
    if not train_config:
        raise ValueError("Training split configuration must be specified")
    
    # Create training dataset
    print("Creating training dataset...")
    train_dataset = DayCentDatasetV2(
        weather_df=prepared_data['weather_df'],
        management_df=prepared_data['management_df'],
        output_df=prepared_data['output_df'],
        init_cond_path=config.data.init_cond_file,
        split_config=train_config,
        year_emb_dim=16
    )
    
    train_size = len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    # Create validation dataset
    if val_config:
        print("Creating validation dataset...")
        val_dataset = DayCentDatasetV2(
            weather_df=prepared_data['weather_df'],
            management_df=prepared_data['management_df'],
            output_df=prepared_data['output_df'],
            init_cond_path=config.data.init_cond_file,
            split_config=val_config,
            year_emb_dim=16
        )
        val_size = len(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
    else:
        val_loader = None
        val_size = 0
    
    # Create test dataset
    if test_config:
        print("Creating test dataset...")
        test_dataset = DayCentDatasetV2(
            weather_df=prepared_data['weather_df'],
            management_df=prepared_data['management_df'],
            output_df=prepared_data['output_df'],
            init_cond_path=config.data.init_cond_file,
            split_config=test_config,
            year_emb_dim=16
        )
        test_size = len(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
    else:
        test_loader = None
        test_size = 0
    
    print(f"\nDataset sizes:")
    print(f"  Train: {train_size}")
    print(f"  Val:   {val_size}")
    print(f"  Test:  {test_size}")
    
    return train_loader, val_loader, test_loader, train_dataset


def initialize_model(config: ExperimentConfig, sample_data: dict):
    """Initialize model with correct dimensions."""
    # Infer dimensions from sample
    seq_feat_dim = sample_data["sequence"].shape[1]
    init_dim = sample_data["init_cond"].shape[0]
    year_dim = sample_data["year_enc"].shape[0]
    
    print(f"\nModel dimensions:")
    print(f"  Input features: {seq_feat_dim}")
    print(f"  Init cond dim:  {init_dim}")
    print(f"  Year enc dim:   {year_dim}")
    
    # Update config with inferred dimensions
    config.model.input_dim = seq_feat_dim
    config.model.init_dim = init_dim
    config.model.year_dim = year_dim
    
    # Create model
    model = DayCentModel(
        input_dim=seq_feat_dim, 
        init_dim=init_dim, 
        year_dim=year_dim
    )
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"  Device: {device}\n")
    
    return model, device


def initialize_wandb(config: ExperimentConfig):
    """Initialize Weights & Biases logging."""
    if not config.wandb.enabled:
        return None
    
    try:
        import wandb
        
        run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.name or config.experiment_id,
            notes=config.wandb.notes or config.description,
            tags=config.wandb.tags,
            config={
                "experiment_id": config.experiment_id,
                "learning_rate": config.training.learning_rate,
                "architecture": "LSTM with Attention",
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "num_scenarios": len(config.data.scenario_ids),
            }
        )
        print("Weights & Biases logging enabled\n")
        return run
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B logging.\n")
        return None

import matplotlib.pyplot as plt
import wandb

def plot_training_sample(pred_seq, true_seq, mask, scaler, epoch, batch_idx, save_dir=None):
    """
    Plots a single sample's SOMSC trajectory during training.
    """
    # 1. Inverse Transform (Un-normalize)
    # Assuming scaler was fit on [SOMSC, Yield, ...] 
    # scaler.mean_[0] is SOMSC mean, scaler.scale_[0] is SOMSC std
    somsc_mean = scaler.mean_[0]
    somsc_scale = scaler.scale_[0]
    
    pred_real = (pred_seq * somsc_scale) + somsc_mean
    true_real = (true_seq * somsc_scale) + somsc_mean
    
    # 2. Filter masked values (if mask is 0, don't plot or cut off)
    # Simply using the mask to filter arrays for plotting
    valid_indices = mask > 0
    
    # Convert to numpy and flatten
    pred_plot = pred_real[valid_indices]
    true_plot = true_real[valid_indices]
    months = np.arange(len(pred_plot))
    
    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, true_plot, 'o-', label='Ground Truth', color='#2E86AB', alpha=0.7)
    ax.plot(months, pred_plot, 's--', label='Prediction', color='#A23B72', alpha=0.9)
    
    ax.set_title(f"Training Snapshot (Epoch {epoch})", fontsize=12, fontweight='bold')
    ax.set_ylabel("SOMSC (g C/m²)")
    ax.set_xlabel("Month Index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample.png"))
    
    plt.close(fig) # Close to free memory

def train_epoch(model, train_loader, optimizer, device, config, scaler, epoch, mtl_loss):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
    
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        out = model(batch)
        
        # if batch_idx == 1:
        #     # Take index 0 from the batch
        #     # detached from graph, moved to cpu, converted to numpy
        #     sample_pred = out["somsc_pred"][0].detach().cpu().numpy()
        #     sample_true = batch["somsc"][0].detach().cpu().numpy()
        #     sample_mask = batch["somsc_mask"][0].detach().cpu().numpy()
            
        #     # Call helper function
        #     # Ensure scaler is passed down from main()
        #     plot_training_sample(
        #         sample_pred, sample_true, sample_mask, 
        #         scaler, epoch, batch_idx, 
        #         save_dir=os.path.join(config.output_dir, "train_plots")
        #     )
        # ---------------------
        # SOMSC loss
        somsc_target = batch["somsc"]
        somsc_mask = batch["somsc_mask"]
        somsc_loss = ((out["somsc_pred"] - somsc_target)**2 * somsc_mask).sum() / somsc_mask.sum()
        
        # Yield loss
        yield_target = batch["yield"]
        yield_mask = batch["yield_mask"]
        yield_loss = ((out["yield_pred"] - yield_target)**2 * yield_mask).sum() / yield_mask.sum()
        
        # Total loss
        # alpha = config.training.somsc_loss_weight
        # beta = config.training.yield_loss_weight
        # loss = alpha * somsc_loss + beta * yield_loss
        
        loss = mtl_loss(somsc_loss, yield_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch["sequence"].size(0)
    
    return total_loss / len(train_loader.dataset)


def train(config: ExperimentConfig, skip_data_prep: bool = False):
    """Main training function."""
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
    
    # Step 4: Initialize model
    print("\nStep 4: Initializing model...")
    sample = dataset[0]
    model, device = initialize_model(config, sample)
    
    # Step 5: Initialize optimizer and scheduler
    print("Step 5: Initializing optimizer and scheduler...")
    mtl_loss = MultiTaskLoss().to(device)
    # Add these params to your optimizer so they get updated!
    optimizer = optim.Adam(
        list(model.parameters()) + list(mtl_loss.parameters()), 
        lr=config.training.learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=1e-7
    )
    
    # Step 6: Initialize W&B
    print("\nStep 6: Initializing logging...")
    wandb_run = initialize_wandb(config)
    
    # Step 7: Training loop
    print(f"Step 7: Training for {config.training.epochs} epochs...")
    print(f"{'='*80}\n")
    
    # LOAD SCALER (Add this before the loop)
    scaler_path = config.get_scaler_path() # Or hardcode path if needed
    print(f"Loading scaler from {scaler_path} for plotting...")
    scaler = joblib.load(scaler_path)

    best_val_loss = float('inf')
    
    for epoch in range(config.training.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config, scaler, epoch, mtl_loss)
        
        # Validate (if val_loader exists)
        if val_loader:
            val_somsc_loss, val_yield_loss = evaluate(model, val_loader, device)
            val_total_loss = val_somsc_loss + val_yield_loss
        else:
            val_somsc_loss = val_yield_loss = val_total_loss = 0.0
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.training.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loader:
            print(f"  Val SOMSC:  {val_somsc_loss:.4f}")
            print(f"  Val Yield:  {val_yield_loss:.4f}")
            print(f"  Val Total:  {val_total_loss:.4f}")
        
        # Save best model
        if val_loader and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), config.get_model_path())
            print(f"  ✓ Saved best model (val_total_loss: {val_total_loss:.4f})")
        elif not val_loader:
            # If no validation set, save based on train loss
            torch.save(model.state_dict(), config.get_model_path())
            print(f"  ✓ Saved model checkpoint")
        
        print()
        
        # Log to W&B
        if wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            if val_loader:
                log_dict.update({
                    "val_somsc_loss": val_somsc_loss,
                    "val_yield_loss": val_yield_loss,
                    "val_total_loss": val_total_loss,
                })
            wandb_run.log(log_dict)
        
        # Update learning rate
        scheduler.step(val_total_loss if val_loader else train_loss)
    
    # Step 8: Final evaluation on test set
    if test_loader:
        print(f"{'='*80}")
        print("Step 8: Final evaluation on test set...")
        test_somsc_loss, test_yield_loss = evaluate(model, test_loader, device)
        print(f"  Test SOMSC Loss: {test_somsc_loss:.4f}")
        print(f"  Test Yield Loss: {test_yield_loss:.4f}")
        print(f"  Test Total Loss: {test_somsc_loss + test_yield_loss:.4f}")
        
        if wandb_run:
            wandb_run.log({
                "test_somsc_loss": test_somsc_loss,
                "test_yield_loss": test_yield_loss,
                "test_total_loss": test_somsc_loss + test_yield_loss,
            })
    
    if wandb_run:
        wandb_run.finish()
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best model saved to: {config.get_model_path()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train DayCent experiment')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment YAML config file')
    parser.add_argument('--skip-data-prep', action='store_true',
                      help='Skip data preparation and use existing .npy files')
    
    args = parser.parse_args()
    
    # Load configuration
    print(os.listdir('.'))
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = ExperimentConfig.from_yaml(args.config)
    
    # Run training
    train(config, skip_data_prep=args.skip_data_prep)


if __name__ == "__main__":
    main()
