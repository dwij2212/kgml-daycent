"""
Evaluation script for DayCent experiments using DayCentDatasetV2.

Usage:
    python eval_experiment.py --config configs/experiment11.yaml --split test
    python eval_experiment.py --config configs/experiment11.yaml --split all --num-samples 5
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import joblib

from utils.config import ExperimentConfig
from data.preprocessing import prepare_data_for_datasetv2
from data import DayCentDatasetV2
from model import DayCentModel


def collect_predictions(model, loader, device):
    """
    Run inference on a data loader and collect all predictions.
    
    Returns:
        dict: Contains predictions, ground truth, masks, and metadata
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    all_masks = []
    
    somsc_preds = []
    somsc_trues = []
    somsc_masks = []
    
    all_metadata = []
    
    print("Running inference...")
    for batch in tqdm(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(batch)
        
        # Collect yield predictions
        all_preds.append(outputs['yield_pred'].cpu().numpy())
        all_trues.append(batch['yield'].cpu().numpy())
        all_masks.append(batch['yield_mask'].cpu().numpy())
        
        # Collect SOMSC predictions
        somsc_preds.append(outputs['somsc_pred'].cpu().numpy())
        somsc_trues.append(batch['somsc'].cpu().numpy())
        somsc_masks.append(batch['somsc_mask'].cpu().numpy())
        
        # Collect metadata
        batch_size = len(batch['pid'])
        for i in range(batch_size):
            all_metadata.append({
                'scenario_id': batch['scenario_id'][i] if isinstance(batch['scenario_id'], list) else batch['scenario_id'],
                'pid': batch['pid'][i] if isinstance(batch['pid'], list) else batch['pid'],
                'year': batch['year'][i] if isinstance(batch['year'], list) else batch['year']
            })
    
    # Concatenate all batches
    pred_flat = np.concatenate(all_preds, axis=0)
    true_flat = np.concatenate(all_trues, axis=0)
    mask_flat = np.concatenate(all_masks, axis=0)
    
    somsc_pred_flat = np.concatenate(somsc_preds, axis=0)
    somsc_true_flat = np.concatenate(somsc_trues, axis=0)
    somsc_mask_flat = np.concatenate(somsc_masks, axis=0)
    
    # Squeeze SOMSC predictions if needed
    if somsc_pred_flat.ndim == 3 and somsc_pred_flat.shape[2] == 1:
        somsc_pred_flat = somsc_pred_flat.squeeze(-1)
    if somsc_true_flat.ndim == 3 and somsc_true_flat.shape[2] == 1:
        somsc_true_flat = somsc_true_flat.squeeze(-1)
    
    return {
        'yield_pred': pred_flat,
        'yield_true': true_flat,
        'yield_mask': mask_flat,
        'somsc_pred': somsc_pred_flat,
        'somsc_true': somsc_true_flat,
        'somsc_mask': somsc_mask_flat,
        'metadata': all_metadata
    }


def inverse_transform_predictions(predictions, scaler_path):
    """
    Apply inverse transformation to normalized predictions.
    
    Args:
        predictions: Dict from collect_predictions()
        scaler_path: Path to saved StandardScaler
    
    Returns:
        dict: Predictions in original scale
    """
    scaler_Y = joblib.load(scaler_path)
    
    # Get scale parameters
    cgrain_mean = scaler_Y.mean_[1]
    cgrain_scale = scaler_Y.scale_[1]
    somsc_mean = scaler_Y.mean_[0]
    somsc_scale = scaler_Y.scale_[0]
    
    print("\nInverse transforming predictions...")
    print(f"  CGRAIN - mean: {cgrain_mean:.4f}, scale: {cgrain_scale:.4f}")
    print(f"  SOMSC - mean: {somsc_mean:.4f}, scale: {somsc_scale:.4f}")
    
    # Inverse transform yield
    yield_pred = predictions['yield_pred'] * cgrain_scale + cgrain_mean
    yield_pred = yield_pred * predictions['yield_mask']
    yield_true = predictions['yield_true'] * cgrain_scale + cgrain_mean
    yield_true = yield_true * predictions['yield_mask']
    
    # Inverse transform SOMSC
    somsc_pred = predictions['somsc_pred'] * somsc_scale + somsc_mean
    somsc_pred = somsc_pred * predictions['somsc_mask']
    somsc_true = predictions['somsc_true'] * somsc_scale + somsc_mean
    somsc_true = somsc_true * predictions['somsc_mask']
    
    return {
        'yield_pred': yield_pred,
        'yield_true': yield_true,
        'yield_mask': predictions['yield_mask'],
        'somsc_pred': somsc_pred,
        'somsc_true': somsc_true,
        'somsc_mask': predictions['somsc_mask'],
        'metadata': predictions['metadata']
    }


def calculate_metrics(predictions):
    """
    Calculate evaluation metrics for predictions.
    
    Returns:
        dict: Metrics for yield and SOMSC
    """
    # Yield metrics (only on valid samples)
    valid_yield = predictions['yield_mask'] > 0
    yield_pred = predictions['yield_pred'][valid_yield]
    yield_true = predictions['yield_true'][valid_yield]
    
    yield_mse = np.mean((yield_pred - yield_true)**2)
    yield_rmse = np.sqrt(yield_mse)
    yield_mae = np.mean(np.abs(yield_pred - yield_true))
    
    # R² calculation
    ss_res = np.sum((yield_true - yield_pred)**2)
    ss_tot = np.sum((yield_true - np.mean(yield_true))**2)
    yield_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # SOMSC metrics (flatten and filter valid values)
    somsc_pred_flat = predictions['somsc_pred'].flatten()
    somsc_true_flat = predictions['somsc_true'].flatten()
    somsc_mask_flat = predictions['somsc_mask'].flatten()
    
    valid_somsc = somsc_mask_flat > 0
    somsc_pred = somsc_pred_flat[valid_somsc]
    somsc_true = somsc_true_flat[valid_somsc]
    
    somsc_mse = np.mean((somsc_pred - somsc_true)**2)
    somsc_rmse = np.sqrt(somsc_mse)
    somsc_mae = np.mean(np.abs(somsc_pred - somsc_true))
    
    # R² calculation
    ss_res = np.sum((somsc_true - somsc_pred)**2)
    ss_tot = np.sum((somsc_true - np.mean(somsc_true))**2)
    somsc_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'yield': {
            'mse': yield_mse,
            'rmse': yield_rmse,
            'mae': yield_mae,
            'r2': yield_r2,
            'n_samples': len(yield_pred)
        },
        'somsc': {
            'mse': somsc_mse,
            'rmse': somsc_rmse,
            'mae': somsc_mae,
            'r2': somsc_r2,
            'n_samples': len(somsc_pred)
        }
    }


def print_metrics(metrics, split_name):
    """Print evaluation metrics in a formatted table."""
    print(f"\n{'='*80}")
    print(f"EVALUATION METRICS - {split_name.upper()} SET")
    print(f"{'='*80}\n")
    
    print("YIELD PREDICTIONS:")
    print(f"  Samples:  {metrics['yield']['n_samples']}")
    print(f"  MSE:      {metrics['yield']['mse']:.4f}")
    print(f"  RMSE:     {metrics['yield']['rmse']:.4f}")
    print(f"  MAE:      {metrics['yield']['mae']:.4f}")
    print(f"  R²:       {metrics['yield']['r2']:.4f}")
    
    print("\nSOMSC PREDICTIONS:")
    print(f"  Samples:  {metrics['somsc']['n_samples']}")
    print(f"  MSE:      {metrics['somsc']['mse']:.4f}")
    print(f"  RMSE:     {metrics['somsc']['rmse']:.4f}")
    print(f"  MAE:      {metrics['somsc']['mae']:.4f}")
    print(f"  R²:       {metrics['somsc']['r2']:.4f}")
    
    print(f"\n{'='*80}\n")


def create_metadata_index(predictions):
    """
    Create an index mapping (scenario_id, pid) to sample indices for easy lookup.
    
    Returns:
        dict: Maps (scenario_id, pid) -> list of sample indices
    """
    index = {}
    for i, meta in enumerate(predictions['metadata']):
        key = (meta['scenario_id'], meta['pid'])
        if key not in index:
            index[key] = []
        index[key].append(i)
    return index


def plot_predictions(scenario_id, pid, sample_indices, predictions, plots_dir):
    """
    Plot yield and SOMSC predictions for a specific scenario-point combination.
    
    Args:
        scenario_id: Scenario ID
        pid: Point ID
        sample_indices: List of sample indices for this (scenario, point) combination
        predictions: Dict with predictions (already inverse transformed)
        plots_dir: Directory to save plots
    """
    # Extract data for these samples
    yield_pred = predictions['yield_pred'][sample_indices]
    yield_true = predictions['yield_true'][sample_indices]
    
    somsc_pred = predictions['somsc_pred'][sample_indices]  # (n_years, 12)
    somsc_true = predictions['somsc_true'][sample_indices]  # (n_years, 12)
    somsc_mask = predictions['somsc_mask'][sample_indices]  # (n_years, 12)
    
    # Flatten SOMSC for time series plot
    somsc_pred_flat = (somsc_pred * somsc_mask).flatten()
    somsc_true_flat = (somsc_true * somsc_mask).flatten()
    somsc_mask_flat = somsc_mask.flatten()
    
    # Filter non-zero SOMSC values
    nonzero_mask = somsc_mask_flat > 0
    somsc_pred_clean = somsc_pred_flat[nonzero_mask]
    somsc_true_clean = somsc_true_flat[nonzero_mask]
    
    # Calculate metrics
    yield_mse = np.mean((yield_pred - yield_true)**2)
    yield_r2 = 1 - (np.sum((yield_true - yield_pred)**2) / 
                    np.sum((yield_true - np.mean(yield_true))**2)) if len(yield_true) > 1 else 0
    
    if len(somsc_pred_clean) > 0:
        somsc_mse = np.mean((somsc_pred_clean - somsc_true_clean)**2)
        somsc_r2 = 1 - (np.sum((somsc_true_clean - somsc_pred_clean)**2) / 
                       np.sum((somsc_true_clean - np.mean(somsc_true_clean))**2)) if len(somsc_true_clean) > 1 else 0
    else:
        somsc_mse = 0
        somsc_r2 = 0
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # 1. Yield time series
    ax1 = fig.add_subplot(gs[0, 0])
    years = np.arange(len(yield_pred))
    ax1.plot(years, yield_true, 'o-', label='True', linewidth=2, markersize=6, color='#2E86AB')
    ax1.plot(years, yield_pred, 's--', label='Predicted', linewidth=2, markersize=6, 
             alpha=0.7, color='#A23B72')
    ax1.set_title(f'Yield Over Time\nMSE: {yield_mse:.2f}, R²: {yield_r2:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year Index', fontsize=11)
    ax1.set_ylabel('Yield (kg/ha)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. SOMSC time series
    ax2 = fig.add_subplot(gs[1, 0])
    months = np.arange(len(somsc_pred_clean))
    ax2.plot(months, somsc_true_clean, '-', label='True', linewidth=1.5, 
             alpha=0.8, color='#2E86AB')
    ax2.plot(months, somsc_pred_clean, '--', label='Predicted', linewidth=1.5, 
             alpha=0.8, color='#A23B72')
    ax2.set_title(f'SOMSC Over Time\nMSE: {somsc_mse:.2f}, R²: {somsc_r2:.3f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month Index', fontsize=11)
    ax2.set_ylabel('SOMSC (g C/m²)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Model Predictions - Scenario: {scenario_id}, Point: {pid}', 
                 fontsize=14, fontweight='bold')
    
    # Save plot
    scenario_dir = os.path.join(plots_dir, str(scenario_id))
    os.makedirs(scenario_dir, exist_ok=True)
    save_path = os.path.join(scenario_dir, f'point_{pid}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {save_path}")


def generate_sample_plots(predictions, plots_dir, num_scenarios=5, num_points_per_scenario=3):
    """
    Generate sample plots for a subset of scenarios and points.
    
    Args:
        predictions: Dict with predictions (already inverse transformed)
        plots_dir: Directory to save plots
        num_scenarios: Number of scenarios to plot
        num_points_per_scenario: Number of points per scenario
    """
    print(f"\n{'='*80}")
    print("GENERATING SAMPLE PLOTS")
    print(f"{'='*80}\n")
    
    # Create index for fast lookup
    index = create_metadata_index(predictions)
    
    # Get unique scenario-point combinations
    all_keys = list(index.keys())
    
    # Group by scenario
    scenario_groups = {}
    for scenario_id, pid in all_keys:
        if scenario_id not in scenario_groups:
            scenario_groups[scenario_id] = []
        scenario_groups[scenario_id].append(pid)
    
    # Sample scenarios
    all_scenarios = list(scenario_groups.keys())
    if len(all_scenarios) > num_scenarios:
        sampled_scenarios = random.sample(all_scenarios, num_scenarios)
    else:
        sampled_scenarios = all_scenarios
    
    print(f"Plotting {len(sampled_scenarios)} scenarios...")
    
    total_plots = 0
    for scenario_id in sampled_scenarios:
        pids = scenario_groups[scenario_id]
        
        # Sample points for this scenario
        if len(pids) > num_points_per_scenario:
            sampled_pids = random.sample(pids, num_points_per_scenario)
        else:
            sampled_pids = pids
        
        print(f"\nScenario {scenario_id}:")
        for pid in sampled_pids:
            sample_indices = index[(scenario_id, pid)]
            plot_predictions(scenario_id, pid, sample_indices, predictions, plots_dir)
            total_plots += 1
            break  # Only plot one point per scenario for brevity
    
    print(f"\n{'='*80}")
    print(f"Generated {total_plots} plots in: {plots_dir}")
    print(f"{'='*80}\n")


def save_predictions_to_csv(predictions, output_path):
    """
    Save predictions to CSV for further analysis.
    
    Args:
        predictions: Dict with predictions
        output_path: Path to save CSV file
    """
    print(f"\nSaving predictions to CSV: {output_path}")
    
    rows = []
    for i, meta in enumerate(predictions['metadata']):
        # Yield data
        row = {
            'scenario_id': meta['scenario_id'],
            'point_id': meta['pid'],
            'year': meta['year'],
            'yield_pred': predictions['yield_pred'][i],
            'yield_true': predictions['yield_true'][i],
            'yield_mask': predictions['yield_mask'][i],
        }
        
        # Add SOMSC monthly data
        for month in range(12):
            row[f'somsc_pred_m{month+1}'] = predictions['somsc_pred'][i, month]
            row[f'somsc_true_m{month+1}'] = predictions['somsc_true'][i, month]
            row[f'somsc_mask_m{month+1}'] = predictions['somsc_mask'][i, month]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} predictions")


def evaluate(config: ExperimentConfig, split: str = 'test', 
             num_samples: int = 5, save_csv: bool = True):
    """
    Main evaluation function.
    
    Args:
        config: ExperimentConfig instance
        split: Which split to evaluate ('train', 'val', 'test', or 'all')
        num_samples: Number of sample plots to generate
        save_csv: Whether to save predictions to CSV
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING EXPERIMENT: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    prepared_data = prepare_data_for_datasetv2(config)
    
    # Step 2: Determine which split to evaluate
    print(f"\nStep 2: Loading {split} dataset...")
    
    if split == 'train':
        split_config = config.data.get_train_config()
    elif split == 'val':
        split_config = config.data.get_val_config()
    elif split == 'test':
        split_config = config.data.get_test_config()
    elif split == 'all':
        # Create a combined split
        split_config = {
            'scenarios': config.data.get_all_scenario_ids(),
            'points': [],
            'years': []
        }
        # Get all points and years from all splits
        for s in [config.data.train, config.data.val, config.data.test]:
            if s:
                split_config['points'].extend(s.get_point_ids(config.data.points_lookup))
                split_config['years'].extend(s.get_years())
        split_config['points'] = sorted(list(set(split_config['points'])))
        split_config['years'] = sorted(list(set(split_config['years'])))
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'")
    
    if not split_config:
        raise ValueError(f"No configuration found for split: {split}")
    
    # Create dataset
    dataset = DayCentDatasetV2(
        weather_df=prepared_data['weather_df'],
        management_df=prepared_data['management_df'],
        output_df=prepared_data['output_df'],
        init_cond_path=config.data.init_cond_file,
        split_config=split_config,
        year_emb_dim=16
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Step 3: Load model
    print("\nStep 3: Loading model...")
    sample = dataset[0]
    seq_feat_dim = sample["sequence"].shape[1]
    init_dim = sample["init_cond"].shape[0]
    year_dim = sample["year_enc"].shape[0]
    
    print(f"  Input features: {seq_feat_dim}")
    print(f"  Init cond dim:  {init_dim}")
    print(f"  Year enc dim:   {year_dim}")
    
    model = DayCentModel(input_dim=seq_feat_dim, init_dim=init_dim, year_dim=year_dim)
    
    model_path = config.get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Model loaded from: {model_path}")
    print(f"  Device: {device}")
    
    # Step 4: Collect predictions
    print("\nStep 4: Collecting predictions...")
    predictions = collect_predictions(model, loader, device)
    
    # Step 5: Inverse transform
    print("\nStep 5: Inverse transforming to original scale...")
    scaler_path = config.get_scaler_path()
    predictions = inverse_transform_predictions(predictions, scaler_path)
    
    # Step 6: Calculate metrics
    print("\nStep 6: Calculating metrics...")
    metrics = calculate_metrics(predictions)
    print_metrics(metrics, split)
    
    # Step 7: Save predictions to CSV
    if save_csv:
        print("\nStep 7: Saving predictions...")
        csv_path = os.path.join(config.output_dir, f'{split}_predictions.csv')
        save_predictions_to_csv(predictions, csv_path)
    
    # Step 8: Generate sample plots
    print(f"\nStep 8: Generating sample plots...")
    plots_dir = os.path.join(config.plots_dir, split)
    generate_sample_plots(predictions, plots_dir, 
                         num_scenarios=num_samples, 
                         num_points_per_scenario=3)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")
    
    return {
        'metrics': metrics,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DayCent experiment')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment YAML config file')
    parser.add_argument('--split', type=str, default='test',
                      choices=['train', 'val', 'test', 'all'],
                      help='Which split to evaluate (default: test)')
    parser.add_argument('--num-samples', type=int, default=5,
                      help='Number of sample scenarios to plot (default: 5)')
    parser.add_argument('--no-csv', action='store_true',
                      help='Skip saving predictions to CSV')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = ExperimentConfig.from_yaml(args.config)
    
    # Run evaluation
    evaluate(config, split=args.split, num_samples=args.num_samples, 
            save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
