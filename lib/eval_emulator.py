"""
Evaluation script for DayCent emulator experiments.

Usage:
    python eval_emulator.py --config configs/emulator/experiment_1.yaml --split test
    python eval_emulator.py --config configs/emulator/experiment_1.yaml --split all --num-samples 5
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

from utils.config import ExperimentConfig
from data.preprocessing import prepare_data_for_datasetv2
from data.loader import create_dataset
from model import build_model
from utils import (
    get_device,
    move_batch_to_device,
    compute_emulator_metrics,
    plot_dual_timeseries,
)


def collect_predictions(model, loader, device):
    """Run inference on a data loader and collect all predictions."""
    model.eval()
    
    all_yield_preds = []
    all_yield_trues = []
    all_yield_masks = []
    all_somsc_preds = []
    all_somsc_trues = []
    all_somsc_masks = []
    all_metadata = []
    
    print("Running inference...")
    for batch in tqdm(loader, desc="Inference"):
        batch = move_batch_to_device(batch, device)
        
        with torch.no_grad():
            outputs = model(batch)
        
        all_yield_preds.append(outputs['yield_pred'].cpu().numpy())
        all_yield_trues.append(batch['yield'].cpu().numpy())
        all_yield_masks.append(batch['yield_mask'].cpu().numpy())
        all_somsc_preds.append(outputs['somsc_pred'].cpu().numpy())
        all_somsc_trues.append(batch['somsc'].cpu().numpy())
        all_somsc_masks.append(batch['somsc_mask'].cpu().numpy())
        
        batch_size = len(batch['pid'])
        for i in range(batch_size):
            all_metadata.append({
                'scenario_id': batch['scenario_id'][i] if isinstance(batch['scenario_id'], list) else batch['scenario_id'],
                'pid': batch['pid'][i] if isinstance(batch['pid'], list) else batch['pid'],
                'year': batch['year'][i] if isinstance(batch['year'], list) else batch['year']
            })
    
    predictions = {
        'yield_pred': np.concatenate(all_yield_preds, axis=0),
        'yield_true': np.concatenate(all_yield_trues, axis=0),
        'yield_mask': np.concatenate(all_yield_masks, axis=0),
        'somsc_pred': np.concatenate(all_somsc_preds, axis=0),
        'somsc_true': np.concatenate(all_somsc_trues, axis=0),
        'somsc_mask': np.concatenate(all_somsc_masks, axis=0),
        'metadata': all_metadata
    }
    
    if predictions['somsc_pred'].ndim == 3 and predictions['somsc_pred'].shape[2] == 1:
        predictions['somsc_pred'] = predictions['somsc_pred'].squeeze(-1)
    if predictions['somsc_true'].ndim == 3 and predictions['somsc_true'].shape[2] == 1:
        predictions['somsc_true'] = predictions['somsc_true'].squeeze(-1)
    
    return predictions


def inverse_transform_predictions(predictions, scaler_path):
    """Apply inverse transformation to normalized predictions."""
    scaler_Y = joblib.load(scaler_path)
    
    cgrain_mean = scaler_Y.mean_[1]
    cgrain_scale = scaler_Y.scale_[1]
    somsc_mean = scaler_Y.mean_[0]
    somsc_scale = scaler_Y.scale_[0]
    
    print("\nInverse transforming predictions...")
    print(f"  CGRAIN - mean: {cgrain_mean:.4f}, scale: {cgrain_scale:.4f}")
    print(f"  SOMSC - mean: {somsc_mean:.4f}, scale: {somsc_scale:.4f}")
    
    # Inverse transform without applying mask - mask is for filtering only
    yield_pred = predictions['yield_pred'] * cgrain_scale + cgrain_mean
    yield_true = predictions['yield_true'] * cgrain_scale + cgrain_mean
    
    somsc_pred = predictions['somsc_pred'] * somsc_scale + somsc_mean
    somsc_true = predictions['somsc_true'] * somsc_scale + somsc_mean
    
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
    """Calculate evaluation metrics — delegates to utils.metrics."""
    return compute_emulator_metrics(predictions)


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
    print(f"  R2:       {metrics['yield']['r2']:.4f}")
    
    print("\nSOMSC PREDICTIONS:")
    print(f"  Samples:  {metrics['somsc']['n_samples']}")
    print(f"  MSE:      {metrics['somsc']['mse']:.4f}")
    print(f"  RMSE:     {metrics['somsc']['rmse']:.4f}")
    print(f"  MAE:      {metrics['somsc']['mae']:.4f}")
    print(f"  R2:       {metrics['somsc']['r2']:.4f}")
    
    print(f"\n{'='*80}\n")


def create_metadata_index(predictions):
    """Create an index mapping (scenario_id, pid) to sample indices."""
    index = {}
    for i, meta in enumerate(predictions['metadata']):
        key = (meta['scenario_id'], meta['pid'])
        if key not in index:
            index[key] = []
        index[key].append(i)
    return index


def plot_predictions(scenario_id, pid, sample_indices, predictions, plots_dir):
    """Plot yield and SOMSC predictions for a scenario-point pair via utils.plotting."""
    yield_pred = predictions['yield_pred'][sample_indices]
    yield_true = predictions['yield_true'][sample_indices]
    yield_mask = predictions['yield_mask'][sample_indices]

    somsc_pred = predictions['somsc_pred'][sample_indices]
    somsc_true = predictions['somsc_true'][sample_indices]
    somsc_mask = predictions['somsc_mask'][sample_indices]

    # Filter by masks
    ym = yield_mask.astype(bool).flatten()
    sm = somsc_mask.astype(bool).flatten()
    valid_yield_pred  = yield_pred.flatten()[ym]
    valid_yield_true  = yield_true.flatten()[ym]
    somsc_pred_clean  = somsc_pred.flatten()[sm]
    somsc_true_clean  = somsc_true.flatten()[sm]

    from utils.metrics import compute_regression_metrics
    y_m = compute_regression_metrics(valid_yield_true, valid_yield_pred)
    s_m = compute_regression_metrics(somsc_true_clean, somsc_pred_clean)

    scenario_dir = os.path.join(plots_dir, str(scenario_id))
    os.makedirs(scenario_dir, exist_ok=True)
    save_path = os.path.join(scenario_dir, f"point_{pid}.png")

    plot_dual_timeseries(
        series=[
            (valid_yield_true,  valid_yield_pred,  "Yield"),
            (somsc_true_clean,  somsc_pred_clean,  "SOMSC"),
        ],
        titles=[
            f"Yield  MSE={y_m['mse']:.2f}  R²={y_m['r2']:.3f}",
            f"SOMSC  MSE={s_m['mse']:.2f}  R²={s_m['r2']:.3f}",
        ],
        xlabels=["Year Index", "Month Index"],
        ylabels=["Yield (kg/ha)", "SOMSC (g C/m²)"],
        suptitle=f"Scenario {scenario_id}  ·  Point {pid}",
        save_path=save_path,
    )
    print(f"  Saved plot: {save_path}")



def generate_sample_plots(predictions, plots_dir, num_scenarios=5, num_points_per_scenario=3):
    """Generate sample plots for a subset of scenarios and points."""
    print(f"\n{'='*80}")
    print("GENERATING SAMPLE PLOTS")
    print(f"{'='*80}\n")
    
    index = create_metadata_index(predictions)
    all_keys = list(index.keys())
    
    scenario_groups = {}
    for scenario_id, pid in all_keys:
        if scenario_id not in scenario_groups:
            scenario_groups[scenario_id] = []
        scenario_groups[scenario_id].append(pid)
    
    all_scenarios = list(scenario_groups.keys())
    if len(all_scenarios) > num_scenarios:
        sampled_scenarios = random.sample(all_scenarios, num_scenarios)
    else:
        sampled_scenarios = all_scenarios
    
    print(f"Plotting {len(sampled_scenarios)} scenarios...")
    
    total_plots = 0
    for scenario_id in sampled_scenarios:
        pids = scenario_groups[scenario_id]
        
        if len(pids) > num_points_per_scenario:
            sampled_pids = random.sample(pids, num_points_per_scenario)
        else:
            sampled_pids = pids
        
        print(f"\nScenario {scenario_id}:")
        for pid in sampled_pids:
            sample_indices = index[(scenario_id, pid)]
            plot_predictions(scenario_id, pid, sample_indices, predictions, plots_dir)
            total_plots += 1
    
    print(f"\n{'='*80}")
    print(f"Generated {total_plots} plots in: {plots_dir}")
    print(f"{'='*80}\n")


def save_predictions_to_csv(predictions, output_path):
    """Save predictions to CSV for further analysis."""
    print(f"\nSaving predictions to CSV: {output_path}")
    
    rows = []
    for i, meta in enumerate(predictions['metadata']):
        row = {
            'scenario_id': meta['scenario_id'],
            'point_id': meta['pid'],
            'year': meta['year'],
            'yield_pred': predictions['yield_pred'][i],
            'yield_true': predictions['yield_true'][i],
            'yield_mask': predictions['yield_mask'][i],
        }
        
        for month in range(12):
            row[f'somsc_pred_m{month+1}'] = predictions['somsc_pred'][i, month]
            row[f'somsc_true_m{month+1}'] = predictions['somsc_true'][i, month]
            row[f'somsc_mask_m{month+1}'] = predictions['somsc_mask'][i, month]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} predictions")


def evaluate_experiment(config, split='test', num_samples=5, save_csv=True,
                        prepared_data=None):
    """Main evaluation function.

    Args:
        config:        ExperimentConfig instance.
        split:         Which split to evaluate ('train', 'val', 'test', 'all').
        num_samples:   Number of sample plots to generate.
        save_csv:      Whether to save predictions to CSV.
        prepared_data: Optional pre-loaded & normalized data dict (output of
            prepare_data_for_datasetv2 / normalize_raw_data).  When provided
            the expensive data-loading step is skipped entirely.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING EXPERIMENT: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")

    print("Step 1: Preparing data...")
    if prepared_data is None:
        prepared_data = prepare_data_for_datasetv2(config, test_only=(split == 'test'))
    else:
        print("  (using pre-loaded data, skipping disk I/O)")
    
    print(f"\nStep 2: Loading {split} dataset...")
    
    if split == 'train':
        split_config = config.data.get_train_config()
    elif split == 'val':
        split_config = config.data.get_val_config()
    elif split == 'test':
        split_config = config.data.get_test_config()
    elif split == 'all':
        split_config = {
            'scenarios': config.data.get_all_scenario_ids(),
            'points': [],
            'years': []
        }
        for s in [config.data.train, config.data.val, config.data.test]:
            if s:
                split_config['points'].extend(s.get_point_ids(config.data.points_lookup))
                split_config['years'].extend(s.get_years())
        split_config['points'] = sorted(list(set(split_config['points'])))
        split_config['years'] = sorted(list(set(split_config['years'])))
    else:
        raise ValueError(f"Invalid split: {split}")
    
    if not split_config:
        raise ValueError(f"No configuration found for split: {split}")
    
    dataset = create_dataset(
        prepared_data=prepared_data,
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
    
    print("\nStep 3: Loading model...")
    sample = dataset[0]
    model = build_model(config.model, sample, verbose=True)
    
    model_path = config.get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = get_device(config.training.device)
    model.to(device)
    print(f"  Model loaded from: {model_path}")
    print(f"  Device: {device}")
    
    print("\nStep 4: Collecting predictions...")
    predictions = collect_predictions(model, loader, device)
    
    print("\nStep 5: Inverse transforming to original scale...")
    scaler_path = config.get_scaler_path()
    predictions = inverse_transform_predictions(predictions, scaler_path)
    
    print("\nStep 6: Calculating metrics...")
    metrics = calculate_metrics(predictions)
    print_metrics(metrics, split)
    
    if save_csv:
        print("\nStep 7: Saving predictions...")
        csv_path = os.path.join(config.output_dir, f'{split}_predictions.csv')
        save_predictions_to_csv(predictions, csv_path)
    
    print(f"\nStep 8: Generating sample plots...")
    plots_dir = os.path.join(config.plots_dir, split)
    generate_sample_plots(predictions, plots_dir, 
                         num_scenarios=num_samples, 
                         num_points_per_scenario=3)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")
    
    return {'metrics': metrics, 'predictions': predictions}


def main():
    parser = argparse.ArgumentParser(description='Evaluate DayCent experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment YAML config file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of sample scenarios to plot (default: 5)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save predictions to CSV')

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = ExperimentConfig.from_yaml(args.config)
    
    evaluate_experiment(config, split=args.split, num_samples=args.num_samples, 
                        save_csv=args.save_csv)


if __name__ == "__main__":
    main()
