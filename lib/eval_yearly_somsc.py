"""
Evaluation script for yearly December SOMSC experiments.

Usage:
    python eval_yearly_somsc.py --config configs/yearly/december_somsc_example.yaml --split test
"""
import argparse
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.yearly import create_yearly_dataset, prepare_yearly_data
from model import build_model
from utils import (
    compute_masked_metrics,
    get_device,
    move_batch_to_device,
    plot_timeseries,
)
from utils.config import ExperimentConfig
from utils.metrics import compute_grouped_masked_metrics, compute_regression_metrics


def collect_predictions(model, loader, device):
    """Run inference on a yearly data loader and collect all predictions."""
    model.eval()

    all_somsc_preds = []
    all_somsc_trues = []
    all_somsc_masks = []
    all_metadata = []

    print("Running yearly inference...")
    for batch in tqdm(loader, desc="Inference"):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            outputs = model(batch)

        all_somsc_preds.append(outputs["somsc_pred"].cpu().numpy())
        all_somsc_trues.append(batch["somsc"].cpu().numpy())
        all_somsc_masks.append(batch["somsc_mask"].cpu().numpy())

        batch_size = len(batch["pid"])
        for idx in range(batch_size):
            all_metadata.append(
                {
                    "scenario_id": batch["scenario_id"][idx]
                    if isinstance(batch["scenario_id"], list)
                    else batch["scenario_id"],
                    "pid": batch["pid"][idx] if isinstance(batch["pid"], list) else batch["pid"],
                    "year": batch["year"][idx] if isinstance(batch["year"], list) else batch["year"],
                }
            )

    return {
        "somsc_pred": np.concatenate(all_somsc_preds, axis=0),
        "somsc_true": np.concatenate(all_somsc_trues, axis=0),
        "somsc_mask": np.concatenate(all_somsc_masks, axis=0),
        "metadata": all_metadata,
    }


def inverse_transform_predictions(predictions, scaler_path):
    """Inverse-transform yearly SOMSC predictions back to the original scale."""
    scaler_y = joblib.load(scaler_path)
    somsc_mean = scaler_y.mean_[0]
    somsc_scale = scaler_y.scale_[0]

    print("\nInverse transforming yearly SOMSC predictions...")
    print(f"  SOMSC - mean: {somsc_mean:.4f}, scale: {somsc_scale:.4f}")

    return {
        "somsc_pred": predictions["somsc_pred"] * somsc_scale + somsc_mean,
        "somsc_true": predictions["somsc_true"] * somsc_scale + somsc_mean,
        "somsc_mask": predictions["somsc_mask"],
        "metadata": predictions["metadata"],
    }


def calculate_metrics(predictions):
    """Calculate pooled and scenario-point averaged December SOMSC metrics."""
    pooled_metrics = compute_masked_metrics(
        predictions["somsc_true"],
        predictions["somsc_pred"],
        predictions["somsc_mask"],
    )
    scenario_point_avg = compute_grouped_masked_metrics(
        predictions["somsc_true"],
        predictions["somsc_pred"],
        predictions["somsc_mask"],
        group_keys=[
            (meta["scenario_id"], meta["pid"])
            for meta in predictions["metadata"]
        ],
    )
    return {
        **pooled_metrics,
        "scenario_point_avg": scenario_point_avg,
    }


def print_metrics(metrics, split_name):
    """Print evaluation metrics in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"YEARLY EVALUATION METRICS - {split_name.upper()} SET")
    print(f"{'=' * 80}\n")
    print("Pooled over all sample-years")
    print(f"  December SOMSC samples: {metrics['n_samples']}")
    print(f"  MSE:   {metrics['mse']:.4f}")
    print(f"  RMSE:  {metrics['rmse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  R2:    {metrics['r2']:.4f}")

    scenario_point_avg = metrics.get("scenario_point_avg")
    if scenario_point_avg is not None:
        print("\nAverage of per-(scenario_id, pid) metrics")
        print(f"  Scenario-point groups: {scenario_point_avg['n_groups']}")
        print(f"  Total valid samples:   {scenario_point_avg['n_samples_total']}")
        print(
            "  Mean samples/group:   "
            f"{scenario_point_avg['mean_samples_per_group']:.2f}"
        )
        print(f"  MSE:   {scenario_point_avg['mse']:.4f}")
        print(f"  RMSE:  {scenario_point_avg['rmse']:.4f}")
        print(f"  MAE:   {scenario_point_avg['mae']:.4f}")
        print(f"  R2:    {scenario_point_avg['r2']:.4f}")
    print(f"\n{'=' * 80}\n")


def create_metadata_index(predictions):
    """Create an index mapping (scenario_id, pid) to sample indices."""
    index = {}
    for idx, meta in enumerate(predictions["metadata"]):
        key = (meta["scenario_id"], meta["pid"])
        if key not in index:
            index[key] = []
        index[key].append(idx)
    return index


def plot_predictions(scenario_id, pid, sample_indices, predictions, plots_dir):
    """Plot yearly December SOMSC trajectories for a scenario-point pair."""
    years = np.array(
        [int(predictions["metadata"][idx]["year"]) for idx in sample_indices],
        dtype=int,
    )
    sort_idx = np.argsort(years)
    years = years[sort_idx]

    somsc_pred = predictions["somsc_pred"][sample_indices][sort_idx]
    somsc_true = predictions["somsc_true"][sample_indices][sort_idx]
    somsc_mask = predictions["somsc_mask"][sample_indices][sort_idx]

    valid = somsc_mask.astype(bool).flatten()
    valid_years = years[valid]
    valid_pred = somsc_pred.flatten()[valid]
    valid_true = somsc_true.flatten()[valid]

    metrics = compute_regression_metrics(valid_true, valid_pred)
    scenario_dir = os.path.join(plots_dir, str(scenario_id))
    os.makedirs(scenario_dir, exist_ok=True)
    save_path = os.path.join(scenario_dir, f"point_{pid}.png")

    plot_timeseries(
        true=valid_true,
        pred=valid_pred,
        title=f"Scenario {scenario_id} | Point {pid}",
        xlabel="Year",
        ylabel="December SOMSC (g C/m²)",
        metrics=metrics,
        x=valid_years,
        save_path=save_path,
    )
    print(f"  Saved plot: {save_path}")


def generate_sample_plots(predictions, plots_dir, num_scenarios=5, num_points_per_scenario=3):
    """Generate yearly sample plots for a subset of scenarios and points."""
    print(f"\n{'=' * 80}")
    print("GENERATING YEARLY SAMPLE PLOTS")
    print(f"{'=' * 80}\n")

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

    print(f"\n{'=' * 80}")
    print(f"Generated {total_plots} yearly plots in: {plots_dir}")
    print(f"{'=' * 80}\n")


def save_predictions_to_csv(predictions, output_path):
    """Save yearly predictions to CSV for further analysis."""
    print(f"\nSaving yearly predictions to CSV: {output_path}")

    rows = []
    for idx, meta in enumerate(predictions["metadata"]):
        rows.append(
            {
                "scenario_id": meta["scenario_id"],
                "point_id": meta["pid"],
                "year": meta["year"],
                "somsc_pred_dec": predictions["somsc_pred"][idx],
                "somsc_true_dec": predictions["somsc_true"][idx],
                "somsc_mask_dec": predictions["somsc_mask"][idx],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} yearly predictions")


def evaluate_experiment(config, split="test", num_samples=5, save_csv=True, prepared_data=None):
    """Main evaluation function for yearly SOMSC experiments."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING YEARLY EXPERIMENT: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"Split: {split}")
    print(f"{'=' * 80}\n")

    print("Step 1: Preparing yearly data...")
    if prepared_data is None:
        prepared_data = prepare_yearly_data(config)
    else:
        print("  (using pre-loaded yearly data, skipping disk I/O)")

    print(f"\nStep 2: Loading {split} yearly dataset...")
    if split == "train":
        split_config = config.data.get_train_config()
    elif split == "val":
        split_config = config.data.get_val_config()
    elif split == "test":
        split_config = config.data.get_test_config()
    elif split == "all":
        split_config = {
            "scenarios": config.data.get_all_scenario_ids(),
            "points": [],
            "years": [],
        }
        for split_obj in [config.data.train, config.data.val, config.data.test]:
            if split_obj:
                split_config["points"].extend(split_obj.get_point_ids(config.data.points_lookup))
                split_config["years"].extend(split_obj.get_years())
        split_config["points"] = sorted(list(set(split_config["points"])))
        split_config["years"] = sorted(list(set(split_config["years"])))
    else:
        raise ValueError(f"Invalid split: {split}")

    if not split_config:
        raise ValueError(f"No configuration found for split: {split}")

    dataset = create_yearly_dataset(
        prepared_data=prepared_data,
        init_cond_path=config.data.init_cond_file,
        split_config=split_config,
        year_emb_dim=16,
    )
    if len(dataset) == 0:
        raise ValueError(f"Yearly dataset for split '{split}' is empty.")

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    print(f"  Dataset size: {len(dataset)}")

    print("\nStep 3: Loading yearly model...")
    sample = dataset[0]
    model = build_model(config.model, sample, verbose=True)

    model_path = config.get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    device = get_device(config.training.device)
    model.to(device)
    print(f"  Model loaded from: {model_path}")
    print(f"  Device: {device}")

    print("\nStep 4: Collecting yearly predictions...")
    predictions = collect_predictions(model, loader, device)

    print("\nStep 5: Inverse transforming to original scale...")
    # predictions = inverse_transform_predictions(predictions, config.get_scaler_path())

    print("\nStep 6: Calculating metrics...")
    metrics = calculate_metrics(predictions)
    print_metrics(metrics, split)

    if save_csv:
        print("\nStep 7: Saving yearly predictions...")
        csv_path = os.path.join(config.output_dir, f"{split}_yearly_predictions.csv")
        save_predictions_to_csv(predictions, csv_path)

    print("\nStep 8: Generating yearly sample plots...")
    plots_dir = os.path.join(config.plots_dir, f"yearly_{split}")
    generate_sample_plots(
        predictions,
        plots_dir,
        num_scenarios=num_samples,
        num_points_per_scenario=3,
    )

    print(f"\n{'=' * 80}")
    print("YEARLY EVALUATION COMPLETE!")
    print(f"{'=' * 80}\n")

    return {"metrics": metrics, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Evaluate yearly December SOMSC experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample scenarios to plot (default: 5)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save predictions to CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(args.config)
    evaluate_experiment(
        config,
        split=args.split,
        num_samples=args.num_samples,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
