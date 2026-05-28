"""
Evaluation script for yearly December SOMSC state experiments.

Usage:
    python eval_yearly_somsc.py --config configs/yearly/experiment_state_v1.yaml --split test
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.yearly import SOC_STATE_COLS, create_yearly_dataset, prepare_yearly_data
from model import build_model
from utils import (
    compute_masked_metrics,
    get_device,
    move_batch_to_device,
    plot_timeseries,
)
from utils.config import ExperimentConfig
from utils.metrics import compute_grouped_masked_metrics, compute_regression_metrics


def _batch_value_at(batch, key, idx):
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def collect_state_predictions(model, loader, device):
    """Run teacher-forced yearly inference and collect raw-unit outputs."""
    model.eval()

    predictions = {
        "somsc_pred": [],
        "somsc_true": [],
        "somsc_mask": [],
        "somsc_delta_pred": [],
        "somsc_delta_true": [],
        "somsc_persistence_pred": [],
        "metadata": [],
        "eval_mode": "teacher_forced",
        "target_mode": getattr(model, "target_mode", "pool_deltas"),
        "prev_state_context": getattr(model, "prev_state_context", "target_pools"),
        "target_pool_cols": getattr(model, "target_pool_cols", SOC_STATE_COLS),
        "context_pool_cols": getattr(model, "context_pool_cols", []),
    }

    state_keys_initialized = False

    print("Running yearly inference (teacher_forced)...")
    for batch in tqdm(loader, desc="Inference"):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            outputs = model(batch)

        prev_somsc = batch["prev_somsc_state"]
        somsc_delta_pred = outputs.get("somsc_delta_pred")
        if somsc_delta_pred is None:
            somsc_delta_pred = outputs["somsc_pred"] - prev_somsc

        predictions["somsc_pred"].append(outputs["somsc_pred"].cpu().numpy())
        predictions["somsc_true"].append(batch["somsc"].cpu().numpy())
        predictions["somsc_mask"].append(batch["somsc_mask"].cpu().numpy())
        predictions["somsc_delta_pred"].append(somsc_delta_pred.cpu().numpy())
        predictions["somsc_delta_true"].append(
            (batch["somsc"] - prev_somsc).cpu().numpy()
        )
        predictions["somsc_persistence_pred"].append(prev_somsc.cpu().numpy())

        if "soc_state_pred" in outputs:
            if not state_keys_initialized:
                predictions["soc_state_pred"] = []
                predictions["soc_state_true"] = []
                predictions["soc_state_delta_pred"] = []
                predictions["soc_state_delta_true"] = []
                predictions["prev_soc_state_true"] = []
                state_keys_initialized = True
            predictions["soc_state_pred"].append(outputs["soc_state_pred"].cpu().numpy())
            predictions["soc_state_true"].append(
                batch["target_soc_state_raw"].cpu().numpy()
            )
            predictions["soc_state_delta_pred"].append(
                outputs["soc_state_delta_pred"].cpu().numpy()
            )
            predictions["soc_state_delta_true"].append(
                batch["soc_state_delta_raw"].cpu().numpy()
            )
            predictions["prev_soc_state_true"].append(
                batch["prev_soc_state_raw"].cpu().numpy()
            )

        batch_size = len(batch["pid"])
        for idx in range(batch_size):
            predictions["metadata"].append(
                {
                    "scenario_id": _batch_value_at(batch, "scenario_id", idx),
                    "pid": _batch_value_at(batch, "pid", idx),
                    "year": _batch_value_at(batch, "year", idx),
                }
            )

    for key in [
        "somsc_pred",
        "somsc_true",
        "somsc_mask",
        "somsc_delta_pred",
        "somsc_delta_true",
        "somsc_persistence_pred",
    ]:
        predictions[key] = np.concatenate(predictions[key], axis=0)

    for key in [
        "soc_state_pred",
        "soc_state_true",
        "soc_state_delta_pred",
        "soc_state_delta_true",
        "prev_soc_state_true",
    ]:
        if key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)

    return predictions


def _tensorize_single_sample(sample):
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def _normalize_state(state_raw: np.ndarray, state_input_scaler) -> np.ndarray:
    return state_input_scaler.transform(state_raw.reshape(1, -1))[0].astype(np.float32)


def _normalize_somsc(value: float, somsc_input_scaler) -> np.float32:
    return np.float32(
        somsc_input_scaler.transform(np.asarray([[value]], dtype=np.float32))[0, 0]
    )


def collect_state_rollout_predictions(
    model,
    dataset,
    device,
    state_input_scaler,
    somsc_input_scaler,
):
    """
    Roll out one continuous trajectory per scenario-point.

    The first sample for each trajectory uses the true previous pool state; later
    samples feed the model's predicted state forward.
    """
    model.eval()
    target_mode = getattr(model, "target_mode", "pool_deltas")
    prev_state_context = getattr(model, "prev_state_context", "target_pools")
    if target_mode == "somsc_delta" and prev_state_context != "somsc":
        raise ValueError(
            "Rollout for scalar SOMSC-delta models is only defined when "
            "model.prev_state_context='somsc'. Full-pool scalar models need "
            "teacher-forced true previous pools."
        )

    predictions = {
        "somsc_pred": [],
        "somsc_true": [],
        "somsc_mask": [],
        "somsc_delta_pred": [],
        "somsc_delta_true": [],
        "somsc_persistence_pred": [],
        "metadata": [],
        "eval_mode": "rollout",
        "target_mode": target_mode,
        "prev_state_context": prev_state_context,
        "target_pool_cols": getattr(model, "target_pool_cols", SOC_STATE_COLS),
        "context_pool_cols": getattr(model, "context_pool_cols", []),
    }
    if target_mode == "pool_deltas":
        predictions["soc_state_pred"] = []
        predictions["soc_state_true"] = []
        predictions["soc_state_delta_pred"] = []
        predictions["soc_state_delta_true"] = []
        predictions["prev_soc_state_true"] = []

    grouped_indices = {}
    for idx, (scenario_id, pid, year) in enumerate(dataset.samples):
        grouped_indices.setdefault((scenario_id, pid), []).append((int(year), idx))

    print("Running yearly state inference (rollout)...")
    for _, year_indices in tqdm(
        grouped_indices.items(),
        desc="Rollout trajectories",
    ):
        current_state_raw = None
        current_somsc = None
        for _, sample_idx in sorted(year_indices):
            sample = dataset[sample_idx]
            true_prev_state = sample["prev_soc_state_raw"].numpy().astype(np.float32)
            true_prev_somsc = sample["prev_somsc_state"].reshape(-1)

            if target_mode == "pool_deltas" and current_state_raw is None:
                input_prev_state_raw = true_prev_state
                input_prev_state_norm = sample["prev_soc_state_norm"].numpy().astype(
                    np.float32
                )
                input_prev_somsc = float(input_prev_state_raw[:3].sum())
                input_prev_somsc_norm = float(sample["prev_somsc_norm"].item())
            elif target_mode == "pool_deltas":
                input_prev_state_raw = current_state_raw.astype(np.float32)
                input_prev_state_norm = _normalize_state(
                    input_prev_state_raw,
                    state_input_scaler,
                )
                input_prev_somsc = float(input_prev_state_raw[:3].sum())
                input_prev_somsc_norm = float(
                    _normalize_somsc(input_prev_somsc, somsc_input_scaler)
                )
            elif current_somsc is None:
                input_prev_state_raw = true_prev_state
                input_prev_state_norm = sample["prev_soc_state_norm"].numpy().astype(
                    np.float32
                )
                input_prev_somsc = float(sample["prev_somsc_state"].item())
                input_prev_somsc_norm = float(sample["prev_somsc_norm"].item())
            else:
                input_prev_state_raw = true_prev_state
                input_prev_state_norm = sample["prev_soc_state_norm"].numpy().astype(
                    np.float32
                )
                input_prev_somsc = float(current_somsc)
                input_prev_somsc_norm = float(
                    _normalize_somsc(input_prev_somsc, somsc_input_scaler)
                )

            batch = _tensorize_single_sample(sample)
            batch["prev_soc_state_raw"] = torch.tensor(
                input_prev_state_raw,
                dtype=torch.float32,
            ).unsqueeze(0)
            batch["prev_soc_state_norm"] = torch.tensor(
                input_prev_state_norm,
                dtype=torch.float32,
            ).unsqueeze(0)
            batch["prev_somsc_state"] = torch.tensor(
                input_prev_somsc,
                dtype=torch.float32,
            ).unsqueeze(0)
            batch["prev_somsc_norm"] = torch.tensor(
                input_prev_somsc_norm,
                dtype=torch.float32,
            ).unsqueeze(0)
            batch = move_batch_to_device(batch, device)

            with torch.no_grad():
                outputs = model(batch)

            if target_mode == "pool_deltas":
                current_state_raw = outputs["soc_state_pred"].squeeze(0).cpu().numpy()
            else:
                current_somsc = float(outputs["somsc_pred"].squeeze(0).cpu().item())

            somsc_delta_pred = outputs.get("somsc_delta_pred")
            if somsc_delta_pred is None:
                somsc_delta_pred = outputs["somsc_pred"] - batch["prev_somsc_state"]

            predictions["somsc_pred"].append(outputs["somsc_pred"].cpu().numpy())
            predictions["somsc_true"].append(sample["somsc"].reshape(1).numpy())
            predictions["somsc_mask"].append(sample["somsc_mask"].reshape(1).numpy())
            predictions["somsc_delta_pred"].append(somsc_delta_pred.cpu().numpy())
            predictions["somsc_delta_true"].append(
                (sample["somsc"].reshape(1) - true_prev_somsc).numpy()
            )
            predictions["somsc_persistence_pred"].append(
                np.asarray([input_prev_somsc], dtype=np.float32)
            )
            if target_mode == "pool_deltas":
                predictions["soc_state_pred"].append(
                    outputs["soc_state_pred"].cpu().numpy()
                )
                predictions["soc_state_true"].append(
                    sample["target_soc_state_raw"].unsqueeze(0).numpy()
                )
                predictions["soc_state_delta_pred"].append(
                    outputs["soc_state_delta_pred"].cpu().numpy()
                )
                predictions["soc_state_delta_true"].append(
                    sample["soc_state_delta_raw"].unsqueeze(0).numpy()
                )
                predictions["prev_soc_state_true"].append(
                    sample["prev_soc_state_raw"].unsqueeze(0).numpy()
                )
            predictions["metadata"].append(
                {
                    "scenario_id": sample["scenario_id"],
                    "pid": sample["pid"],
                    "year": sample["year"],
                }
            )

    for key in [
        "somsc_pred",
        "somsc_true",
        "somsc_mask",
        "somsc_delta_pred",
        "somsc_delta_true",
        "somsc_persistence_pred",
    ]:
        predictions[key] = np.concatenate(predictions[key], axis=0)

    for key in [
        "soc_state_pred",
        "soc_state_true",
        "soc_state_delta_pred",
        "soc_state_delta_true",
        "prev_soc_state_true",
    ]:
        if key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)

    return predictions


def calculate_state_metrics(predictions):
    """Calculate raw-unit metrics for state-model predictions."""
    somsc_level = compute_masked_metrics(
        predictions["somsc_true"],
        predictions["somsc_pred"],
        predictions["somsc_mask"],
    )
    somsc_delta = compute_masked_metrics(
        predictions["somsc_delta_true"],
        predictions["somsc_delta_pred"],
        predictions["somsc_mask"],
    )
    persistence = compute_masked_metrics(
        predictions["somsc_true"],
        predictions["somsc_persistence_pred"],
        predictions["somsc_mask"],
    )

    pool_delta = {}
    if "soc_state_delta_true" in predictions and "soc_state_delta_pred" in predictions:
        mask = predictions["somsc_mask"].reshape(-1).astype(bool)
        target_pool_cols = predictions.get("target_pool_cols", SOC_STATE_COLS)
        for name in target_pool_cols:
            idx = SOC_STATE_COLS.index(name)
            pool_delta[name] = compute_regression_metrics(
                predictions["soc_state_delta_true"][mask, idx],
                predictions["soc_state_delta_pred"][mask, idx],
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
        "somsc_level": somsc_level,
        "somsc_delta": somsc_delta,
        "persistence": persistence,
        "pool_delta": pool_delta,
        "scenario_point_avg": scenario_point_avg,
    }


def _print_metric_block(title, metrics):
    print(title)
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  MSE:     {metrics['mse']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")
    print(f"  R2:      {metrics['r2']:.4f}")


def print_state_metrics(metrics, split_name, eval_mode):
    """Print raw-unit state-model evaluation metrics."""
    print(f"\n{'=' * 80}")
    print(
        "YEARLY SOC STATE EVALUATION METRICS - "
        f"{split_name.upper()} SET ({eval_mode})"
    )
    print(f"{'=' * 80}\n")

    _print_metric_block("SOMSC level", metrics["somsc_level"])
    print()
    _print_metric_block("SOMSC annual delta", metrics["somsc_delta"])
    print()
    _print_metric_block("Persistence baseline (previous true SOMSC)", metrics["persistence"])

    if metrics["pool_delta"]:
        print("\nPool annual deltas")
        for name, pool_metrics in metrics["pool_delta"].items():
            print(
                f"  {name}: RMSE={pool_metrics['rmse']:.4f}, "
                f"MAE={pool_metrics['mae']:.4f}, R2={pool_metrics['r2']:.4f}"
            )

    scenario_point_avg = metrics.get("scenario_point_avg")
    if scenario_point_avg is not None:
        print("\nAverage of per-(scenario_id, pid) SOMSC level metrics")
        print(f"  Scenario-point groups: {scenario_point_avg['n_groups']}")
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


def plot_soc_state_stack(scenario_id, pid, sample_indices, predictions, plots_dir):
    """Plot stacked predicted/true SOC pool trajectories for one scenario-point."""
    if "soc_state_pred" not in predictions:
        return

    import matplotlib.pyplot as plt

    years = np.array(
        [int(predictions["metadata"][idx]["year"]) for idx in sample_indices],
        dtype=int,
    )
    sort_idx = np.argsort(years)
    sorted_indices = np.asarray(sample_indices, dtype=int)[sort_idx]
    years = years[sort_idx]

    true_state = predictions["soc_state_true"][sorted_indices]
    pred_state = predictions["soc_state_pred"][sorted_indices]
    mask = predictions["somsc_mask"][sorted_indices].astype(bool).reshape(-1)
    years = years[mask]
    true_state = true_state[mask]
    pred_state = pred_state[mask]

    scenario_dir = os.path.join(plots_dir, str(scenario_id))
    os.makedirs(scenario_dir, exist_ok=True)
    save_path = os.path.join(scenario_dir, f"point_{pid}_soc_pools.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = ["#3B7A57", "#C85A54", "#4E79A7", "#A06CD5"]
    axes[0].stackplot(
        years,
        true_state.T,
        labels=SOC_STATE_COLS,
        colors=colors,
        alpha=0.75,
    )
    axes[0].set_title(f"Scenario {scenario_id} | Point {pid} | True SOC pools")
    axes[0].set_ylabel("g C/m²")
    axes[0].grid(True, alpha=0.25)

    axes[1].stackplot(
        years,
        pred_state.T,
        labels=SOC_STATE_COLS,
        colors=colors,
        alpha=0.75,
    )
    axes[1].set_title("Predicted SOC pools")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("g C/m²")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
            plot_soc_state_stack(scenario_id, pid, sample_indices, predictions, plots_dir)
            total_plots += 1

    print(f"\n{'=' * 80}")
    print(f"Generated {total_plots} yearly plots in: {plots_dir}")
    print(f"{'=' * 80}\n")


def save_state_predictions_to_csv(predictions, output_path):
    """Save state-model predictions and pool diagnostics to CSV."""
    print(f"\nSaving yearly state predictions to CSV: {output_path}")

    rows = []
    for idx, meta in enumerate(predictions["metadata"]):
        row = {
            "scenario_id": meta["scenario_id"],
            "point_id": meta["pid"],
            "year": meta["year"],
            "eval_mode": predictions.get("eval_mode", "teacher_forced"),
            "target_mode": predictions.get("target_mode", "pool_deltas"),
            "prev_state_context": predictions.get("prev_state_context", "target_pools"),
            "target_pool_cols": ",".join(
                predictions.get("target_pool_cols", SOC_STATE_COLS)
            ),
            "context_pool_cols": ",".join(predictions.get("context_pool_cols", [])),
            "somsc_pred": predictions["somsc_pred"][idx],
            "somsc_true": predictions["somsc_true"][idx],
            "somsc_delta_pred": predictions["somsc_delta_pred"][idx],
            "somsc_delta_true": predictions["somsc_delta_true"][idx],
            "somsc_persistence_pred": predictions["somsc_persistence_pred"][idx],
            "somsc_mask": predictions["somsc_mask"][idx],
        }
        if "soc_state_pred" in predictions:
            for pool_idx, pool_name in enumerate(SOC_STATE_COLS):
                row[f"{pool_name}_pred"] = predictions["soc_state_pred"][idx, pool_idx]
                row[f"{pool_name}_true"] = predictions["soc_state_true"][idx, pool_idx]
                row[f"{pool_name}_delta_pred"] = predictions["soc_state_delta_pred"][
                    idx, pool_idx
                ]
                row[f"{pool_name}_delta_true"] = predictions["soc_state_delta_true"][
                    idx, pool_idx
                ]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} yearly state predictions")


def evaluate_experiment(
    config,
    split="test",
    num_samples=5,
    save_csv=True,
    prepared_data=None,
    eval_mode="teacher_forced",
):
    """Main evaluation function for yearly SOMSC experiments."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING YEARLY EXPERIMENT: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"Split: {split}")
    print(f"Evaluation mode: {eval_mode}")
    print(f"{'=' * 80}\n")

    if config.model.model_type != "yearly_somsc_state":
        raise ValueError(
            "The yearly evaluation pipeline now only supports "
            "model.model_type='yearly_somsc_state'."
        )

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
                split_config["points"].extend(
                    split_obj.get_point_ids(config.data.points_lookup)
                )
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

    modes = ["teacher_forced", "rollout"] if eval_mode == "both" else [eval_mode]
    invalid_modes = sorted(set(modes) - {"teacher_forced", "rollout"})
    if invalid_modes:
        raise ValueError(f"Invalid state evaluation mode(s): {invalid_modes}")

    results = {}
    for mode in modes:
        print(f"\nStep 4: Collecting yearly state predictions ({mode})...")
        if mode == "teacher_forced":
            predictions = collect_state_predictions(model, loader, device)
        else:
            if "state_input_scaler" not in prepared_data or "somsc_input_scaler" not in prepared_data:
                raise ValueError(
                    "Rollout evaluation requires prepared_data['state_input_scaler'] "
                    "and prepared_data['somsc_input_scaler']."
                )
            predictions = collect_state_rollout_predictions(
                model,
                dataset,
                device,
                prepared_data["state_input_scaler"],
                prepared_data["somsc_input_scaler"],
            )

        print("\nStep 5: Calculating state metrics...")
        metrics = calculate_state_metrics(predictions)
        print_state_metrics(metrics, split, mode)

        if save_csv:
            print("\nStep 6: Saving yearly state predictions...")
            csv_path = os.path.join(
                config.output_dir,
                f"{split}_{mode}_yearly_state_predictions.csv",
            )
            save_state_predictions_to_csv(predictions, csv_path)

        print("\nStep 7: Generating yearly state sample plots...")
        plots_dir = os.path.join(config.plots_dir, f"yearly_{split}_{mode}")
        generate_sample_plots(
            predictions,
            plots_dir,
            num_scenarios=num_samples,
            num_points_per_scenario=3,
        )
        results[mode] = {"metrics": metrics, "predictions": predictions}

    print(f"\n{'=' * 80}")
    print("YEARLY EVALUATION COMPLETE!")
    print(f"{'=' * 80}\n")

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate yearly December SOMSC state experiment"
    )
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
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="teacher_forced",
        choices=["teacher_forced", "rollout", "both"],
        help="State-model evaluation mode (default: teacher_forced)",
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
        eval_mode=args.eval_mode,
    )


if __name__ == "__main__":
    main()
