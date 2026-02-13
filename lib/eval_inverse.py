"""
Evaluation script for DayCent inverse model.

This script:
1. Loads the trained inverse model
2. Encodes all test samples into latent codes
3. Evaluates reconstruction quality (MSE per channel)
4. Evaluates static site-condition prediction (MSE, R2 per feature)
5. Visualises the latent space (t-SNE / PCA coloured by point)
6. Optionally reconstructs sample sequences for visual inspection

Usage:
    python eval_inverse.py --config configs/inverse_1.yaml
    python eval_inverse.py --config configs/inverse_1.yaml --num-vis 10
"""
import argparse
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml

# ---- project imports ----
from data.inverse import DayCentInverseDataset, prepare_inverse_data
from model.inverse import InverseModel
from utils.training import setup_reproducibility, get_device
from utils.inverse_config import InverseExperimentConfig


# ======================================================================
# Collect predictions
# ======================================================================
@torch.no_grad()
def encode_dataset(model, dataset, device, batch_size=256):
    """
    Encode every sample in the dataset. Returns arrays of:
        codes           (N, code_dim)
        static_preds    (N, S)
        static_targets  (N, S)
        reconstructions (N, T, C)   -- only if store_recon=True
        metadata        list of dicts {pid, year, scenario_id}
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_codes = []
    all_static_preds = []
    all_static_targets = []
    all_recons = []
    all_inputs = []
    all_meta = []

    for batch in loader:
        seq = batch["sequence"].to(device)
        out = model({"sequence": seq})

        all_codes.append(out["code"].cpu().numpy())
        all_static_preds.append(out["static_pred"].cpu().numpy())
        all_static_targets.append(batch["static_target"].numpy())
        all_recons.append(out["reconstruction"].cpu().numpy())
        all_inputs.append(batch["sequence"].numpy())

        bs = seq.shape[0]
        for i in range(bs):
            all_meta.append({
                "pid": batch["pid"][i],
                "year": batch["year"][i],
                "scenario_id": batch["scenario_id"][i],
            })

    return {
        "codes": np.concatenate(all_codes),
        "static_preds": np.concatenate(all_static_preds),
        "static_targets": np.concatenate(all_static_targets),
        "reconstructions": np.concatenate(all_recons),
        "inputs": np.concatenate(all_inputs),
        "metadata": all_meta,
    }


# ======================================================================
# Metrics
# ======================================================================
def compute_reconstruction_metrics(inputs, recons, feature_names=None):
    """Per-channel MSE between inputs and reconstructions."""
    # inputs, recons: (N, T, C)
    mse_per_channel = np.mean((inputs - recons) ** 2, axis=(0, 1))  # (C,)
    overall_mse = mse_per_channel.mean()

    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION METRICS  (overall MSE: {overall_mse:.6f})")
    print(f"{'='*60}")
    for c, mse in enumerate(mse_per_channel):
        name = feature_names[c] if feature_names else f"ch_{c}"
        print(f"  {name:30s}  MSE = {mse:.6f}")

    return mse_per_channel, overall_mse


def compute_static_metrics(targets, preds, feature_names=None):
    """Per-feature MSE, R2, and Correlation for static site conditions."""
    n_feats = targets.shape[1]
    results = []
    print(f"\n{'='*60}")
    print(f"STATIC SITE-CONDITION PREDICTION")
    print(f"{'='*60}")
    print(f"  {'Feature':>30s}  {'MSE':>10s}  {'R2':>10s}  {'Corr':>10s}")
    for f in range(n_feats):
        mse = mean_squared_error(targets[:, f], preds[:, f])
        r2 = r2_score(targets[:, f], preds[:, f])
        
        # Calculate correlation (handle Nan if variance is 0)
        c_mat = np.corrcoef(targets[:, f], preds[:, f])
        if np.isnan(c_mat).any():
            corr = 0.0
        else:
            corr = c_mat[0, 1]
        
        name = feature_names[f] if feature_names is not None else str(f)
        results.append({"feature": f, "feature_name": name, "mse": mse, "r2": r2, "corr": corr})
        print(f"  {name:>30s}  {mse:10.6f}  {r2:10.4f}  {corr:10.4f}")

    overall_mse = mean_squared_error(targets, preds)
    # Macro-average R2 and Correlation
    avg_r2 = np.mean([r["r2"] for r in results])
    avg_corr = np.mean([r["corr"] for r in results])
    print(f"\n  Overall MSE: {overall_mse:.6f}  |  Mean R2: {avg_r2:.4f}  |  Mean Corr: {avg_corr:.4f}")
    return results, overall_mse, avg_r2, avg_corr




# ======================================================================
# Visualisations
# ======================================================================
def plot_latent_space(codes, metadata, output_dir, method="tsne"):
    """Visualise latent space coloured by point_id."""
    print(f"\nPlotting latent space ({method}) ...")

    # Get unique pids for colouring
    pids = [m["pid"] for m in metadata]
    unique_pids = list(set(pids))
    pid_to_idx = {p: i for i, p in enumerate(unique_pids)}
    colors = [pid_to_idx[p] for p in pids]

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(codes) - 1))
    else:
        reducer = PCA(n_components=2)

    emb = reducer.fit_transform(codes)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb[:, 0], emb[:, 1],
        c=colors, cmap="tab20", s=8, alpha=0.6,
    )
    plt.colorbar(scatter, label="Point index")
    plt.title(f"Latent space ({method.upper()}) coloured by point")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    path = os.path.join(output_dir, f"latent_{method}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved to {path}")


def plot_reconstruction_samples(inputs, recons, metadata, feature_names,
                                output_dir, n_samples=5):
    """Plot a few sample reconstructions overlaid on inputs."""
    print(f"\nPlotting {n_samples} reconstruction samples ...")
    indices = random.sample(range(len(inputs)), min(n_samples, len(inputs)))

    n_feats_to_plot = min(6, inputs.shape[2])

    for idx in indices:
        fig, axes = plt.subplots(n_feats_to_plot, 1, figsize=(14, 2.5 * n_feats_to_plot),
                                 sharex=True)
        if n_feats_to_plot == 1:
            axes = [axes]

        meta = metadata[idx]
        fig.suptitle(
            f"pid={meta['pid']}  year={meta['year']}  scenario={meta['scenario_id']}",
            fontsize=12,
        )

        for c, ax in enumerate(axes):
            name = feature_names[c] if feature_names else f"ch_{c}"
            ax.plot(inputs[idx, :, c], label="Input", alpha=0.7)
            ax.plot(recons[idx, :, c], label="Recon", alpha=0.7, linestyle="--")
            ax.set_ylabel(name, fontsize=9)
            if c == 0:
                ax.legend(fontsize=8)

        axes[-1].set_xlabel("Day of Year")
        plt.tight_layout()
        path = os.path.join(
            output_dir,
            f"recon_pid{meta['pid']}_y{meta['year']}_s{meta['scenario_id']}.png",
        )
        plt.savefig(path, dpi=120)
        plt.close()

    print(f"  Saved to {output_dir}/recon_*.png")


def plot_static_metrics(results, output_dir):
    """
    Plot R2 scores for static metrics.
    1. Histogram/Line plot of all R2 scores.
    2. Top 20 and Bottom 20 features by R2 in a single plot.
    """
    print(f"\nPlotting static metrics (R2) ...")

    # helper to unpack
    features = [str(r["feature_name"]) for r in results]
    r2_scores = [r["r2"] for r in results]
    
    # Sort by R2
    sorted_indices = np.argsort(r2_scores)
    sorted_features = np.array(features)[sorted_indices]
    sorted_r2 = np.array(r2_scores)[sorted_indices]
    
    # 1. Bar plot of all (if reasonable number) or distribution
    plt.figure(figsize=(12, 8))
    if len(results) <= 50:
         plt.barh(np.arange(len(results)), sorted_r2)
         plt.yticks(np.arange(len(results)), sorted_features, fontsize=6)
    else:
         plt.plot(sorted_r2, marker='.')
         plt.xlabel("Feature Rank")
    
    plt.title("Static Condition R2 Scores (All Features)")
    plt.xlabel("R2 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "static_r2_all.png"), dpi=150)
    plt.close()

    # 2. Top 20 and Lowest 20
    n_top = 20
    if len(results) < 2 * n_top:
        # Just plot all of them
        plot_features = sorted_features
        plot_r2 = sorted_r2
        title_suffix = "(All)"
    else:
        # Lowest 20 (first 20 of sorted)
        # Top 20 (last 20 of sorted)
        lowest_indices = np.arange(n_top)
        highest_indices = np.arange(len(results) - n_top, len(results))
        
        select_indices = np.concatenate([lowest_indices, highest_indices])
        
        plot_features = sorted_features[select_indices]
        plot_r2 = sorted_r2[select_indices]
        title_suffix = "(Bottom 20 & Top 20)"
        
    plt.figure(figsize=(12, 12))
    # Horizontal bar plot
    y_pos = np.arange(len(plot_features))
    
    # Color coding: red for low, blue for high
    # We can normalize color map based on value
    norm = plt.Normalize(vmin=min(plot_r2), vmax=max(plot_r2))
    cmap = plt.cm.RdYlBu
    colors = cmap(norm(plot_r2))
    
    bars = plt.barh(y_pos, plot_r2, align='center', color=colors)
    plt.yticks(y_pos, plot_features, fontsize=9)
    plt.xlabel('R2 Score')
    plt.title(f'Static Condition R2 Scores {title_suffix}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(plot_r2):
        plt.text(v, i, f" {v:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "static_r2_top_bottom.png"), dpi=150)
    plt.close()
    
    print(f"  Saved static R2 plots to {output_dir}")



# ======================================================================
# Main
# ======================================================================
def evaluate(config: InverseExperimentConfig, num_vis: int = 5):
    """Full evaluation pipeline."""
    output_dir = config.output_dir
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    seed = config.training.random_seed
    setup_reproducibility(seed)
    random.seed(seed)

    device = get_device(config.training.device)

    # ------------------------------------------------------------------
    # Prepare data (fits scalers on training points only, then transforms
    # the full dataframes so every split uses training-only statistics).
    # ------------------------------------------------------------------
    print("Preparing data (scalers fitted on training points) ...")
    train_points = config.data.get_split_config("train")["points"]
    data_bundle = prepare_inverse_data(
        driver_response_path=config.data.driver_response_path,
        management_path=config.data.management_path,
        init_cond_path=config.data.init_cond_path,
        train_points=train_points,
        scaler_dir=os.path.join(output_dir, "scalers"),
    )

    # Choose test split; fall back to val
    try:
        split_cfg = config.data.get_split_config("test")
        split_name = "test"
    except Exception:
        split_cfg = config.data.get_split_config("val")
        split_name = "val"

    print(f"Evaluating on '{split_name}' split "
          f"({len(split_cfg['points'])} points, "
          f"{len(split_cfg['scenarios'])} scenarios, "
          f"{len(split_cfg['years'])} years) ...")

    ds = DayCentInverseDataset(
        driver_response_df=data_bundle["driver_response_df"],
        management_df=data_bundle["management_df"],
        init_cond_df=data_bundle["init_cond_df"],
        split_config=split_cfg,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    in_channels = ds.input_channels
    static_channels = ds.static_channels
    code_dim = config.training.code_dim
    num_layers = config.training.num_layers

    model = InverseModel(
        in_channels=in_channels,
        static_channels=static_channels,
        code_dim=code_dim,
        num_layers=num_layers,
        device=device,
    ).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(output_dir, "best_inverse_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    results = encode_dataset(
        model, ds, device,
        batch_size=config.training.batch_size,
    )

    # Feature names for driver-response + management channels
    feature_names = ds.dr_feature_cols + ds.mgmt_cols

    # Feature names for static site conditions
    static_feature_names = list(ds.init_conditions.columns)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    recon_mse, overall_recon = compute_reconstruction_metrics(
        results["inputs"], results["reconstructions"], feature_names
    )
    static_results, static_mse, static_r2, static_corr = compute_static_metrics(
        results["static_targets"], results["static_preds"], feature_names=static_feature_names
    )

    plot_static_metrics(static_results, eval_dir)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to {eval_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DayCent inverse model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to inverse experiment YAML config file",
    )
    parser.add_argument(
        "--num-vis", type=int, default=5,
        help="Number of reconstruction samples to visualise",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    config = InverseExperimentConfig.from_yaml(args.config)
    evaluate(config, num_vis=args.num_vis)


if __name__ == "__main__":
    main()
