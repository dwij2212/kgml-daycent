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
import math
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

IMPORTANT_STATIC_FEATURES = [
    "elev",
    "slclay",
    "slfldc",
    "slph",
    "slsand",
    "slwltp",
    "sitlat",
    "som2ci"
]

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
    """
    Computes metrics ONLY for features that match IMPORTANT_STATIC_FEATURES.
    Returns filtered results and the indices of important features.
    """
    n_feats = targets.shape[1]
    results = []
    important_indices = []

    print(f"\n{'='*60}")
    print(f"STATIC SITE-CONDITION PREDICTION (Important Features Only)")
    print(f"{'='*60}")
    print(f"  {'Feature':>20s}  {'MSE':>10s}  {'R2':>10s}  {'Corr':>10s}")

    for f in range(n_feats):
        name = feature_names[f] if feature_names is not None else str(f)
        name_lower = name.lower()

        # --- EARLY FILTERING ---
        # Check if this feature starts with any of the important prefixes
        is_important = any(name_lower.startswith(prefix) for prefix in IMPORTANT_STATIC_FEATURES)
        
        if not is_important:
            continue

        # Keep track of the index for plotting later
        important_indices.append(f)

        # Compute metrics
        mse = mean_squared_error(targets[:, f], preds[:, f])
        r2 = r2_score(targets[:, f], preds[:, f])
        
        c_mat = np.corrcoef(targets[:, f], preds[:, f])
        corr = 0.0 if np.isnan(c_mat).any() else c_mat[0, 1]
        
        results.append({
            "feature_idx": f,
            "feature_name": name,
            "mse": mse,
            "r2": r2,
            "corr": corr
        })
        print(f"  {name:>20s}  {mse:10.6f}  {r2:10.4f}  {corr:10.4f}")

    if not results:
        print("  No important features found!")
        return results, 0, 0, 0, []

    # Compute averages using ONLY the filtered results
    avg_mse = np.mean([r["mse"] for r in results])
    avg_r2 = np.mean([r["r2"] for r in results])
    avg_corr = np.mean([r["corr"] for r in results])

    print(f"{'-'*60}")
    print(f"  AVG (Important)     {avg_mse:10.6f}  {avg_r2:10.4f}  {avg_corr:10.4f}")
    
    return results, avg_mse, avg_r2, avg_corr, important_indices


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
    Only plots IMPORTANT_STATIC_FEATURES (prefix-based).
    """

    print(f"\nPlotting static metrics (R2) ...")

    # Unpack
    features = [str(r["feature_name"]) for r in results]
    r2_scores = [r["r2"] for r in results]

    # -------------------------------------------------------------
    # Filter by IMPORTANT_STATIC_FEATURES (prefix match, lowercase)
    # -------------------------------------------------------------
    important_features = []
    important_r2 = []

    for feat, r2 in zip(features, r2_scores):
        feat_lower = feat.lower()
        if any(feat_lower.startswith(prefix) for prefix in IMPORTANT_STATIC_FEATURES):
            important_features.append(feat)
            important_r2.append(r2)

    if len(important_features) == 0:
        print("  No important features found. Skipping static R2 plot.")
        return

    print(f"  Found {len(important_features)} important features. Plotting only these.")

    # Convert to numpy for sorting
    important_features = np.array(important_features)
    important_r2 = np.array(important_r2)

    # -------------------------------------------------------------
    # Sort by R2
    # -------------------------------------------------------------
    sorted_indices = np.argsort(important_r2)
    sorted_features = important_features[sorted_indices]
    sorted_r2 = important_r2[sorted_indices]

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(10, max(6, 0.4 * len(sorted_features))))

    y_pos = np.arange(len(sorted_features))

    norm = plt.Normalize(vmin=min(sorted_r2), vmax=max(sorted_r2))
    cmap = plt.cm.RdYlBu
    colors = cmap(norm(sorted_r2))

    plt.barh(y_pos, sorted_r2, color=colors)
    plt.yticks(y_pos, sorted_features, fontsize=9)
    plt.xlabel("R2 Score")
    plt.title("Static Condition R2 Scores (Important Features Only)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Add value labels
    for i, v in enumerate(sorted_r2):
        plt.text(v, i, f" {v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "static_r2_important_only.png"), dpi=150)
    plt.close()

    print(f"  Saved static R2 plot to {output_dir}/static_r2_important_only.png")

def plot_static_correlation_bar(results, output_dir):
    """Bar chart of Correlation scores for important features."""
    if not results:
        return

    # Sort by Correlation
    sorted_results = sorted(results, key=lambda x: x["corr"])
    names = [r["feature_name"] for r in sorted_results]
    corrs = [r["corr"] for r in sorted_results]

    plt.figure(figsize=(10, max(6, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    
    # Color bars by value
    norm = plt.Normalize(vmin=min(corrs), vmax=max(corrs))
    cmap = plt.cm.RdYlBu
    colors = cmap(norm(corrs))

    plt.barh(y_pos, corrs, color=colors)
    plt.yticks(y_pos, names, fontsize=9)
    plt.xlabel("Correlation Coefficient")
    plt.title("Prediction Correlation (Important Features Only)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Add value labels
    for i, v in enumerate(corrs):
        plt.text(v, i, f" {v:.2f}", va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, "static_correlation_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved correlation bar chart to {path}")


def plot_static_scatter(results, targets, preds, output_dir):
    """
    Grid of scatter plots (True vs Predicted) for important features.
    Draws the x=y line for reference.
    """
    if not results:
        return
    
    # sort results by correlation for better visualisation
    results = sorted(results, key=lambda x: x["corr"], reverse=True)
    results = results[:9]
    
    n_plots = len(results)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten to make indexing easy

    print(f"  Plotting scatter plots for {n_plots} features...")
    

    for i, res in enumerate(results):
        ax = axes[i]
        f_idx = res["feature_idx"]
        name = res["feature_name"]
        
        # Get raw data
        y_true = targets[:, f_idx]
        y_pred = preds[:, f_idx]
        
        # Determine axis limits to make the plot square and cover all points
        d_min = min(y_true.min(), y_pred.min())
        d_max = max(y_true.max(), y_pred.max())
        # Add slight padding
        span = d_max - d_min
        d_min -= span * 0.05
        d_max += span * 0.05

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=10, c='steelblue', edgecolors='none')
        
        # x = y line
        ax.plot([d_min, d_max], [d_min, d_max], 'k--', lw=1.5, label="Perfect Fit")
        
        ax.set_title(f"{name}\nCorr: {res['corr']:.2f} | R2: {res['r2']:.2f}", fontsize=10)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        
        # Force square aspect ratio
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(d_min, d_max)
        ax.set_ylim(d_min, d_max)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "static_scatter_plots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved scatter plots to {path}")
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
    
    # Compute metrics (and get filtered results + indices)
    static_results, static_mse, static_r2, static_corr, imp_indices = compute_static_metrics(
        results["static_targets"], 
        results["static_preds"], 
        feature_names=static_feature_names
    )

    # Plot Bar Chart
    plot_static_correlation_bar(static_results, eval_dir)

    # Plot Scatter Plots (Pass full targets/preds; function uses indices inside static_results)
    plot_static_scatter(
        static_results, 
        results["static_targets"], 
        results["static_preds"], 
        eval_dir
    )

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
