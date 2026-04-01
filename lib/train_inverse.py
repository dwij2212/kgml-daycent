"""
Training script for DayCent inverse modelling.

The inverse model learns latent representations of spatial points from
driver+response time-series.  Training uses three loss terms:

1. Reconstruction loss  (MSE on input sequence)
2. Static loss          (MSE on predicted vs true static site conditions)
3. Contrastive loss     (NT-Xent / SimCLR on same-point, different-year pairs)

Usage:
    python train_inverse.py --config configs/inverse/inverse_1.yaml
"""
import argparse
import os
import sys
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project imports ----
from data.inverse import DayCentInverseDataset, prepare_inverse_data
from model.inverse import InverseModel, SimCLRLoss
from utils.training import setup_reproducibility, get_device
from utils.config import InverseExperimentConfig


# ======================================================================
# Data Loading — normalisation is done here, fit on training points only
# ======================================================================
def create_datasets(config: InverseExperimentConfig):
    """
    Load raw CSVs, normalise using training-point statistics, then
    build Dataset instances for each split.

    Returns:
        (train_ds, val_ds, prepared_data)
    """
    data_cfg = config.data
    train_split = data_cfg.get_split_config('train')
    train_points = train_split['points']
    train_years = train_split['years']

    # Normalise using training points only
    print("\n--- Preparing & normalising data ---")
    prepared = prepare_inverse_data(
        driver_response_path=data_cfg.driver_response_path,
        management_path=data_cfg.management_path,
        init_cond_path=data_cfg.init_cond_path,
        train_points=train_points,
        train_years=train_years,
        scaler_dir=config.result_dir,
    )

    dr_df = prepared['driver_response_df']
    mgmt_df = prepared['management_df']
    init_cond_df = prepared['init_cond_df']

    # Build datasets for each split
    print("\nCreating training dataset …")
    train_ds = DayCentInverseDataset(
        driver_response_df=dr_df,
        management_df=mgmt_df,
        init_cond_df=init_cond_df,
        split_config=train_split,
        year_emb_dim=config.training.year_emb_dim,
    )

    val_ds = None
    val_split = data_cfg.get_split_config('val')
    if val_split:
        print("Creating validation dataset …")
        val_ds = DayCentInverseDataset(
            driver_response_df=dr_df,
            management_df=mgmt_df,
            init_cond_df=init_cond_df,
            split_config=val_split,
            year_emb_dim=config.training.year_emb_dim,
        )

    return train_ds, val_ds, prepared


# ======================================================================
# Contrastive Pair Sampling
# ======================================================================
def build_contrastive_batches(dataset, batch_size, shuffle=True):
    """
    Generator that yields (anchor_batch, positive_batch) where each pair
    shares the same point but has different (scenario, year) combos.

    Strategy:
        - Group all sample indices by point_id.
        - For each mini-batch, sample `batch_size` points (with replacement
          if necessary), then for each point pick 2 random indices as
          anchor / positive.

    Yields:
        anchor_indices, positive_indices  (lists of dataset indices)
    """
    # Build point -> list of dataset indices mapping
    point_to_indices = {}
    n_scenarios = dataset.len_scenarios
    n_years = dataset.len_years
    n_points = dataset.len_points

    for p_idx in range(n_points):
        pid = dataset.point_ids[p_idx]
        indices = []
        for s_idx in range(n_scenarios):
            for y_idx in range(n_years):
                flat = (
                    s_idx * dataset.year_point_block_size
                    + y_idx * dataset.point_block_size
                    + p_idx
                )
                indices.append(flat)
        point_to_indices[pid] = indices

    # Only keep points with >= 2 samples (needed for contrastive pairs)
    eligible_pids = [
        pid for pid, idxs in point_to_indices.items() if len(idxs) >= 2
    ]

    if shuffle:
        random.shuffle(eligible_pids)

    n_batches = max(1, len(eligible_pids) // batch_size)

    for b in range(n_batches):
        anchor_idxs = []
        positive_idxs = []

        batch_pids = eligible_pids[
            b * batch_size : (b + 1) * batch_size
        ]

        for pid in batch_pids:
            pool = point_to_indices[pid]
            # pick 2 distinct indices
            pair = random.sample(pool, 2)
            anchor_idxs.append(pair[0])
            positive_idxs.append(pair[1])

        yield anchor_idxs, positive_idxs


def collate_pairs(dataset, anchor_idxs, positive_idxs):
    """
    Fetch samples from dataset and collate into a pair of batch dicts.
    Returns (anchor_batch, positive_batch) with stacked tensors.
    """
    def stack_batch(indices):
        samples = [dataset[i] for i in indices]
        batch = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals)
            else:
                batch[key] = vals
        return batch

    return stack_batch(anchor_idxs), stack_batch(positive_idxs)


# ======================================================================
# Training Loop
# ======================================================================
def train_one_epoch(
    model,
    dataset,
    optimizer,
    criterion_mse,
    criterion_contrastive,
    device,
    config,
):
    """
    Train for one epoch using contrastive pair sampling.
    """
    model.train()
    train_cfg = config.training
    batch_size = train_cfg.batch_size
    recon_w = train_cfg.recon_weight
    static_w = train_cfg.static_weight
    contrastive_w = train_cfg.contrastive_weight
    weight_sum = recon_w + static_w + contrastive_w

    total_loss = 0.0
    total_recon = 0.0
    total_static = 0.0
    total_contrastive = 0.0
    n_batches = 0

    for anchor_idxs, positive_idxs in build_contrastive_batches(
        dataset, batch_size, shuffle=True
    ):
        anchor_batch, positive_batch = collate_pairs(
            dataset, anchor_idxs, positive_idxs
        )

        # Move to device
        anchor_seq = anchor_batch["sequence"].to(device)
        positive_seq = positive_batch["sequence"].to(device)
        anchor_static = anchor_batch["static_target"].to(device)
        positive_static = positive_batch["static_target"].to(device)

        # Concatenate anchor + positive for one forward pass
        input_seq = torch.cat([anchor_seq, positive_seq], dim=0)
        static_target = torch.cat([anchor_static, positive_static], dim=0)

        optimizer.zero_grad()

        out = model({"sequence": input_seq})
        code = out["code"]                    # (2B, code_dim)
        reconstruction = out["reconstruction"]  # (2B, T, C)
        static_pred = out["static_pred"]      # (2B, S)

        # --- Reconstruction loss ---
        recon_loss = criterion_mse(reconstruction, input_seq)

        # --- Static loss ---
        static_loss = criterion_mse(static_pred, static_target)

        # --- Contrastive loss ---
        contrastive_loss = criterion_contrastive(code)

        # --- Combined loss ---
        loss = (
            recon_w * recon_loss
            + static_w * static_loss
            + contrastive_w * contrastive_loss
        ) / weight_sum

        loss.backward()

        # Gradient clipping
        max_norm = train_cfg.grad_clip_norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_static += static_loss.item()
        total_contrastive += contrastive_loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_static / n_batches,
        total_contrastive / n_batches,
    )


@torch.no_grad()
def validate(model, dataset, criterion_mse, criterion_contrastive, device, config):
    """Run validation with contrastive pair sampling (no grad)."""
    model.eval()
    train_cfg = config.training
    batch_size = train_cfg.batch_size
    recon_w = train_cfg.recon_weight
    static_w = train_cfg.static_weight
    contrastive_w = train_cfg.contrastive_weight
    weight_sum = recon_w + static_w + contrastive_w

    total_loss = 0.0
    total_recon = 0.0
    total_static = 0.0
    total_contrastive = 0.0
    n_batches = 0

    for anchor_idxs, positive_idxs in build_contrastive_batches(
        dataset, batch_size, shuffle=False
    ):
        anchor_batch, positive_batch = collate_pairs(
            dataset, anchor_idxs, positive_idxs
        )

        anchor_seq = anchor_batch["sequence"].to(device)
        positive_seq = positive_batch["sequence"].to(device)
        anchor_static = anchor_batch["static_target"].to(device)
        positive_static = positive_batch["static_target"].to(device)

        input_seq = torch.cat([anchor_seq, positive_seq], dim=0)
        static_target = torch.cat([anchor_static, positive_static], dim=0)

        out = model({"sequence": input_seq})
        code = out["code"]
        reconstruction = out["reconstruction"]
        static_pred = out["static_pred"]

        recon_loss = criterion_mse(reconstruction, input_seq)
        static_loss = criterion_mse(static_pred, static_target)
        contrastive_loss = criterion_contrastive(code)

        loss = (
            recon_w * recon_loss
            + static_w * static_loss
            + contrastive_w * contrastive_loss
        ) / weight_sum

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_static += static_loss.item()
        total_contrastive += contrastive_loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_static / n_batches,
        total_contrastive / n_batches,
    )


# ======================================================================
# Main
# ======================================================================
def train(config: InverseExperimentConfig):
    """Full training pipeline."""
    train_cfg = config.training
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    seed = train_cfg.random_seed
    setup_reproducibility(seed)
    random.seed(seed)

    # Device
    device = get_device(train_cfg.device)
    print(f"Device: {device}")

    # Data — normalisation happens here, fit on training points only
    train_ds, val_ds, prepared = create_datasets(config)

    # Model
    in_channels = train_ds.input_channels
    static_channels = train_ds.static_channels
    code_dim = train_cfg.code_dim
    num_layers = train_cfg.num_layers
    dropout = train_cfg.dropout

    model = InverseModel(
        in_channels=in_channels,
        static_channels=static_channels,
        code_dim=code_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")
    print(f"  in_channels={in_channels}, static_channels={static_channels}, "
          f"code_dim={code_dim}, num_layers={num_layers}")

    # Loss
    criterion_mse = nn.MSELoss()
    temperature = train_cfg.temperature
    criterion_contrastive = SimCLRLoss(temperature=temperature)

    # Optimizer
    lr = train_cfg.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    epochs = train_cfg.epochs
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")

    for epoch in range(epochs):
        t0 = time.time()

        tr_loss, tr_recon, tr_static, tr_contr = train_one_epoch(
            model, train_ds, optimizer, criterion_mse,
            criterion_contrastive, device, config,
        )
        train_losses.append(tr_loss)

        msg = (
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {tr_loss:.4f} (R={tr_recon:.4f} S={tr_static:.4f} C={tr_contr:.4f})"
        )

        if val_ds is not None:
            va_loss, va_recon, va_static, va_contr = validate(
                model, val_ds, criterion_mse, criterion_contrastive, device, config,
            )
            val_losses.append(va_loss)
            msg += (
                f" | Val: {va_loss:.4f} (R={va_recon:.4f} S={va_static:.4f} C={va_contr:.4f})"
            )

            if va_loss < best_val_loss and va_loss > 0:
                best_val_loss = va_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "best_inverse_model.pt"),
                )
                msg += " *"
        else:
            if tr_loss < best_val_loss and tr_loss > 0:
                best_val_loss = tr_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "best_inverse_model.pt"),
                )
                msg += " *"

        elapsed = time.time() - t0
        msg += f"  ({elapsed:.1f}s)"
        print(msg)

    # Save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Inverse Model Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inverse_loss_curves.png"), dpi=150)
    plt.close()
    print(f"\nTraining complete. Best model saved to {output_dir}/best_inverse_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train DayCent inverse model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to inverse experiment YAML config file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    config = InverseExperimentConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
