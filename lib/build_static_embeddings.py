"""
Build static-feature embeddings as a drop-in replacement for inverse model codes.

Instead of using learned latent embeddings (avg_codes from inverse model),
this script uses the top-K best-predicted static site-condition features
(ranked by the inverse model's prediction correlation) as embeddings.

The output is compatible with selection/embedding_strategy.py:
    - avg_codes.npy      → (P, K) float32 array
    - avg_codes_pids.npy → (P,) string array of point IDs

Usage:
    conda run -n wstatt python build_static_embeddings.py \
        --config configs/inverse/inverse_1.yaml \
        --topk 32 \
        --output-dir /users/6/mehta423/daycent/output/static_emb_32
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data.inverse import DayCentInverseDataset, prepare_inverse_data
from model.inverse import InverseModel
from utils.training import setup_reproducibility, get_device
from utils.config import InverseExperimentConfig
from utils.metrics import compute_per_channel_metrics


@torch.no_grad()
def encode_split(model, dataset, device, batch_size=256):
    """Encode test split to get static feature predictions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, targets = [], []
    for batch in loader:
        seq = batch["sequence"].to(device)
        out = model({"sequence": seq})
        preds.append(out["static_pred"].cpu().numpy())
        targets.append(batch["static_target"].numpy())
    return np.concatenate(targets), np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(
        description="Build static-feature embeddings from inverse model rankings."
    )
    parser.add_argument("--config", required=True, help="Inverse model config YAML")
    parser.add_argument("--topk", type=int, default=32, help="Number of features to select (default: 32)")
    parser.add_argument("--output-dir", required=True, help="Directory to save avg_codes.npy, etc.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to use for ranking (default: test)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}")
        sys.exit(1)

    # ====================================================================
    # Load config and setup
    # ====================================================================
    config = InverseExperimentConfig.from_yaml(args.config)
    setup_reproducibility(config.training.random_seed)
    device = get_device(config.training.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ====================================================================
    # Load data and model
    # ====================================================================
    print(f"Loading inverse data from {config.data.init_cond_path} …")
    train_points = config.data.get_split_config("train")["points"]
    data_bundle = prepare_inverse_data(
        driver_response_path=config.data.driver_response_path,
        management_path=config.data.management_path,
        init_cond_path=config.data.init_cond_path,
        train_points=train_points,
        scaler_dir=os.path.join(config.output_dir, "scalers"),
    )

    # Build dataset for the specified split
    split_cfg = config.data.get_split_config(args.split)
    dataset = DayCentInverseDataset(
        driver_response_df=data_bundle["driver_response_df"],
        management_df=data_bundle["management_df"],
        init_cond_df=data_bundle["init_cond_df"],
        split_config=split_cfg,
    )

    all_static_feature_names = list(data_bundle["init_cond_df"].columns)
    print(f"Total static features available: {len(all_static_feature_names)}")
    print(f"  Features: {all_static_feature_names}")

    # Load model and checkpoint
    model = InverseModel(
        in_channels=dataset.input_channels,
        static_channels=dataset.static_channels,
        code_dim=config.training.code_dim,
        num_layers=config.training.num_layers,
        device=device,
    ).to(device)

    ckpt_path = os.path.join(config.output_dir, "best_inverse_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")

    # ====================================================================
    # Rank static features by correlation on the chosen split
    # ====================================================================
    print(f"\nEncoding {args.split} split ({len(dataset)} samples) …")
    targets, preds = encode_split(model, dataset, device, config.training.batch_size)

    print(f"Computing metrics for {len(all_static_feature_names)} features …")
    results = compute_per_channel_metrics(targets, preds, channel_names=all_static_feature_names)
    results.sort(key=lambda r: r["corr"], reverse=True)

    # Print ranking
    topk = min(args.topk, len(results))
    print(f"\n{'='*70}")
    print(f"Top {topk} static features by correlation (evaluated on {args.split})")
    print(f"{'='*70}")
    print(f"  {'Rank':>4}  {'Feature':>25}  {'Corr':>8}  {'RMSE':>10}  {'R2':>8}")
    print(f"  {'-'*65}")

    selected_features = []
    for rank, r in enumerate(results[:topk], 1):
        print(f"  {rank:>4}  {r['channel_name']:>25}  {r['corr']:8.4f}  {r['rmse']:10.6f}  {r['r2']:8.4f}")
        selected_features.append(r['channel_name'])

    print(f"{'='*70}")

    # ====================================================================
    # Load ALL points from raw init_cond Excel (not just inverse splits)
    # ====================================================================
    print(f"\nLoading all points from {config.data.init_cond_path} …")
    init_cond_all = pd.read_excel(config.data.init_cond_path).set_index("id")
    init_cond_all.dropna(axis=1, inplace=True)
    print(f"  Loaded {len(init_cond_all)} points with {len(init_cond_all.columns)} features")

    # Select only top-K features
    print(f"Selecting top {topk} features …")
    init_cond_selected = init_cond_all[selected_features].copy()

    # ====================================================================
    # Normalize using training points from inverse config
    # ====================================================================
    print(f"Fitting StandardScaler on {len(train_points)} training points …")

    # Convert train_points to match init_cond_all index
    train_ids_int = []
    for p in train_points:
        try:
            train_ids_int.append(int(p))
        except ValueError:
            train_ids_int.append(p)

    train_mask = init_cond_all.index.isin(train_ids_int)
    train_subset = init_cond_all.loc[train_mask, selected_features]
    print(f"  Training subset has {len(train_subset)} rows (matched from {len(train_points)} train points)")

    scaler = StandardScaler()
    scaler.fit(train_subset)

    # Transform all points
    init_cond_normalized = scaler.transform(init_cond_selected)
    print(f"  Normalized shape: {init_cond_normalized.shape}")

    # ====================================================================
    # Prepare output: sort by point ID
    # ====================================================================
    # Create a DataFrame for easier sorting
    embedding_df = pd.DataFrame(
        init_cond_normalized,
        index=init_cond_selected.index,
        columns=[f"feat_{i}" for i in range(init_cond_normalized.shape[1])]
    )

    # Sort by index (point ID)
    embedding_df_sorted = embedding_df.sort_index(key=lambda x: x.astype(int) if x.dtype == object else x)

    avg_codes_final = embedding_df_sorted.values.astype(np.float32)
    avg_codes_pids_final = np.array([str(pid) for pid in embedding_df_sorted.index])

    print(f"\nFinal embedding shape: {avg_codes_final.shape}")
    print(f"Final PIDs shape: {avg_codes_pids_final.shape}")

    # ====================================================================
    # Save embeddings
    # ====================================================================
    codes_path = os.path.join(args.output_dir, "avg_codes.npy")
    pids_path = os.path.join(args.output_dir, "avg_codes_pids.npy")
    features_path = os.path.join(args.output_dir, "feature_names.json")

    np.save(codes_path, avg_codes_final)
    np.save(pids_path, avg_codes_pids_final)

    with open(features_path, "w") as f:
        json.dump({
            "selected_features": selected_features,
            "topk": topk,
            "ranking_split": args.split,
            "all_features": all_static_feature_names,
        }, f, indent=2)

    print(f"\nSaved embeddings to {args.output_dir}")
    print(f"  avg_codes.npy       {avg_codes_final.shape}")
    print(f"  avg_codes_pids.npy  {avg_codes_pids_final.shape}")
    print(f"  feature_names.json  (metadata)")


if __name__ == "__main__":
    main()
