"""
Rank static site-condition predictions by correlation (and RMSE) on the test split.
Reports ALL static features, not just the IMPORTANT_STATIC_FEATURES subset.

Usage:
    conda run -n wstatt python rank_static_predictions.py \
        --config configs/inverse/inverse_1.yaml [--topk 15]
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

from data.inverse import DayCentInverseDataset, prepare_inverse_data
from model.inverse import InverseModel
from utils.training import setup_reproducibility, get_device
from utils.config import InverseExperimentConfig
from utils.metrics import compute_per_channel_metrics


@torch.no_grad()
def encode_split(model, dataset, device, batch_size=256):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--topk", type=int, default=15,
                        help="How many top features to print (0 = all)")
    args = parser.parse_args()

    config = InverseExperimentConfig.from_yaml(args.config)
    setup_reproducibility(config.training.random_seed)
    # device = get_device(config.training.device)
    device = torch.device("cuda:2")

    train_points = config.data.get_split_config("train")["points"]
    data_bundle = prepare_inverse_data(
        driver_response_path=config.data.driver_response_path,
        management_path=config.data.management_path,
        init_cond_path=config.data.init_cond_path,
        train_points=train_points,
        scaler_dir=os.path.join(config.output_dir, "scalers"),
    )

    split_cfg = config.data.get_split_config(args.split)
    dataset = DayCentInverseDataset(
        driver_response_df=data_bundle["driver_response_df"],
        management_df=data_bundle["management_df"],
        init_cond_df=data_bundle["init_cond_df"],
        split_config=split_cfg,
    )

    static_feature_names = list(data_bundle["init_cond_df"].columns)

    model = InverseModel(
        in_channels=dataset.input_channels,
        static_channels=dataset.static_channels,
        code_dim=config.training.code_dim,
        num_layers=config.training.num_layers,
        device=device,
    ).to(device)

    ckpt = os.path.join(config.output_dir, "best_inverse_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found at {ckpt}")
        sys.exit(1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt}")

    targets, preds = encode_split(model, dataset, device, config.training.batch_size)
    print(f"Evaluating {len(targets)} samples from '{args.split}' split")

    results = compute_per_channel_metrics(targets, preds, channel_names=static_feature_names)

    # Sort by correlation descending
    results.sort(key=lambda r: r["corr"], reverse=True)

    topk = args.topk if args.topk > 0 else len(results)
    print(f"\n{'='*65}")
    print(f"Static feature predictions ranked by Pearson correlation (top {topk})")
    print(f"{'='*65}")
    print(f"  {'Rank':>4}  {'Feature':>25}  {'Corr':>8}  {'RMSE':>10}  {'R2':>8}")
    print(f"  {'-'*60}")
    for rank, r in enumerate(results[:topk], 1):
        print(f"  {rank:>4}  {r['channel_name']:>25}  {r['corr']:8.4f}  {r['rmse']:10.6f}  {r['r2']:8.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
