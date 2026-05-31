"""
Generate and optionally run individual-pool delta context ablations.

Each case predicts one annual pool delta and varies only the previous December
pool values provided as context. Pool-level teacher-forced metrics are the
intended comparison; partial-state SOMSC or rollout metrics are not.

Examples:
    python scripts/yearly/run_pool_synergy_cases.py --dry-run
    python scripts/yearly/run_pool_synergy_cases.py --targets som2c_soil --contexts self all_four --train --eval
    python scripts/yearly/run_pool_synergy_cases.py --contexts all_four drop_som2c_soil drop_surface --train --eval
"""
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml


LIB_DIR = Path(__file__).resolve().parents[2]
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from data.yearly import prepare_yearly_data
from eval_yearly_somsc import evaluate_experiment
from train_yearly_somsc import train
from utils.config import ExperimentConfig


SOIL_POOLS = ["som1c_soil", "som2c_soil", "som3c"]
FULL_POOLS = SOIL_POOLS + ["som2c_surface"]
DEFAULT_CONTEXTS = ["somsc", "self", "all_soil", "all_four"]

CONTEXT_CASES = {
    "somsc": {"prev_state_context": "somsc"},
    "self": {"prev_state_context": "target_pools"},
    "all_soil": {"prev_state_context": "soil_pools"},
    "all_four": {"prev_state_context": "full_pools"},
    "only_som1c_soil": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som1c_soil"],
    },
    "only_som2c_soil": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som2c_soil"],
    },
    "only_som3c": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som3c"],
    },
    "only_surface": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som2c_surface"],
    },
    "drop_som1c_soil": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som2c_soil", "som3c", "som2c_surface"],
    },
    "drop_som2c_soil": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som1c_soil", "som3c", "som2c_surface"],
    },
    "drop_som3c": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": ["som1c_soil", "som2c_soil", "som2c_surface"],
    },
    "drop_surface": {
        "prev_state_context": "custom_pools",
        "context_pool_cols": SOIL_POOLS,
    },
}


def _load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, config_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)


def build_case_config(base_config: dict, target_pool: str, context_name: str) -> dict:
    cfg = copy.deepcopy(base_config)
    context = CONTEXT_CASES[context_name]
    case_name = f"{target_pool}_{context_name}"
    base_experiment_id = base_config["experiment_id"]

    cfg["experiment_id"] = f"{base_experiment_id}_pool_synergy_{case_name}"
    cfg["description"] = (
        f"Predict annual {target_pool} delta with previous-state context: {context_name}."
    )
    cfg.setdefault("model", {})
    cfg["model"].update(
        {
            "model_type": "yearly_somsc_state",
            "target_mode": "pool_deltas",
            "target_pool_cols": [target_pool],
            "context_pool_cols": None,
        }
    )
    cfg["model"].update(context)

    cfg.setdefault("wandb", {})
    cfg["wandb"]["name"] = cfg["experiment_id"]
    cfg["wandb"]["notes"] = cfg["description"]
    tags = list(cfg["wandb"].get("tags", []))
    for tag in ["yearly_pool_synergy", target_pool, context_name]:
        if tag not in tags:
            tags.append(tag)
    cfg["wandb"]["tags"] = tags
    return cfg


def summarize_result(target_pool: str, context_name: str, result: dict) -> dict:
    metrics = result["metrics"]["pool_delta"][target_pool]
    return {
        "target_pool": target_pool,
        "context": context_name,
        "context_pool_cols": ",".join(
            result["predictions"].get("context_pool_cols", [])
        ),
        "n": metrics["n_samples"],
        "delta_rmse": metrics["rmse"],
        "delta_mae": metrics["mae"],
        "delta_r2": metrics["r2"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run individual-pool delta context synergy ablations"
    )
    parser.add_argument(
        "--base-config",
        default="configs/yearly/experiment_state_v1.yaml",
        help="Base yearly config to clone.",
    )
    parser.add_argument(
        "--out-dir",
        default="configs/yearly/pool_synergy",
        help="Directory for generated configs.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=FULL_POOLS,
        default=FULL_POOLS,
        help="Pool deltas to predict independently.",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=sorted(CONTEXT_CASES),
        default=DEFAULT_CONTEXTS,
        help="Previous-state contexts to compare.",
    )
    parser.add_argument("--train", action="store_true", help="Train generated cases.")
    parser.add_argument("--eval", action="store_true", help="Evaluate generated cases.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override configured training epochs for screening runs.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Split for teacher-forced pool-delta evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of sample trajectory plots per evaluation.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Save prediction CSVs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate configs and print commands.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = LIB_DIR / base_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = LIB_DIR / out_dir

    base_config = _load_yaml(base_path)
    generated = []
    for target_pool in args.targets:
        for context_name in args.contexts:
            cfg = build_case_config(base_config, target_pool, context_name)
            if args.epochs is not None:
                cfg.setdefault("training", {})["epochs"] = args.epochs
            out_path = out_dir / f"{target_pool}_{context_name}.yaml"
            _write_yaml(out_path, cfg)
            generated.append((target_pool, context_name, out_path))

    print("Generated pool-synergy configs:")
    for target_pool, context_name, path in generated:
        print(f"  {target_pool} / {context_name}: {path}")

    if args.dry_run or not args.train and not args.eval:
        print("\nSuggested commands:")
        for _, _, path in generated:
            print(f"  python train_yearly_somsc.py --config {path}")
            print(
                "  python eval_yearly_somsc.py "
                f"--config {path} --split {args.split} "
                "--eval-mode teacher_forced --save-csv"
            )
        return

    first_config = ExperimentConfig.from_yaml(str(generated[0][2]))
    prepared_data = prepare_yearly_data(first_config)
    summary_rows = []
    for target_pool, context_name, path in generated:
        config = ExperimentConfig.from_yaml(str(path))
        if args.train:
            train(config, prepared_data=prepared_data)
        if args.eval:
            result = evaluate_experiment(
                config,
                split=args.split,
                num_samples=args.num_samples,
                save_csv=args.save_csv,
                prepared_data=prepared_data,
                eval_mode="teacher_forced",
            )
            summary_rows.append(summarize_result(target_pool, context_name, result))

    if summary_rows:
        summary_path = (
            Path(first_config.output_dir).parent
            / f"{base_config['experiment_id']}_pool_synergy_summary.csv"
        )
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSaved pool-synergy summary: {summary_path}")


if __name__ == "__main__":
    main()
