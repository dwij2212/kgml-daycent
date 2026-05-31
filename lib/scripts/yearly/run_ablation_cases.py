"""
Generate and optionally run yearly SOMSC ablation configs.

Examples:
    python scripts/yearly/run_ablation_cases.py --dry-run
    python scripts/yearly/run_ablation_cases.py --train
    python scripts/yearly/run_ablation_cases.py --eval --eval-mode both --save-csv
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml


LIB_DIR = Path(__file__).resolve().parents[2]
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from eval_yearly_somsc import evaluate_experiment
from train_yearly_somsc import train
from utils.config import ExperimentConfig


SOIL_POOLS = ["som1c_soil", "som2c_soil", "som3c"]
FULL_POOLS = SOIL_POOLS + ["som2c_surface"]


ABLATION_CASES = {
    "scalar_prev_somsc": {
        "description": "Scalar annual SOMSC-delta target with previous aggregate SOMSC context.",
        "model": {
            "target_mode": "somsc_delta",
            "prev_state_context": "somsc",
            "target_pool_cols": FULL_POOLS,
        },
        "supports_rollout": True,
    },
    "scalar_full_pools": {
        "description": "Scalar annual SOMSC-delta target with previous full SOC pool context.",
        "model": {
            "target_mode": "somsc_delta",
            "prev_state_context": "full_pools",
            "target_pool_cols": FULL_POOLS,
        },
        "supports_rollout": False,
    },
    "three_pool_no_aux": {
        "description": "Three soil-pool annual delta target without the surface auxiliary pool.",
        "model": {
            "target_mode": "pool_deltas",
            "prev_state_context": "target_pools",
            "target_pool_cols": SOIL_POOLS,
        },
        "supports_rollout": True,
    },
    "four_state_current": {
        "description": "Current four-state model with soil pools plus surface auxiliary pool.",
        "model": {
            "target_mode": "pool_deltas",
            "prev_state_context": "target_pools",
            "target_pool_cols": FULL_POOLS,
        },
        "supports_rollout": True,
    },
    "three_pool_somsc_context": {
        "description": "Three soil-pool annual delta target with only previous aggregate SOMSC context.",
        "model": {
            "target_mode": "pool_deltas",
            "prev_state_context": "somsc",
            "target_pool_cols": SOIL_POOLS,
        },
        "supports_rollout": True,
    },
}


def _load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, config_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)


def build_case_config(base_config: dict, case_name: str) -> dict:
    case = ABLATION_CASES[case_name]
    cfg = copy.deepcopy(base_config)
    base_experiment_id = base_config["experiment_id"]

    cfg["experiment_id"] = f"{base_experiment_id}_{case_name}"
    cfg["description"] = case["description"]
    cfg.setdefault("model", {})
    cfg["model"].update(case["model"])
    cfg["model"]["model_type"] = "yearly_somsc_state"

    cfg.setdefault("wandb", {})
    cfg["wandb"]["name"] = cfg["experiment_id"]
    cfg["wandb"]["notes"] = case["description"]
    tags = list(cfg["wandb"].get("tags", []))
    for tag in ["yearly_ablation", case_name]:
        if tag not in tags:
            tags.append(tag)
    cfg["wandb"]["tags"] = tags
    return cfg


def requested_eval_mode(case_name: str, eval_mode: str) -> str:
    if ABLATION_CASES[case_name]["supports_rollout"]:
        return eval_mode
    if eval_mode in {"rollout", "both"}:
        print(
            f"Case '{case_name}' uses full pool context but predicts only scalar SOMSC; "
            "using teacher_forced evaluation because rollout cannot propagate pools."
        )
    return "teacher_forced"


def main():
    parser = argparse.ArgumentParser(description="Run yearly SOMSC ablation cases")
    parser.add_argument(
        "--base-config",
        default="configs/yearly/experiment_state_v1.yaml",
        help="Base yearly config to clone for each ablation.",
    )
    parser.add_argument(
        "--out-dir",
        default="configs/yearly/ablations",
        help="Directory where generated ablation YAMLs are written.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(ABLATION_CASES),
        default=list(ABLATION_CASES),
        help="Subset of ablation cases to generate/run.",
    )
    parser.add_argument("--train", action="store_true", help="Train generated cases.")
    parser.add_argument("--eval", action="store_true", help="Evaluate generated cases.")
    parser.add_argument(
        "--eval-mode",
        choices=["teacher_forced", "rollout", "both"],
        default="both",
        help="Evaluation mode for cases that support rollout.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample trajectories to plot during evaluation.",
    )
    parser.add_argument("--save-csv", action="store_true", help="Save prediction CSVs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate configs and print the commands to run.",
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
    for case_name in args.cases:
        cfg = build_case_config(base_config, case_name)
        out_path = out_dir / f"{case_name}.yaml"
        _write_yaml(out_path, cfg)
        generated.append((case_name, out_path))

    print("Generated ablation configs:")
    for case_name, path in generated:
        print(f"  {case_name}: {path}")

    if args.dry_run or not args.train and not args.eval:
        print("\nSuggested commands:")
        for case_name, path in generated:
            print(f"  python train_yearly_somsc.py --config {path}")
            mode = requested_eval_mode(case_name, args.eval_mode)
            save_csv = " --save-csv" if args.save_csv else ""
            print(
                "  python eval_yearly_somsc.py "
                f"--config {path} --split {args.split} --eval-mode {mode}{save_csv}"
            )
        return

    for case_name, path in generated:
        config = ExperimentConfig.from_yaml(str(path))
        if args.train:
            train(config)
        if args.eval:
            evaluate_experiment(
                config,
                split=args.split,
                num_samples=args.num_samples,
                save_csv=args.save_csv,
                eval_mode=requested_eval_mode(case_name, args.eval_mode),
            )


if __name__ == "__main__":
    main()
