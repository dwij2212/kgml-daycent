"""Train a final yield or SOC recipe from random weights on a prepared split."""
import argparse
from pathlib import Path

from .common import write_json
from .config import (DEFAULT_CONFIG, METADATA, MODEL_SEEDS, check_run_config,
                     effective_plan, load_config, read)
from .prepare import prepare
from .runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Defaults to config.yml when --run-dir is omitted')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    settings = None
    try:
        if args.config is not None or args.run_dir is None:
            settings = load_config(args.config or DEFAULT_CONFIG, run_dir=args.run_dir,
                                   input_root=args.input_root, device=args.device)
            args.run_dir = Path(settings['paths']['run_dir'])
            args.input_root = Path(settings['paths']['input_root'])
            args.device = settings['device']
        record = read(args.run_dir / 'run.json') if (args.run_dir / 'run.json').exists() else None
        if record is not None and settings is not None:
            check_run_config(args.run_dir, settings)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    if record is None and settings is None:
        parser.error('Prepare this run first or use --config config.yml to prepare and train it')
    if record and record.get('frozen'):
        parser.error('This is a frozen final checkpoint. Prepare a new run directory to retrain.')
    args.device = args.device or 'cuda:0'
    import torch
    import numpy as np
    expected = read(METADATA / 'reference.json')
    if str(torch.__version__) != expected['reference_runtime']['torch']:
        parser.error(f"Exact recipe requires Torch {expected['reference_runtime']['torch']}; found {torch.__version__}.")
    if str(np.__version__) != expected['reference_runtime']['numpy']:
        parser.error(f"Exact recipe requires NumPy {expected['reference_runtime']['numpy']}; found {np.__version__}.")
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('The original training recipe requires a CUDA GPU.')
    if record is None:
        prepare(settings['target'], args.input_root, args.run_dir, settings['site_split_seed'],
                settings['verify_input_hashes'], settings)
    workflow, record = configure(args.run_dir, args.input_root)
    arm, target = record['arm'], record['target']
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    plan = effective_plan(read(args.run_dir / 'data/plan.json'), target)
    reference_initialization = (record['model_training_seed'] == MODEL_SEEDS[target]
                                and plan['architecture'] == expected['architecture'][target])
    if target == 'yield':
        workflow.check(arm, device)
        if reference_initialization:
            assert read(args.run_dir / 'probe.json')['initial_hash'] == expected['initial_hash']['yield']
        if not args.probe_only:
            workflow.train(arm, device)
    else:
        device = workflow.base.runtime(args.device, record['model_training_seed'])
        workflow.probe(arm, device)
        probe = read(args.run_dir / 'probe.json')
        assert probe['micro_batch'] == plan['training']['micro_batch']
        if reference_initialization:
            assert probe['initialization']['initial_weight_sha256'] == expected['initial_hash']['soc']
        if not args.probe_only:
            workflow.base.train(arm, device)
    write_json(args.run_dir / 'execution.json', {'target': target, 'site_split_seed': record['site_split_seed'],
               'model_training_seed': record['model_training_seed'], 'torch': str(torch.__version__),
               'numpy': str(np.__version__), 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version(), 'tf32': True, 'cudnn_benchmark': True,
               'gpu': torch.cuda.get_device_name(device), 'probe_only': args.probe_only})


if __name__ == '__main__':
    main()
