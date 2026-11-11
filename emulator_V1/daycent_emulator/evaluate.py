"""Evaluate a final checkpoint on its test sites and all 198 scenarios."""
import argparse
from pathlib import Path

from .runtime import configure
from .common import sha256
from .config import DEFAULT_CONFIG, check_run_config, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Defaults to config.yml when --run-dir is omitted')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--device')
    args = parser.parse_args()
    try:
        if args.config is not None or args.run_dir is None:
            settings = load_config(args.config or DEFAULT_CONFIG, run_dir=args.run_dir,
                                   input_root=args.input_root, device=args.device)
            args.run_dir = Path(settings['paths']['run_dir'])
            args.input_root = Path(settings['paths']['input_root'])
            args.device = settings['device']
            check_run_config(args.run_dir, settings)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    args.device = args.device or 'cuda:0'
    if not (args.run_dir / 'best.pt').is_file():
        parser.error('No best.pt checkpoint in this run directory.')
    import torch
    workflow, record = configure(args.run_dir, args.input_root)
    if record.get('frozen') and sha256(args.run_dir / 'best.pt') != record['checkpoint_sha256']:
        parser.error('The frozen checkpoint checksum does not match its run record.')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if record['target'] == 'yield':
        workflow.evaluate(record['arm'], device)
    else:
        workflow.base.evaluate(record['arm'], device)
    from .report import report
    report(args.run_dir)
    from .score_results import score_run
    score_run(args.run_dir)


if __name__ == '__main__':
    main()
