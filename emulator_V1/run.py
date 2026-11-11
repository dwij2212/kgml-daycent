"""Prepare data, train, and evaluate the final yield and SOC models. See README.md."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from daycent_emulator.integrity import RELEASE_ROOT, writable_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'results', 'prepare-data', 'evaluate', 'train'])
    parser.add_argument('--target', choices=['yield', 'soc'])
    parser.add_argument('--site-split-seed', type=int,
                        help='Evaluate: selected checkpoint seed; train/verify: config default')
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--raw-root', type=Path, help='Original KGML_Midwest_Daily_2026 directory')
    parser.add_argument('--workers', type=int, default=2, help='Raw-data preparation workers')
    parser.add_argument('--output', type=Path, help='New output directory outside the release')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        if args.command == 'results':
            print((RELEASE_ROOT / 'results/summary.csv').read_text(), end='')
            return
        from daycent_emulator.config import load_config
        settings = load_config(target=args.target, input_root=args.input_root)
        if args.command == 'prepare-data':
            from daycent_emulator.raw_data import RawBuilder
            raw = args.raw_root or Path(settings['paths']['raw_root'])
            output = args.output or Path(settings['paths']['input_root'])
            if args.workers < 1:
                parser.error('--workers must be positive')
            RawBuilder(raw, output).build(args.workers)
            return
        if args.target is None:
            parser.error('--target is required for evaluate, train, or verify (prepared inputs)')
        checkpoint = None
        site_seed = args.site_split_seed
        if args.command == 'evaluate':
            inference = RELEASE_ROOT / 'emulator_inference'
            selected = json.loads((inference / 'preprocessing.json').read_text())['models'][args.target]
            if site_seed is None:
                site_seed = selected['site_split_seed']
            if site_seed != selected['site_split_seed']:
                parser.error(f'Only the selected {args.target} checkpoint (site seed '
                             f'{selected["site_split_seed"]}) is retained. Other splits keep '
                             'their configurations and results; retrain them to create new weights.')
            if selected['source_run'] != f'{args.target}_split{site_seed}':
                parser.error('Selected checkpoint and source-run metadata disagree.')
            checkpoint = inference / 'weights' / f'{args.target}.pt'
            if not checkpoint.is_file():
                parser.error(f'Missing selected checkpoint: {checkpoint}')
        elif site_seed is None:
            site_seed = settings['site_split_seed']
        source = RELEASE_ROOT / 'runs' / f'{args.target}_split{site_seed}'
        if not source.is_dir():
            parser.error('Retained splits: yield 20260914/20260915/20260916; SOC 20260914/20260915.')
        base = Path(settings['paths']['input_root'])
        if args.command == 'verify':
            from daycent_emulator.prepare import verify_inputs
            n = verify_inputs(base, args.target, hashes=True)
            print(f'PASS: {n} prepared input files match the reference dataset.')
            return
        if args.output is None:
            parser.error('--output is required; saved results are never overwritten')
        output = writable_output(args.output)
        if output.exists():
            parser.error(f'Use a new output directory: {output}')
        if args.command == 'evaluate':
            from daycent_emulator.prepare import verify_inputs
            # Size checks on every execution; full dataset hash verification is
            # available explicitly with run.py verify --target yield/soc.
            verify_inputs(base, args.target, hashes=False)
            shutil.copytree(source, output)
            shutil.copy2(checkpoint, output / 'best.pt')
            sys.argv = ['evaluate', '--run-dir', str(output), '--input-root', str(base),
                        '--device', args.device]
            from daycent_emulator.evaluate import main as execute
        else:
            sys.argv = ['train', '--config', str(source / 'config.yml'),
                        '--run-dir', str(output), '--input-root', str(base), '--device', args.device]
            from daycent_emulator.train import main as execute
        execute()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
