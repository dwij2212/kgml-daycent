"""Predict yield and December SOC from new input files, without PBM targets."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import numpy as np
import torch

from inputs import CROPS, Inputs, load_stats
from yield_model import YieldModel
from soc_model import SOCModel

ROOT = Path(__file__).resolve().parent
YIELD_HEADER = ['scenario_id', 'id', 'Year', 'doy', 'crop', 'cgrain']
SOC_HEADER = ['scenario_id', 'id', 'Year', 'month', 'somsc', 'delta_somsc',
              'som1c_soil', 'som2c_soil', 'som3c', 'som2c_surface']


def load_model(target, settings, device):
    checkpoint = torch.load(ROOT/'weights'/f'{target}.pt', map_location='cpu', weights_only=False)
    cls = YieldModel if target == 'yield' else SOCModel
    model = cls(**settings['architecture'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), load_stats(settings['stats'])


def tensor_batch(features, device):
    return {key: torch.from_numpy(np.stack([row[key] for row in features])).to(device)
            for key in features[0]}


@torch.inference_mode()
def yield_rows(data, model, stats, device, batch_size, yield_start_year):
    metadata, features = [], []

    def predict_batch():
        scaled = model(tensor_batch(features, device))['yield_pred'].cpu().numpy()
        crop = np.array([row[4] for row in metadata])
        values = scaled*stats['yield_crop_std'][crop]+stats['yield_crop_mean'][crop]
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite yield predictions')
        for row, value in zip(metadata, values):
            yield (*row[:4], CROPS[row[4]], float(value))

    for pair in data.pairs:
        for record, sample in data.yield_samples(pair, stats, yield_start_year):
            metadata.append(record); features.append(sample)
            if len(features) == batch_size:
                yield from predict_batch()
                metadata.clear(); features.clear()
    if features:
        yield from predict_batch()


@torch.inference_mode()
def soc_rows(data, model, stats, device, batch_size):
    mean = torch.as_tensor(stats['pool_mean'], device=device)
    std = torch.as_tensor(stats['pool_std'], device=device)
    for start in range(0, len(data.pairs), batch_size):
        pairs = data.pairs[start:start+batch_size]
        features = []
        initial = []
        for pair in pairs:
            sequence, static = data.features(pair, stats)
            features.append({'sequence': sequence, 'init_cond': static, 'year_enc': data.year_enc})
            initial.append(data.initial_pools(pair[1]))
        batch = tensor_batch(features, device)
        state = torch.from_numpy(np.stack(initial)).to(device)
        previous_stock = state.cpu().numpy()[:, :3].sum(-1, dtype=np.float64)
        for yi, year in enumerate(data.years):
            inputs = {'sequence': batch['sequence'][:, yi], 'init_cond': batch['init_cond'],
                      'year_enc': batch['year_enc'][:, yi], 'prev_soc_state_raw': state,
                      'prev_soc_state_norm': (state-mean)/std}
            state = model(inputs)['soc_state_pred']
            if not torch.isfinite(state).all():
                raise FloatingPointError('Non-finite SOC predictions')
            pools = state.cpu().numpy()
            stock = pools[:, :3].sum(-1, dtype=np.float64)
            for i, pair in enumerate(pairs):
                yield (pair[0], pair[1], int(year), 12, float(stock[i]),
                       float(stock[i]-previous_stock[i]), *map(float, pools[i]))
            previous_stock = stock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['yield', 'soc', 'both'], default='both')
    parser.add_argument('--weather-dir', type=Path, required=True, help='CSV files named <id>.csv')
    parser.add_argument('--site-conditions', type=Path, required=True, help='Initial site CSV or XLSX workbook')
    parser.add_argument('--management', type=Path, nargs='+', required=True, help='One or more wide or long management CSVs')
    parser.add_argument('--first-year', type=int, required=True, help='First full input year; initial SOC pools apply at its start')
    parser.add_argument('--last-year', type=int, required=True, help='Last full input year, inclusive')
    parser.add_argument('--yield-start-year', type=int, help='First yield output year; defaults to first-year + 1')
    parser.add_argument('--output', type=Path, required=True, help='Output directory; existing prediction CSVs are never overwritten')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0 (etc.)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    temporary = []
    try:
        if args.batch_size < 1 or args.cpu_threads < 1:
            raise ValueError('batch-size and cpu-threads must be positive')
        device = torch.device(args.device)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable; use --device cpu')
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        elif device.type != 'cpu':
            raise ValueError('Use cpu or a CUDA device')
        torch.set_num_threads(args.cpu_threads)
        targets = ['yield', 'soc'] if args.target == 'both' else [args.target]
        yield_start = args.yield_start_year if args.yield_start_year is not None else args.first_year+1
        if 'yield' in targets and not args.first_year < yield_start <= args.last_year:
            raise ValueError('Yield requires first-year < yield-start-year <= last-year; supply a prior year of drivers')
        settings = json.loads((ROOT/'preprocessing.json').read_text())
        data = Inputs(args.weather_dir, args.site_conditions, args.management,
                      settings['static_columns'], args.first_year, args.last_year)
        outputs = {target: args.output/f'{target}_predictions.csv' for target in targets}
        for path in outputs.values():
            if path.exists():
                raise ValueError(f'Output already exists; use a new directory: {path}')
        args.output.mkdir(parents=True, exist_ok=True)
        print(f'{len(data.pairs)} site/scenario pairs; {args.first_year}–{args.last_year}; device={device}', flush=True)
        with ExitStack() as stack:
            for target in targets:
                model, stats = load_model(target, settings['models'][target], device)
                stream = stack.enter_context(tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', newline='', dir=args.output,
                    prefix=f'.{target}_', suffix='.tmp', delete=False))
                temporary.append((Path(stream.name), outputs[target]))
                writer = csv.writer(stream)
                writer.writerow(YIELD_HEADER if target == 'yield' else SOC_HEADER)
                iterator = (yield_rows(data, model, stats, device, args.batch_size, yield_start)
                            if target == 'yield' else soc_rows(data, model, stats, device, args.batch_size))
                count = 0
                for row in iterator:
                    writer.writerow(row); count += 1
                    if count % 5000 == 0:
                        print(f'{target}: {count:,} rows', flush=True)
                stream.flush()
                print(f'{target}: {count:,} predictions complete', flush=True)
                del model
        for source, dest in temporary:
            source.replace(dest)
            print(dest.resolve(), flush=True)
    except (ValueError, OSError, KeyError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
