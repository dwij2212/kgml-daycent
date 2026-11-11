"""Recompute the final trajectory metrics and heatmaps from saved predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .config import DEFAULT_CONFIG, METADATA, check_run_config, load_config, read
from .integrity import writable_output


def trajectory_scores(truth, prediction, minimum=1):
    valid = np.isfinite(truth) & np.isfinite(prediction)
    n = valid.sum(-1)
    total = np.where(valid, truth, 0).sum(-1)
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    sse = np.where(valid, (prediction-truth)**2, 0).sum(-1)
    sst = np.where(valid, (truth-mean[..., None])**2, 0).sum(-1)
    r2 = np.full(n.shape, np.nan, dtype=np.float64)
    np.divide(sse, sst, out=r2, where=(n >= minimum) & (sst > 1e-12))
    return n, 1-r2, sse


def collect(root, predictions):
    record, plan = read(root / 'run.json'), read(root / 'data/plan.json')
    sites = np.asarray(plan['test_sites'])
    if len(list(predictions.glob('*.npz'))) != 198:
        raise ValueError('Expected all 198 scenario prediction files.')
    rows = []
    for c, sc in enumerate(plan['scenarios']):
        if 'holdout' not in sc['group']:
            continue
        name = f'{sc["family"]}_{sc["id"]:03d}'
        with np.load(predictions / f'{name}.npz') as z:
            if record['target'] == 'soc':
                assert np.array_equal(z['site'], sites)
                assert np.array_equal(z['years'], np.arange(2000, 2024))
                truth = z['true'][:, :, :3].sum(-1, dtype=np.float64).T
                prediction = z['closed_loop'][:, :, :3].sum(-1, dtype=np.float64).T
                n, r2, sse = trajectory_scores(truth, prediction)
                rows.append(pd.DataFrame({'site': sites, 'scenario': name, 'n': n,
                    'r2': r2, 'rmse': np.sqrt(sse/24), 'final_stock_error': prediction[:, -1]-truth[:, -1]}))
            else:
                assert set(np.unique(z['point_id'])) == set(sites)
                si, yi = np.searchsorted(sites, z['point_id']), z['year']-2001
                assert (yi >= 0).all() and (yi < 23).all()
                for k, crop in enumerate(['corn', 'soybean', 'winter_wheat']):
                    truth = np.full((len(sites), 23), np.nan)
                    prediction = truth.copy()
                    take = z['crop'] == k
                    truth[si[take], yi[take]] = z['yield_true'][take]
                    prediction[si[take], yi[take]] = z['yield_pred'][take]
                    n, r2, sse = trajectory_scores(truth, prediction, minimum=3)
                    rows.append(pd.DataFrame({'site': sites, 'scenario': name, 'crop': crop, 'n': n, 'r2': r2}))
    frame = pd.concat(rows, ignore_index=True)
    summary = {'target': record['target'], 'site_split_seed': record['site_split_seed'],
               'model_training_seed': record['model_training_seed'], 'test_sites': len(sites),
               'reserved_scenarios': 22, 'evaluation_files': 198}
    if record['target'] == 'soc':
        summary.update(median_trajectory_r2=float(frame.r2.median()), negative_cells=int((frame.r2 < 0).sum()),
                       final_stock_rmse=float(np.sqrt(np.mean(frame.final_stock_error**2))))
    else:
        summary['crops'] = {crop: {'median_trajectory_r2': float(g.r2.median()),
                                  'scored_cells': int(g.r2.notna().sum()), 'negative_cells': int((g.r2 < 0).sum())}
                            for crop, g in frame.groupby('crop')}
    return frame, summary


def report(run_dir, prediction_dir=None, output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(run_dir)
    frame, summary = collect(root, Path(prediction_dir or root / 'predictions'))
    out = writable_output(output_dir or root / 'report')
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'trajectory_metrics.csv', index=False)
    write_json(out / 'metrics.json', summary)
    scenarios = sorted(frame.scenario.unique(), key=lambda s: (not s.startswith('synthetic'), s))
    geography = pd.DataFrame(read(METADATA / 'site_geography.json'))
    geography = geography[geography.id.isin(frame.site.unique())].copy()
    states = ['MN', 'IA', 'WI', 'IL', 'IN', 'OH']
    geography['state_order'] = geography.STATE.map({s: i for i, s in enumerate(states)})
    geography = geography.sort_values(['state_order', 'POINT_X', 'POINT_Y', 'id'])
    panels = [(crop, frame[frame.crop == crop]) for crop in ['corn', 'soybean', 'winter_wheat']] if summary['target'] == 'yield' else [('SOC', frame)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 5*len(panels)), squeeze=False, layout='constrained')
    cmap = plt.get_cmap('Greens').copy()
    cmap.set_under('#e31a1c')
    cmap.set_bad('#cccccc')
    for ax, (label, data) in zip(axes[:, 0], panels):
        values = data.pivot(index='scenario', columns='site', values='r2').reindex(index=scenarios, columns=geography.id)
        im = ax.imshow(values, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(scenarios)), [s.replace('synthetic_', 'S').replace('realistic_', 'R') for s in scenarios])
        positions = [(state, np.flatnonzero(geography.STATE.to_numpy() == state)) for state in states]
        ax.set_xticks([p.mean() for s, p in positions if len(p)], [s for s, p in positions if len(p)])
        ax.set(xlabel='Locations (grouped by state)', ylabel='Scenario',
               title=f'{label} · median trajectory R² = {data.r2.median():.3f} · site split {summary["site_split_seed"]}')
        fig.colorbar(im, ax=ax, extend='min', label='R² (red < 0)')
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'{summary["target"]}_heatmap.{suffix}', dpi=200)
    plt.close(fig)
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.config is not None or args.run_dir is None:
        try:
            settings = load_config(args.config or DEFAULT_CONFIG, run_dir=args.run_dir)
            args.run_dir = Path(settings['paths']['run_dir'])
            check_run_config(args.run_dir, settings)
        except (ValueError, OSError) as e:
            parser.error(str(e))
    report(args.run_dir, args.predictions, args.output_dir)


if __name__ == '__main__':
    main()
