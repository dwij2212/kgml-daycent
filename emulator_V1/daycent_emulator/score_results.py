"""Unambiguous pooled, per-site, and trajectory R² on the 22 reserved scenarios.

Only saved predictions are read. Both SOC modes are reported; the absolute
December stock is the sum of the three soil pools, excluding the surface pool.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .config import read
from .integrity import writable_output
from .report import trajectory_scores
from .yield_metrics import metrics

CROPS = ['corn', 'soybean', 'winter_wheat']


def yield_scores(root, plan, output):
    frames = []
    for c, sc in enumerate(plan['scenarios']):
        if 'holdout' not in sc['group']:
            continue
        with np.load(root / 'predictions' / f'{sc["key"]}.npz', allow_pickle=False) as z:
            frame = pd.DataFrame(dict(z))
        assert set(frame.point_id.unique()) == set(plan['test_sites'])
        assert (frame.scenario_index == c).all()
        assert frame.year.between(2001, 2023).all()
        assert np.isfinite(frame[['yield_true', 'yield_pred']].to_numpy()).all()
        assert not frame.duplicated(['point_id', 'year']).any()
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    rows = []
    for (site, crop), group in frame.groupby(['point_id', 'crop']):
        rows.append({'site': int(site), 'crop': CROPS[crop],
                     **metrics(group.yield_true, group.yield_pred)})
    pd.DataFrame(rows).to_csv(output / 'site_metrics.csv', index=False)
    trajectories = pd.read_csv(output / 'trajectory_metrics.csv')
    return {'years': [2001, 2023], 'overall': metrics(frame.yield_true, frame.yield_pred),
            'per_crop': {CROPS[k]: metrics(g.yield_true, g.yield_pred)
                         for k, g in frame.groupby('crop')},
            'median_site_r2': {crop: float(g.r2.median())
                               for crop, g in pd.DataFrame(rows).groupby('crop')},
            'median_trajectory_r2': {crop: float(g.r2.median())
                                     for crop, g in trajectories.groupby('crop')}}


def soc_scores(root, plan, output):
    truths, modes = [], {'closed_loop': [], 'teacher_forced': []}
    names = []
    for sc in plan['scenarios']:
        if 'holdout' not in sc['group']:
            continue
        with np.load(root / 'predictions' / f'{sc["key"]}.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['site'], plan['test_sites'])
            np.testing.assert_array_equal(z['years'], np.arange(2000, 2024))
            assert z['true'].shape == (24, len(plan['test_sites']), 4)
            truths.append(z['true'][:, :, :3].sum(-1, dtype=np.float64))
            for name in modes:
                assert z[name].shape == z['true'].shape
                modes[name].append(z[name][:, :, :3].sum(-1, dtype=np.float64))
        names.append(sc['key'])
    true = np.stack(truths)  # scenario, year, site
    assert true.shape == (22, 24, len(plan['test_sites']))
    results, site_rows, trajectory_rows = {}, [], []
    for mode, parts in modes.items():
        pred = np.stack(parts)
        assert np.isfinite(true).all() and np.isfinite(pred).all()
        label = 'autoregressive_rollout' if mode == 'closed_loop' else mode
        _, trajectory_r2, _ = trajectory_scores(true.transpose(0, 2, 1), pred.transpose(0, 2, 1))
        site_true = true.transpose(2, 0, 1).reshape(len(plan['test_sites']), -1)
        site_pred = pred.transpose(2, 0, 1).reshape(len(plan['test_sites']), -1)
        _, site_r2, _ = trajectory_scores(site_true, site_pred)
        results[label] = {'overall': metrics(true.ravel(), pred.ravel()),
                          'median_site_r2': float(np.nanmedian(site_r2)),
                          'median_trajectory_r2': float(np.nanmedian(trajectory_r2)),
                          'final_stock_rmse_gC_m2': float(np.sqrt(np.mean((pred[:, -1]-true[:, -1])**2)))}
        for j, site in enumerate(plan['test_sites']):
            site_rows.append({'site': site, 'mode': label, 'r2': site_r2[j]})
            for c, scenario in enumerate(names):
                trajectory_rows.append({'site': site, 'scenario': scenario,
                                        'mode': label, 'r2': trajectory_r2[c, j]})
    pd.DataFrame(site_rows).to_csv(output / 'site_metrics.csv', index=False)
    pd.DataFrame(trajectory_rows).to_csv(output / 'trajectory_metrics_both_modes.csv', index=False)
    return {'years': [2000, 2023], 'stock': 'absolute December SOC; first three soil pools',
            'modes': results}


def score_run(root):
    root = Path(root)
    record, plan = read(root / 'run.json'), read(root / 'data/plan.json')
    output = writable_output(root / 'report')
    output.mkdir(parents=True, exist_ok=True)
    fn = yield_scores if record['target'] == 'yield' else soc_scores
    result = {'target': record['target'], 'site_split_seed': record['site_split_seed'],
              'model_training_seed': record['model_training_seed'],
              'test_sites': len(plan['test_sites']), 'reserved_scenarios': 22,
              'r2_definition': '1 - sum((prediction-truth)^2) / sum((truth-mean(truth))^2)',
              'site_definition': 'one site (and one crop for yield), pooling reserved scenarios and years',
              'trajectory_definition': 'one site and scenario (and crop for yield), across years',
              **fn(root, plan, output)}
    write_json(output / 'full_period_metrics.json', result)
    return result
