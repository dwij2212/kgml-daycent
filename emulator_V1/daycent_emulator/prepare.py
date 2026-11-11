"""Prepare the final recipe with an independently selectable site-split seed."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np

from .common import sha256, write_json
from .config import (ARMS, DEFAULT_CONFIG, METADATA, apply_recipe, load_config,
                     read, recipe, write_config)
from .normalization import fit_soc_stats, fit_yield_stats
from .integrity import writable_output

def make_plan(target, site_seed, settings=None):
    settings = settings or load_config(target=target, site_seed=site_seed)
    reference = read(METADATA / 'reference.json')
    names = ['sites', 'selection_scenario_indices', 'scenarios', 'scenario_groups',
             'years', 'synthetic_scenario_holdout', 'realistic_scenario_holdout']
    plan = {k: copy.deepcopy(reference[k]) for k in names}
    sites = np.asarray(plan['sites'])
    order = np.random.default_rng(site_seed).permutation(len(sites))
    n = int(.1 * len(sites))
    plan.update(train_sites=sorted(sites[order[2*n:]].tolist()),
                validation_sites=sorted(sites[order[n:2*n]].tolist()),
                test_sites=sorted(sites[order[:n]].tolist()))
    plan['selection_sites'] = np.asarray(plan['validation_sites'])[reference['selection_ranks']].tolist()
    plan = apply_recipe(plan, target, recipe(settings, target))
    plan.update(config_version=1, site_split_seed=site_seed, reference_site_split_seed=20260914,
                years_evaluated=list(range(2001 if target == 'yield' else 2000, 2024)),
                selection='128 validation sites; 18 allowed schedules; original balanced validation loss.',
                normalization='Original recipe fitted to this split\'s training sites only.')
    train, val, test = (set(plan[k]) for k in ['train_sites', 'validation_sites', 'test_sites'])
    assert (len(train), len(val), len(test)) == (7805, 975, 975)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(sites)
    assert set(plan['selection_sites']) <= val
    if site_seed == 20260914:
        for k in ['train_sites', 'validation_sites', 'test_sites', 'selection_sites']:
            assert plan[k] == reference[k], k
    return plan


def verify_inputs(base, target, hashes=True):
    manifest = read(METADATA / 'input_manifest.json')
    for i, rel in enumerate(manifest['targets'][target], 1):
        path = base / rel
        expected = manifest['files'][rel]
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing or wrong-size input: {path}')
        if hashes and sha256(path) != expected['sha256']:
            raise ValueError(f'Input checksum mismatch: {path}')
        if i % 100 == 0:
            print(f'Checked {i} input files', flush=True)
    return len(manifest['targets'][target])


def prepare(target, base, root, site_seed, check_hashes=True, settings=None):
    base, root = Path(base).resolve(), writable_output(root)
    settings = copy.deepcopy(settings or load_config(target=target, site_seed=site_seed))
    settings.update(target=target, site_split_seed=site_seed, verify_input_hashes=check_hashes)
    settings['paths'].update(input_root=str(base), run_dir=str(root))
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Use a new or empty run directory: {root}')
    count = verify_inputs(base, target, check_hashes)
    plan = make_plan(target, site_seed, settings)
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    write_config(root / 'config.yml', settings)
    write_json(root / 'plan.json', plan)
    write_json(data / 'plan.json', plan)
    if target == 'yield':
        stats, buckets = fit_yield_stats(base, plan, data)
    else:
        stats, buckets = fit_soc_stats(base, plan), {}
    if site_seed == 20260914:
        reference = read(METADATA / 'reference.json')['reference_stats'][target]
        for k in reference:
            if isinstance(reference[k], str):
                assert stats[k] == reference[k]
            else:
                np.testing.assert_allclose(stats[k], reference[k], rtol=1e-12, atol=1e-8, err_msg=k)
        stats = reference
    write_json(data / 'stats.json', stats)
    write_json(root / 'stats.json', stats)
    write_json(data / 'normalization.json', {'state': 'complete', 'fitted_fresh': True,
               'training_sites': 7805, 'test_or_validation_sites_used': False,
               'heldout_scenarios_used': False, 'stats_sha256': sha256(data / 'stats.json')})
    write_json(data / 'verification.json', {'passed': True, 'input_files': count,
               'input_hashes_checked': check_hashes, 'buckets': buckets,
               'stats_sha256': sha256(data / 'stats.json'), 'plan_sha256': sha256(data / 'plan.json')})
    write_json(root / 'run.json', {'target': target, 'arm': ARMS[target],
               'site_split_seed': site_seed, 'model_training_seed': plan['seed'],
               'input_root': str(base), 'frozen': False, 'created_epoch': time.time(),
               'config_sha256': sha256(root / 'config.yml'),
               'source_sha256': {str(p.relative_to(METADATA.parent)): sha256(p)
                                 for p in sorted(METADATA.parent.rglob('*'))
                                 if p.is_file() and p.suffix in {'.py', '.json'}},
               'data_plan_sha256': sha256(data / 'plan.json'), 'stats_sha256': sha256(data / 'stats.json')})
    print(f'Prepared {target}: site seed {site_seed}, model seed {plan["seed"]}, run {root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--target', choices=ARMS)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--site-split-seed', type=int)
    parser.add_argument('--skip-input-hashes', action='store_true', help='Only for a cache whose manifest has already been verified; file sizes are still checked.')
    args = parser.parse_args()
    try:
        settings = load_config(args.config, target=args.target, site_seed=args.site_split_seed,
                               input_root=args.input_root, run_dir=args.run_dir)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    prepare(settings['target'], settings['paths']['input_root'], settings['paths']['run_dir'],
            settings['site_split_seed'], settings['verify_input_hashes'] and not args.skip_input_hashes, settings)


if __name__ == '__main__':
    main()
