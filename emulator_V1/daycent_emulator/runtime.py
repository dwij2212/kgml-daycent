"""Bind a prepared run to input caches without importing historical campaigns."""
import os
from pathlib import Path

from .common import sha256
from .config import ARMS, effective_plan, read
from .integrity import writable_output


def link_input(path, target):
    if not target.exists():
        raise FileNotFoundError(target)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    if path.exists():
        raise ValueError(f'Refusing to replace a regular file/directory: {path}')
    path.symlink_to(target, target_is_directory=target.is_dir())


def configure(run_dir, input_root=None):
    root = writable_output(run_dir)
    record = read(root / 'run.json')
    target, arm = record['target'], record['arm']
    assert arm == ARMS[target]
    data = root / 'data'
    if 'config_sha256' in record and sha256(root / 'config.yml') != record['config_sha256']:
        raise ValueError('Saved run configuration changed; prepare a new run for different settings')
    for name, field in [('plan.json', 'data_plan_sha256'), ('stats.json', 'stats_sha256')]:
        if sha256(data / name) != record[field]:
            raise ValueError(f'Prepared run changed: {data / name}')
    plan = effective_plan(read(data / 'plan.json'), target)
    assert plan['seed'] == record['model_training_seed']
    base = Path(input_root or os.environ.get('KGML_INPUT_ROOT') or record['input_root']).resolve()
    if target == 'yield':
        from . import yield_workflow as workflow
        for name in ['target.npy', 'crop.npy', 'harvest_day.npy']:
            link_input(data / name, base / 'yield_improvement_v1' / name)
        workflow.ROOT = root
        workflow.yd.WORK = data
        workflow.yd.DRIVER_WORK = base / 'soc_improvement_v1'
        workflow.shared.WORK = root
    else:
        from . import soc_data
        from . import soc_workflow as workflow
        for name in ['cache', 'targets']:
            link_input(data / name, base / 'soc_improvement_v1' / name)
        link_input(data / 'monthly_targets', base / 'soc_followup_v1/monthly_targets')
        soc_data.BASE = soc_data.WORK = data
        workflow.FOLLOWUP, workflow.ROOT = data, root
        workflow.setup()
    return workflow, record
