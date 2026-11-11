"""User-editable YAML settings and compatibility with frozen experiment records."""
import copy
import json
import math
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'config.yml'
METADATA = Path(__file__).with_name('metadata')
ARMS = {'yield': 'y_season_context_pairloss', 'soc': 's_monthly_width512'}
MODEL_SEEDS = {'yield': 20261009, 'soc': 20261008}


def read(path):
    return json.loads(Path(path).read_text())


def reference_recipe(target):
    """Original settings, independent of edits to the user's YAML file."""
    ref = read(METADATA / 'reference.json')
    training = copy.deepcopy(ref['training'][target])
    training['min_lr'] = 0.0 if target == 'yield' else 0.00002
    if target == 'soc':
        training['micro_batch'] = 64
    return {'training_seed': MODEL_SEEDS[target],
            'architecture': copy.deepcopy(ref['architecture'][target]),
            'training': training, 'unroll': 8,
            'loss': {'pair_weight': 0.5} if target == 'yield' else {'monthly_weight': 0.5, 'somsc_weight': 0.5}}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(f'{name}: missing keys {set(expected)-found}; unknown keys {found-set(expected)}')


def _number(value, name, *, integer=False, minimum=0):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"} >= {minimum}')


def validate(cfg):
    _keys(cfg, ['target', 'site_split_seed', 'device', 'verify_input_hashes', 'paths', 'models'], 'config')
    if not isinstance(cfg['target'], str) or cfg['target'] not in ARMS:
        raise ValueError('target must be yield or soc')
    _number(cfg['site_split_seed'], 'site_split_seed', integer=True)
    if not isinstance(cfg['device'], str) or not cfg['device']:
        raise ValueError('device must be a nonempty string')
    if type(cfg['verify_input_hashes']) is not bool:
        raise ValueError('verify_input_hashes must be true or false')
    _keys(cfg['paths'], ['input_root', 'run_dir'] + (['raw_root'] if 'raw_root' in cfg['paths'] else []), 'paths')
    for name, value in cfg['paths'].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'paths.{name} must be a nonempty path')
    profiles = cfg['models']
    if not isinstance(profiles, dict) or not profiles or set(profiles)-set(ARMS):
        raise ValueError('models must contain yield and/or soc settings')
    if cfg['target'] not in profiles:
        raise ValueError(f"models must contain settings for target {cfg['target']}")
    for target, model in cfg['models'].items():
        prefix = f'models.{target}'
        _keys(model, ['training_seed', 'architecture', 'training', 'loss'] + (['unroll'] if target == 'soc' else []), prefix)
        _number(model['training_seed'], prefix+'.training_seed', integer=True)
        if model['training_seed'] >= 2**32:
            raise ValueError(prefix+'.training_seed must be < 2**32')
        _keys(model['architecture'], ['hidden_dim', 'latent_dim', 'lstm_layers'], prefix+'.architecture')
        for name, value in model['architecture'].items():
            _number(value, prefix+'.architecture.'+name, integer=True, minimum=1)
        training = model['training']
        batches = ['batch_size'] if target == 'yield' else ['effective_batch', 'micro_batch']
        _keys(training, ['epochs', 'steps_per_epoch', 'lr', 'weight_decay', 'gradient_clip', 'min_lr']+batches, prefix+'.training')
        for name in ['epochs', 'steps_per_epoch']+batches:
            _number(training[name], prefix+'.training.'+name, integer=True, minimum=1)
        for name in ['lr', 'weight_decay', 'gradient_clip', 'min_lr']:
            _number(training[name], prefix+'.training.'+name)
        if training['lr'] <= 0 or training['gradient_clip'] <= 0 or training['min_lr'] > training['lr']:
            raise ValueError(prefix+': require lr > 0, gradient_clip > 0 and min_lr <= lr')
        if target == 'yield' and training['batch_size'] % 2:
            raise ValueError('Yield batch_size must be even for paired examples')
        if target == 'soc':
            if training['effective_batch'] % training['micro_batch']:
                raise ValueError('SOC micro_batch must divide effective_batch')
            _number(model['unroll'], prefix+'.unroll', integer=True, minimum=1)
            if 24 % model['unroll']:
                raise ValueError('SOC unroll must divide the 24-year training trajectory')
        _keys(model['loss'], ['pair_weight'] if target == 'yield' else ['monthly_weight', 'somsc_weight'], prefix+'.loss')
        for name, value in model['loss'].items():
            _number(value, prefix+'.loss.'+name)
    return cfg


def load_config(path=DEFAULT_CONFIG, *, target=None, site_seed=None, run_dir=None, input_root=None, device=None):
    path = Path(path).expanduser().resolve()
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f'Invalid YAML in {path}: {e}') from e
    validate(cfg)
    for name, value in [('target', target), ('site_split_seed', site_seed), ('device', device)]:
        if value is not None:
            cfg[name] = value
    for name, value in [('run_dir', run_dir), ('input_root', input_root)]:
        if value is not None:
            # Command-line paths have their usual working-directory semantics.
            cfg['paths'][name] = str(Path(value).expanduser().resolve())
    validate(cfg)
    for name, value in cfg['paths'].items():
        if name == 'run_dir':
            try:
                value = value.format(target=cfg['target'], site_split_seed=cfg['site_split_seed'])
            except (KeyError, ValueError, IndexError) as e:
                raise ValueError('run_dir supports only {target} and {site_split_seed}') from e
        value = Path(value).expanduser()
        cfg['paths'][name] = str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    return cfg


def recipe(cfg, target=None):
    target = target or cfg['target']
    result = reference_recipe(target)
    user = cfg['models'][target]
    result['training_seed'] = user['training_seed']
    result['architecture'].update(user['architecture'])
    result['training'].update(user['training'], seed=user['training_seed'])
    result['loss'] = copy.deepcopy(user['loss'])
    result['unroll'] = user.get('unroll', 8)
    return result


def apply_recipe(plan, target, settings):
    plan = copy.deepcopy(plan)
    plan.update(seed=settings['training_seed'], model_training_seed=settings['training_seed'],
                architecture=copy.deepcopy(settings['architecture']), training=copy.deepcopy(settings['training']),
                loss=copy.deepcopy(settings['loss']),
                arms={ARMS[target]: {'coverage': 'expanded', 'unroll': settings['unroll']}})
    return plan


def effective_plan(plan, target):
    # Earlier data-preparation plans predate the final model overrides. Preserve
    # those overrides in memory without modifying their recorded files/hashes.
    return plan if plan.get('config_version') == 1 else apply_recipe(plan, target, reference_recipe(target))


def write_config(path, cfg):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    temp.replace(path)


def check_run_config(root, cfg):
    record = read(Path(root)/'run.json')
    target = cfg['target']
    if record['target'] != target or record['site_split_seed'] != cfg['site_split_seed']:
        raise ValueError('Configuration target/site split differs from this run; use a new run_dir')
    plan = effective_plan(read(Path(root)/'data/plan.json'), target)
    requested = apply_recipe(plan, target, recipe(cfg))
    for key in ['seed', 'architecture', 'training', 'loss', 'arms']:
        if plan[key] != requested[key]:
            raise ValueError(f'Configuration {key} differs from the prepared run; use a new run_dir')
