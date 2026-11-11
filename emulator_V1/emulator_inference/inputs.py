"""Input-only features for the retained yield and SOC checkpoints.

Reads weather, initial site conditions and management; no PBM targets, training
plans, site splits or prepared experiment caches are used.
"""
import calendar
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

EVENT_TYPES = [
    'conventional_till_molboadplow', 'corn_planting', 'cycle_end',
    'harvest_grain', 'herbicide', 'notill_rodweederrow',
    'reduced_till_tandemdisk', 'ryegrass_planting', 'soybean_planting',
    'winterwheat_planting',
]
CROPS = ['corn', 'soybean', 'winter_wheat']
PLANTING = ['corn_planting', 'soybean_planting', 'winterwheat_planting', 'ryegrass_planting']
POOL_COLUMNS = ['SOM1CI(2,1)', 'SOM2CI(2,1)', 'SOM3CI(1)', 'SOM2CI(1,1)']
NITROGEN = re.compile(r'^nitrogen_fertilization_([0-9]+(?:\.[0-9]+)?)gNm2$')
EVENT_DTYPE = np.dtype([('year', '<i4'), ('day', '<i2'), ('code', 'i1'), ('rate', '<f4')])
ARRAY_STATS = ['weather_mean', 'weather_std', 'static_mean', 'static_std',
               'pool_mean', 'pool_std', 'yield_crop_mean', 'yield_crop_std']


def load_stats(values):
    return {k: np.asarray(v, np.float32) if k in ARRAY_STATS else v for k, v in values.items()}


def integer_column(values, name):
    numeric = pd.to_numeric(values, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
        raise ValueError(f'{name} must contain finite integers')
    if np.any(np.abs(numeric) > 2**31-1):
        raise ValueError(f'{name} is outside the supported integer range')
    return numeric.astype(np.int64)


def labels(values, name):
    if values.isna().any():
        raise ValueError(f'Missing {name}')
    result = values.astype(str).str.strip()
    if result.eq('').any():
        raise ValueError(f'Empty {name}')
    return result


def calendar_days(years, days, name):
    if np.any((years < 1) | (years > 9999)):
        raise ValueError(f'{name}: invalid calendar year')
    lengths = np.array([366 if calendar.isleap(int(y)) else 365 for y in years])
    if np.any((days < 1) | (days > lengths)):
        raise ValueError(f'{name}: DOY must be a valid day in its calendar year')


def year_encoding(year):
    # Preserve the encoding used to train these checkpoints, including its
    # fixed 2000 origin when the requested inference window starts later.
    rel = int(year)-2000
    out = np.zeros(16, np.float32)
    i = np.arange(0, 16, 2)
    div = np.power(10000.0, 2.0*i/16)
    out[0::2], out[1::2] = np.sin(rel/div), np.cos(rel/div)
    return out


def canonical_events(parts):
    events = np.concatenate(parts)
    events.sort(order=['year', 'day', 'code', 'rate'])
    if len(events):
        same = np.ones(len(events)-1, bool)
        for key in ['year', 'day', 'code']:
            same &= events[key][:-1] == events[key][1:]
        # Binary practices; maximum N rate when the same event is repeated.
        events = events[np.r_[~same, True]]
    return events


def management_events(frame):
    """Return canonical-format events for one site/scenario in a CSV chunk."""
    rows, codes, rates = [], [], []
    if 'management' in frame:
        for i, name in enumerate(frame.management):
            if name in EVENT_TYPES:
                rows.append(i); codes.append(EVENT_TYPES.index(name)); rates.append(0.)
            else:
                match = NITROGEN.fullmatch(str(name))
                if match is None:
                    raise ValueError(f'Unsupported management practice: {name!r}')
                rate = float(match.group(1))
                if rate > 0:
                    rows.append(i); codes.append(-1); rates.append(rate)
    else:
        for code, name in enumerate(EVENT_TYPES):
            values = pd.to_numeric(frame[name], errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
                raise ValueError(f'{name} must be binary (0 or 1)')
            take = np.flatnonzero(values)
            rows.extend(take); codes.extend([code]*len(take)); rates.extend([0.]*len(take))
        for name in frame.columns:
            match = NITROGEN.fullmatch(name)
            if match is None:
                continue
            values = pd.to_numeric(frame[name], errors='raise').to_numpy(float)
            if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
                raise ValueError(f'{name} must be binary (0 or 1); its name specifies the N rate')
            rate = float(match.group(1))
            take = np.flatnonzero(values) if rate > 0 else np.array([], int)
            rows.extend(take); codes.extend([-1]*len(take)); rates.extend([rate]*len(take))
    out = np.empty(len(rows), EVENT_DTYPE)
    rows = np.asarray(rows, np.int64)
    out['year'], out['day'] = frame.Year.to_numpy()[rows], frame.doy.to_numpy()[rows]
    out['code'], out['rate'] = codes, rates
    if not np.isfinite(out['rate']).all():
        raise ValueError('Non-finite nitrogen rate')
    return out


class Inputs:
    def __init__(self, weather_dir, site_conditions, management, columns, first_year, last_year):
        if not 1 <= first_year <= last_year <= 9999:
            raise ValueError('Require 1 <= first-year <= last-year <= 9999')
        self.weather_dir = Path(weather_dir)
        self.columns = columns
        self.first_year, self.last_year = first_year, last_year
        self.years = np.arange(first_year, last_year+1)
        self.lengths = np.array([366 if calendar.isleap(int(y)) else 365 for y in self.years])
        self.starts = np.r_[0, np.cumsum(self.lengths)]
        self.year_enc = np.stack([year_encoding(y) for y in self.years])
        doy = np.arange(1, 367, dtype=np.float32)
        self.doy = np.stack([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)], -1)
        path = Path(site_conditions)
        if path.suffix.lower() in {'.xlsx', '.xlsm'}:
            with pd.ExcelFile(path, engine='openpyxl') as workbook:
                sheet = 'initial_conditions' if 'initial_conditions' in workbook.sheet_names else 0
                static = pd.read_excel(workbook, sheet_name=sheet, dtype={'id': str})
        else:
            static = pd.read_csv(path, dtype={'id': str})
        missing = set(['id']+columns)-set(static.columns)
        if missing:
            raise ValueError(f'Missing site-condition columns: {sorted(missing)}')
        static['id'] = labels(static.id, 'site id')
        if static.id.duplicated().any():
            raise ValueError('Site conditions must have exactly one row per id')
        self.static = static.set_index('id')[columns].apply(pd.to_numeric, errors='raise').astype(np.float32)
        if not np.isfinite(self.static.to_numpy()).all():
            raise ValueError('All 165 required site features must be finite numeric values')
        self.events = self.read_management(management)
        self.pairs = sorted(self.events)
        if not self.pairs:
            raise ValueError('No site/scenario management rows within the requested years')

    def read_management(self, paths):
        parts = {}
        known = set(self.static.index)
        for path in paths:
            for frame in pd.read_csv(path, chunksize=100000,
                                     dtype={'id': str, 'scenario_id': str, 'scenario': str}):
                aliases = {'scenario': 'scenario_id', 'simyear': 'Year', 'DOY': 'doy'}
                for old, new in aliases.items():
                    if old in frame:
                        if new in frame:
                            raise ValueError(f'{path}: supply {new} or {old}, not both')
                        frame = frame.rename(columns={old: new})
                keys = {'scenario_id', 'id', 'Year', 'doy'}
                if keys-set(frame.columns):
                    raise ValueError(f'{path}: management needs scenario_id,id,Year,doy')
                if 'management' not in frame:
                    missing = set(EVENT_TYPES)-set(frame.columns)
                    unknown = [c for c in frame.columns if c not in keys and c not in EVENT_TYPES and not NITROGEN.fullmatch(c)]
                    if missing or unknown:
                        raise ValueError(f'{path}: missing practice columns={sorted(missing)}, unknown columns={unknown}')
                frame['Year'] = integer_column(frame.Year, 'management Year')
                frame['doy'] = integer_column(frame.doy, 'management doy')
                frame = frame[frame.Year.between(self.first_year, self.last_year)].copy()
                calendar_days(frame.Year.to_numpy(), frame.doy.to_numpy(), 'management')
                frame['id'] = labels(frame.id, 'management id')
                frame['scenario_id'] = labels(frame.scenario_id, 'scenario_id')
                if set(frame.id)-known:
                    raise ValueError(f'Missing static rows for management ids: {sorted(set(frame.id)-known)[:10]}')
                for pair, group in frame.groupby(['scenario_id', 'id'], sort=False):
                    parts.setdefault(pair, []).append(management_events(group))
        return {pair: canonical_events(chunks) for pair, chunks in parts.items()}

    @lru_cache(maxsize=32)
    def weather(self, site):
        if Path(site).name != site or site in {'.', '..'} or '\\' in site:
            raise ValueError(f'Site id cannot contain path separators: {site!r}')
        path = self.weather_dir/f'{site}.csv'
        frame = pd.read_csv(path)
        needed = ['Year', 'DOY', 'Tmax', 'Tmin', 'Precip']
        if set(needed)-set(frame.columns):
            raise ValueError(f'{path}: expected columns {needed}')
        frame['Year'] = integer_column(frame.Year, 'weather Year')
        frame['DOY'] = integer_column(frame.DOY, 'weather DOY')
        frame = frame[frame.Year.between(self.first_year, self.last_year)]
        if frame.duplicated(['Year', 'DOY']).any():
            raise ValueError(f'{path}: duplicate weather dates')
        year, day = frame.Year.to_numpy(), frame.DOY.to_numpy()
        calendar_days(year, day, str(path))
        yi = year-self.first_year
        if not np.array_equal(np.bincount(yi, minlength=len(self.years)), self.lengths):
            raise ValueError(f'{path}: complete daily weather is required for every requested year')
        raw = np.zeros((len(self.years), 366, 3), np.float32)
        raw[yi, day-1] = frame[['Tmax', 'Tmin', 'Precip']].to_numpy(np.float32)
        if not np.isfinite(raw).all() or np.any(raw[:, :, 2] < 0):
            raise ValueError(f'{path}: non-finite weather or negative precipitation')
        # The trained models used a zero-valued raw day 366 in non-leap years.
        return raw

    def features(self, pair, stats):
        scenario, site = pair
        seq = np.zeros((len(self.years), 366, 16), np.float32)
        seq[:, :, :3] = (self.weather(site)-stats['weather_mean'])/stats['weather_std']
        seq[:, :, -2:] = self.doy
        events = self.events[pair]
        yi, day = events['year'].astype(np.int64)-self.first_year, events['day'].astype(np.int64)-1
        flags = events['code'] >= 0
        np.add.at(seq, (yi[flags], day[flags], 3+events['code'][flags]), 1.0)
        np.maximum.at(seq, (yi[~flags], day[~flags], 13), events['rate'][~flags])
        seq[:, :, 13] /= stats['nitrogen_scale']
        static = (self.static.loc[site].to_numpy()-stats['static_mean'])/stats['static_std']
        return seq, np.asarray(static, np.float32)

    def initial_pools(self, site):
        return self.static.loc[site, POOL_COLUMNS].to_numpy(np.float32)

    def yield_samples(self, pair, stats, yield_start_year):
        seq, static = self.features(pair, stats)
        events = self.events[pair]
        harvests = events[events['code'] == EVENT_TYPES.index('harvest_grain')]
        harvests = harvests[harvests['year'] >= yield_start_year]
        if len(np.unique(harvests['year'])) != len(harvests):
            raise ValueError(f'{pair}: the retained yield model supports at most one grain harvest per year')
        planting_cols = [3+EVENT_TYPES.index(p) for p in PLANTING]
        for event in harvests:
            year, day = int(event['year']), int(event['day'])
            yi = year-self.first_year
            if yi < 1:
                raise ValueError('Yield inference needs a preceding full calendar year of drivers')
            previous = seq[yi-1, self.lengths[yi-1]-365:self.lengths[yi-1]]
            daily = np.concatenate([previous, seq[yi]])
            daily = np.concatenate([daily, np.r_[np.zeros(365, np.float32), np.ones(366, np.float32)][:, None]], axis=1)
            end = 365+day
            plants = np.flatnonzero((daily[:end-1, planting_cols] > 0).any(1))
            if not len(plants):
                raise ValueError(f'{pair}, {year}: no planting before grain harvest within available season')
            start = int(plants[-1])
            types = np.flatnonzero(daily[start, planting_cols] > 0)
            if len(types) != 1 or types[0] == 3:
                raise ValueError(f'{pair}, {year}: grain harvest needs an unambiguous corn, soybean or winter-wheat planting')
            crop, length = int(types[0]), end-start
            aligned = np.zeros_like(daily)
            aligned[:length] = daily[start:end]
            mask = np.zeros(len(daily), np.float32); mask[:length] = 1
            planting = int(self.starts[yi])+day-length
            begin = max(0, planting-730)
            pieces = []
            for hy in range(max(0, yi-3), yi+1):
                lo, hi = max(begin, int(self.starts[hy])), min(planting, int(self.starts[hy+1]))
                if lo < hi:
                    pieces.append(seq[hy, lo-self.starts[hy]:hi-self.starts[hy]])
            antecedent = np.concatenate(pieces) if pieces else np.zeros((0, 16), np.float32)
            if len(antecedent) != planting-begin or len(antecedent) > 730:
                raise ValueError('Antecedent calendar alignment failed')
            summary = []
            for part in np.array_split(antecedent, 4):
                if not len(part):
                    summary.extend([0.]*16); continue
                weather = part[:, :3]*stats['weather_std']+stats['weather_mean']
                base = 0 if crop == 2 else 10
                summary.extend([part[:, 0].mean(), part[:, 1].mean(), weather[:, 2].sum()/100,
                                np.maximum(weather[:, :2].mean(1)-base, 0).sum()/1000])
                summary.extend(part[:, 3:13].sum(0).tolist())
                summary.extend([part[:, 13].sum(), len(part)/200])
            features = {'sequence': aligned, 'harvest_mask': mask, 'init_cond': static,
                        'year_enc': self.year_enc[yi], 'season_length': np.int64(length),
                        'antecedent_summary': np.asarray(summary, np.float32)}
            yield (pair[0], pair[1], year, day, crop), features
