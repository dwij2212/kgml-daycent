"""Rebuild the final models' prepared arrays from the supplied Round-3 folder.

No existing derived folder, earlier dataset, or DayCent installation is read.
Parsing and floating-point operations preserve the original preparation rules.
Every required output is checked against the retained input SHA-256 manifest.
Daily PBM output files are not needed: only weather, site conditions, schedules,
harvest outputs, and monthly outputs are read, directly from ZIPs where supplied.
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import io
from pathlib import Path
import re
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from .common import EVENT_TYPES, sha256, write_json
from .config import METADATA, read
from .integrity import writable_output

EVENT_DTYPE = np.dtype([('id', '<i4'), ('year', '<i2'), ('day', '<i2'),
                        ('code', 'i1'), ('rate', '<f4')])
HARVEST_DTYPE = np.dtype([('id', '<i4'), ('year', '<i2'), ('day', '<i2'), ('cgrain', '<f4')])
N_PATTERN = re.compile(r'^nitrogen_fertilization_([0-9]+(?:\.[0-9]+)?)gNm2$')


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def canonical_events(records):
    """Binary practice events; maximum N rate for repeated same-day N events."""
    records.sort(order=['id', 'year', 'day', 'code', 'rate'])
    if len(records):
        same = np.ones(len(records)-1, bool)
        for field in ['id', 'year', 'day', 'code']:
            same &= records[field][:-1] == records[field][1:]
        records = records[np.r_[~same, True]]
    return records


def crop_classes(events, offsets, n_sites):
    """Most recent planting strictly before harvest, within the same site's history."""
    n = n_sites * 24
    trajectory = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
    code = events['code']
    crops = [EVENT_TYPES.index(k) for k in ['corn_planting', 'soybean_planting', 'winterwheat_planting']]
    cover = [EVENT_TYPES.index('ryegrass_planting')]
    key = trajectory * 400 + events['day'].astype(np.int64)
    planted = np.isin(code, crops + cover)
    pk, pc = key[planted], code[planted]
    if pk.size > 1 and np.any(np.diff(pk) < 0):
        order = np.argsort(pk, kind='stable'); pk, pc = pk[order], pc[order]
    harvest = code == EVENT_TYPES.index('harvest_grain')
    hk, ht = key[harvest], trajectory[harvest]
    j = np.searchsorted(pk, hk, side='left') - 1
    valid = j >= 0
    valid[valid] = (pk[j[valid]] // 400) // 24 == ht[valid] // 24
    lookup = np.full(max(crops + cover)+1, 3, np.int8)
    for k, event in enumerate(crops):
        lookup[event] = k
    classes = np.full(len(hk), 3, np.int8)
    classes[valid] = lookup[pc[j[valid]]]
    out = np.full(n, 3, np.int8); out[ht] = classes
    return out.reshape(n_sites, 24)


class RawBuilder:
    def __init__(self, raw_root, output):
        self.raw = Path(raw_root).expanduser().resolve()
        self.output = writable_output(output)
        if self.output == self.raw or self.raw in self.output.parents:
            raise ValueError('Put prepared data outside the original supplied data folder.')
        self.reference = read(METADATA / 'reference.json')
        self.sites = np.asarray(self.reference['sites'])
        self.scenarios = self.reference['scenarios']
        self.manifest = read(METADATA / 'input_manifest.json')
        self.driver = self.output / 'soc_improvement_v1'
        self.cache = self.driver / 'cache'
        self.yield_root = self.output / 'yield_improvement_v1'
        self.monthly = self.output / 'soc_followup_v1/monthly_targets'
        self.started = time.time()
        self.identity = {'raw_root': str(self.raw),
                         'reference_sha256': sha256(METADATA / 'reference.json'),
                         'input_manifest_sha256': sha256(METADATA / 'input_manifest.json')}
        state = self.output / 'raw_build.json'
        if self.output.exists() and any(self.output.iterdir()):
            if not state.is_file() or read(state)['identity'] != self.identity:
                raise ValueError('Use a new output directory or resume this same raw-data build.')
        self.output.mkdir(parents=True, exist_ok=True)
        self.status('starting')

    def status(self, stage, **extra):
        value = {'identity': self.identity, 'stage': stage,
                 'elapsed_seconds': round(time.time()-self.started), **extra}
        write_json(self.output / 'raw_build.json', value)
        print(stage, extra, flush=True)

    def checked(self, path):
        """An interrupted build resumes only files matching the frozen bytes."""
        relative = str(path.relative_to(self.output))
        expected = self.manifest['files'].get(relative)
        if expected is None:
            raise ValueError(f'Unexpected prepared-data file: {relative}')
        return (path.is_file() and path.stat().st_size == expected['bytes']
                and sha256(path) == expected['sha256'])

    def save(self, path, array):
        save_array(path, array)
        if not self.checked(path):
            raise ValueError(f'Raw data did not reproduce the frozen input: {path}')

    def inputs(self):
        columns = read(METADATA / 'static_columns.json')
        for family, folder in [('synthetic', 'Synthetic/InputData'), ('realistic', 'Realistic/InputFiles')]:
            dest = self.cache / f'static_{family}.npy'
            if self.checked(dest):
                continue
            path = self.raw / folder / 'InitConditions/initial_site_conditions.xlsx'
            frame = pd.read_excel(path, sheet_name='initial_conditions', engine='openpyxl').set_index('id')
            if frame.index.has_duplicates or not set(self.sites).issubset(frame.index):
                raise ValueError(f'Incomplete or duplicate site IDs: {path}')
            values = frame.loc[self.sites, columns].to_numpy(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite static feature: {path}')
            self.save(dest, values)
        path = self.cache / 'static_columns.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((METADATA / 'static_columns.json').read_bytes())
        if not self.checked(path):
            raise ValueError('Static feature-order checksum differs.')
        dest = self.cache / 'weather.npy'
        if self.checked(dest):
            return
        archive = self.raw / 'Synthetic/InputData/Weather.zip'
        realistic = self.raw / 'Realistic/InputFiles/Weather.zip'
        if sha256(archive) != sha256(realistic):
            raise ValueError('Synthetic and realistic weather archives differ.')
        weather = np.zeros((len(self.sites), 24, 366, 3), np.float32)
        lengths = np.array([366 if calendar.isleap(y) else 365 for y in range(2000, 2024)])
        with zipfile.ZipFile(archive) as z:
            for si, site in enumerate(self.sites):
                frame = pd.read_csv(io.BytesIO(z.read(f'Weather/{site}.csv')))
                frame = frame[frame.Year.between(2000, 2023)]
                if frame.duplicated(['Year', 'DOY']).any():
                    raise ValueError(f'Duplicate weather day at {site}')
                yi, di = frame.Year.to_numpy(np.int64)-2000, frame.DOY.to_numpy(np.int64)-1
                if not ((di >= 0) & (di < lengths[yi])).all():
                    raise ValueError(f'Invalid weather day at {site}')
                if not np.array_equal(np.bincount(yi, minlength=24), lengths):
                    raise ValueError(f'Incomplete weather at {site}')
                weather[si, yi, di] = frame[['Tmax', 'Tmin', 'Precip']].to_numpy(np.float32)
                if (si+1) % 1000 == 0:
                    self.status('weather', completed=si+1, total=len(self.sites))
        if not np.isfinite(weather).all():
            raise ValueError('Non-finite weather')
        # Non-leap padded day 366 is zero, as in the retained implementation.
        self.save(dest, weather)

    def parse_management(self, stream, family, realistic_id=None):
        allowed = {s['id'] for s in self.scenarios if s['family'] == family}
        codes = {name: i for i, name in enumerate(EVENT_TYPES)}
        rates = {name: 0.0 for name in EVENT_TYPES}
        # Spill chunks to disk instead of retaining the 15 GB synthetic CSV in RAM.
        with tempfile.TemporaryDirectory(prefix='management-', dir=self.output) as temp:
            parts = set(); scanned = 0
            for chunk in pd.read_csv(stream, chunksize=500000,
                                     usecols=['scenario', 'id', 'simyear', 'doy', 'management']):
                selected = chunk[chunk.id.isin(self.sites) & chunk.simyear.between(2000, 2023)]
                if len(selected):
                    for name in selected.management.unique():
                        if name not in codes:
                            match = N_PATTERN.fullmatch(str(name))
                            if match is None:
                                raise ValueError(f'Unsupported management practice: {name}')
                            codes[name], rates[name] = -1, float(match.group(1))
                    if realistic_id is None:
                        sid = selected.scenario.str.extract(r'^scenario_(\d+)$', expand=False)
                        if sid.isna().any():
                            raise ValueError('Invalid synthetic scenario label')
                        sid = sid.to_numpy(np.int16)
                    else:
                        if selected.scenario.nunique() != 1:
                            raise ValueError('Multiple realistic scenarios in one schedule')
                        sid = np.full(len(selected), realistic_id, np.int16)
                    events = np.empty(len(selected), EVENT_DTYPE)
                    events['id'] = selected.id.to_numpy(np.int32)
                    events['year'] = selected.simyear.to_numpy(np.int16)
                    events['day'] = selected.doy.to_numpy(np.int16)
                    events['code'] = selected.management.map(codes).to_numpy(np.int8)
                    events['rate'] = selected.management.map(rates).to_numpy(np.float32)
                    if np.any((events['day'] < 1) | (events['day'] > 366)):
                        raise ValueError('Invalid management day')
                    keep = ((events['code'] >= 0) | (events['rate'] > 0)) & np.isin(sid, list(allowed))
                    events, sid = events[keep], sid[keep]
                    for scenario in np.unique(sid):
                        with (Path(temp) / f'{scenario}.bin').open('ab') as f:
                            events[sid == scenario].tofile(f)
                        parts.add(int(scenario))
                scanned += len(chunk)
                if scanned % 5000000 == 0:
                    self.status('management', family=family, scanned_rows=scanned)
            for completed, scenario in enumerate(sorted(parts), 1):
                records = canonical_events(np.fromfile(Path(temp)/f'{scenario}.bin', dtype=EVENT_DTYPE))
                self.save(self.cache / 'management' / family / f'{scenario:03d}.npy', records)
                if completed % 10 == 0 or completed == len(parts):
                    self.status('management_arrays', family=family,
                                completed=completed, total=len(parts))

    def management(self):
        synthetic = [s for s in self.scenarios if s['family'] == 'synthetic']
        if not all(self.checked(self.cache/'management/synthetic'/f'{s["id"]:03d}.npy') for s in synthetic):
            archive = self.raw / 'Synthetic/InputData/ScheduleFiles/Synthetic_schedule_management_history_scen1_200.zip'
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n.endswith('.csv')]
                if len(names) != 1:
                    raise ValueError('Expected one consolidated synthetic schedule CSV')
                with z.open(names[0]) as stream:
                    self.parse_management(stream, 'synthetic')
        for sc in self.scenarios:
            if sc['family'] == 'realistic':
                dest = self.cache/'management/realistic'/f'{sc["id"]:03d}.npy'
                if not self.checked(dest):
                    path = self.raw/'Realistic/InputFiles/ScheduleFIles_Realistic'/f'{sc["name"]}_management_history.csv'
                    with path.open('rb') as stream:
                        self.parse_management(stream, 'realistic', sc['id'])
            dest = self.cache/'offsets'/f'{sc["key"]}.npy'
            if not self.checked(dest):
                events = np.load(self.cache/'management'/sc['family']/f'{sc["id"]:03d}.npy', mmap_mode='r')
                si = np.searchsorted(self.sites, events['id'])
                np.testing.assert_array_equal(self.sites[si], events['id'])
                trajectory = si*24+events['year'].astype(np.int64)-2000
                if np.any(np.diff(trajectory) < 0):
                    raise ValueError('Unsorted event trajectories')
                offsets = np.r_[0, np.cumsum(np.bincount(trajectory, minlength=len(self.sites)*24))]
                self.save(dest, offsets)

    @contextmanager
    def output_csv(self, scenario, kind):
        if scenario['family'] == 'realistic':
            path = self.raw/'Realistic/OutputData'/f'SAS_{scenario["name"]}_{kind}.csv'
            with path.open('rb') as stream:
                yield stream
        else:
            path = self.raw/'Synthetic/Output_new_ZippedFolders'/Path(scenario['source_archive']).name
            name = scenario['monthly_member'].replace('_monthly.csv', f'_{kind}.csv')
            with zipfile.ZipFile(path) as z:
                with z.open(name) as stream:
                    yield stream

    def yield_target(self, scenario):
        with self.output_csv(scenario, 'harvest') as stream:
            frame = pd.read_csv(stream, usecols=['id', 'simyear', 'dayofyr', 'cgrain'])
        years = np.floor(frame.simyear.to_numpy(float)).astype(np.int16)
        keep = frame.id.isin(self.sites).to_numpy() & (years >= 2000) & (years <= 2023)
        frame, years = frame[keep].copy(), years[keep]
        if not np.isfinite(frame.cgrain.to_numpy(float)).all():
            raise ValueError('Non-finite harvest yield')
        if pd.DataFrame({'id': frame.id.to_numpy(), 'year': years}).duplicated().any():
            raise ValueError('Multiple harvests per site-year are unsupported by this release.')
        records = np.empty(len(frame), HARVEST_DTYPE)
        records['id'], records['year'], records['day'], records['cgrain'] = frame.id, years, frame.dayofyr, frame.cgrain
        records.sort(order=['id', 'year', 'day'])
        events = np.load(self.cache/'management'/scenario['family']/f'{scenario["id"]:03d}.npy', mmap_mode='r')
        offsets = np.load(self.cache/'offsets'/f'{scenario["key"]}.npy', mmap_mode='r')
        crop = crop_classes(events, offsets, len(self.sites))
        harvest = np.zeros((len(self.sites), 24), np.int16)
        ev = events[events['code'] == EVENT_TYPES.index('harvest_grain')]
        t = np.searchsorted(self.sites, ev['id'])*24+ev['year'].astype(np.int64)-2000
        if len(np.unique(t)) != len(t):
            raise ValueError('Multiple scheduled harvests per site-year')
        harvest.ravel()[t] = ev['day']
        target = np.full((len(self.sites), 24), np.nan, np.float32)
        si, yi = np.searchsorted(self.sites, records['id']), records['year'].astype(np.int64)-2000
        np.testing.assert_array_equal(self.sites[si], records['id'])
        np.testing.assert_array_equal(harvest[si, yi], records['day'])
        target[si, yi] = records['cgrain']
        valid = np.isfinite(target); valid[:, 0] = False
        if np.any(crop[valid] >= 3):
            raise ValueError('Unknown crop at a scored harvest')
        return target, crop, harvest

    def soc_target(self, scenario):
        annual = self.driver/'targets'/f'{scenario["key"]}.npy'
        monthly = self.monthly/f'{scenario["key"]}.npy'
        training = 'holdout' not in scenario['group']
        if self.checked(annual) and (not training or self.checked(monthly)):
            return
        arr = np.full((24, len(self.sites), 12, 4), np.nan, np.float32)
        seen = np.zeros(arr.shape[:3], bool)
        cols = ['id', 'simyear', 'month', 'somsc', 'som2c.1.', 'som2c.2.', 'som3c']
        with self.output_csv(scenario, 'monthly') as stream:
            for chunk in pd.read_csv(stream, usecols=cols, chunksize=250000):
                year = np.floor(chunk.simyear.to_numpy(float)).astype(np.int64)
                keep = (year >= 2000) & (year <= 2023) & chunk.id.isin(self.sites).to_numpy()
                d = chunk.loc[keep]; yy = year[keep]-2000
                si = np.searchsorted(self.sites, d.id.to_numpy())
                raw_month = d.month.to_numpy(); mm = raw_month.astype(np.int64)-1
                if not np.array_equal(raw_month, mm+1) or not ((mm >= 0) & (mm < 12)).all():
                    raise ValueError('Invalid monthly timestamp')
                idx = np.ravel_multi_index((yy, si, mm), seen.shape)
                if len(np.unique(idx)) != len(idx) or seen[yy, si, mm].any():
                    raise ValueError('Duplicate monthly target')
                values = d[['somsc', 'som2c.2.', 'som3c', 'som2c.1.']].to_numpy(np.float32, copy=True)
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite monthly target')
                values[:, 0] = values[:, 0]-values[:, 1]-values[:, 2]
                arr[yy, si, mm] = values; seen[yy, si, mm] = True
        expected = np.ones_like(seen); expected[0, :, 0] = False
        if not np.array_equal(seen, expected):
            raise ValueError(f'Unexpected monthly coverage: {scenario["key"]}')
        self.save(annual, arr[:, :, 11].copy())
        if training:
            self.save(monthly, arr)

    def build(self, workers=2):
        try:
            self.inputs()
            self.status('weather_and_static_complete')
            self.management()
            self.status('management_complete')
            names = ['target', 'crop', 'harvest_day']
            need_yield = not all(self.checked(self.yield_root/f'{name}.npy') for name in names)
            arrays = {}
            if need_yield:
                self.yield_root.mkdir(parents=True, exist_ok=True)
                for name, dtype in zip(names, [np.float32, np.int8, np.int16]):
                    arrays[name] = np.lib.format.open_memmap(self.yield_root/f'{name}.tmp', mode='w+',
                                      dtype=dtype, shape=(len(self.scenarios), len(self.sites), 24))
            def one(index, scenario):
                if need_yield:
                    for name, data in zip(names, self.yield_target(scenario)):
                        arrays[name][index] = data
                self.soc_target(scenario)
                return scenario['key']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one, c, sc) for c, sc in enumerate(self.scenarios)]
                for completed, future in enumerate(as_completed(futures), 1):
                    self.status('targets', completed=completed, total=len(futures), scenario=future.result())
            if need_yield:
                for name in names:
                    arrays[name].flush()
                arrays.clear()
                for name in names:
                    dest = self.yield_root/f'{name}.npy'
                    dest.with_suffix('.tmp').replace(dest)
                    if not self.checked(dest):
                        raise ValueError(f'Yield cache checksum mismatch: {name}')
            files = sorted(set(self.manifest['targets']['yield']) | set(self.manifest['targets']['soc']))
            for relative in files:
                if not self.checked(self.output/relative):
                    raise ValueError(f'Prepared input checksum mismatch: {relative}')
            self.status('complete', files_verified=len(files),
                        byte_identical_to_retained_inputs=True,
                        depends_on_existing_derived_folder=False)
        except BaseException as exc:
            self.status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    RawBuilder(args.raw_root, args.output).build(args.workers)


if __name__ == '__main__':
    main()
