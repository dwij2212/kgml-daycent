"""Annual four-pool targets, initial states, and monthly training labels."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import GridStore, INIT_POOL_COLS, load_stats, sha256, YEARS
BASE = WORK = Path('runs/unconfigured/data')

def key(sc):
    return f'{sc["family"]}_{sc["id"]:03d}'


class AnnualData:
    """Memory-mapped full-site drivers; independent of target availability at inference."""
    def __init__(self,work=BASE):
        self.work=Path(work);self.plan=json.loads((self.work/'plan.json').read_text())
        norm=json.loads((self.work/'normalization.json').read_text())
        assert sha256(self.work/'stats.json')==norm['stats_sha256']
        self.sites=np.asarray(self.plan['sites']);self.stats=load_stats(self.work/'stats.json')
        self.weather=np.load(self.work/'cache/weather.npy',mmap_mode='r')
        self.columns=json.loads((self.work/'cache/static_columns.json').read_text())
        self.static=np.load(self.work/'cache/static_synthetic.npy',mmap_mode='r')
        assert np.array_equal(self.static,np.load(self.work/'cache/static_realistic.npy',mmap_mode='r'))
        self.init=np.asarray(self.static[:,[self.columns.index(c) for c in INIT_POOL_COLS]],np.float32)
        self.static_norm=np.asarray((self.static-self.stats['static_mean'])/self.stats['static_std'],np.float32)
        self.year=np.stack([GridStore.year_encoding(SimpleNamespace(years=YEARS),y) for y in range(24)])
        doy=np.arange(1,367,dtype=np.float32)
        self.doy=np.stack([np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)],-1)
        self.loaded={}
    def scenario(self,c):
        if c not in self.loaded:
            sc=self.plan['scenarios'][c]
            self.loaded[c]=(np.load(self.work/f'cache/management/{sc["family"]}/{sc["id"]:03d}.npy',mmap_mode='r'),
                            np.load(self.work/'cache/offsets'/f'{key(sc)}.npy',mmap_mode='r'),
                            np.load(self.work/'targets'/f'{key(sc)}.npy',mmap_mode='r'))
        return self.loaded[c]
    def inputs(self,c,si):
        events,offsets,_=self.scenario(c)
        seq=np.zeros((24,366,16),np.float32)
        seq[:,:,:3]=(self.weather[si]-self.stats['weather_mean'])/self.stats['weather_std']
        seq[:,:,-2:]=self.doy
        ev=events[offsets[si*24]:offsets[(si+1)*24]]
        y=ev['year'].astype(np.int64)-2000;d=ev['day'].astype(np.int64)-1;code=ev['code']
        assert np.all((d>=0)&(d<366))
        flag=code>=0
        np.add.at(seq,(y[flag],d[flag],3+code[flag]),1.0)
        np.maximum.at(seq,(y[~flag],d[~flag],13),ev['rate'][~flag])
        seq[:,:,13]/=self.stats['nitrogen_scale']
        return {'sequence':torch.from_numpy(seq),'year_enc':torch.from_numpy(self.year.copy()),
                'init_cond':torch.from_numpy(self.static_norm[si].copy()),'prev0':torch.from_numpy(self.init[si].copy())}
    def item(self,c,si):
        b=self.inputs(c,si)
        b['true']=torch.from_numpy(self.scenario(c)[2][:,si].copy())
        return b


class Pairs(Dataset):
    def __init__(self,data,pairs):self.data=data;self.pairs=np.asarray(pairs,np.int64)
    def __len__(self):return len(self.pairs)
    def __getitem__(self,i):return self.data.item(*self.pairs[i])

class Data(AnnualData):
    def __init__(self, arm, monthly=True):
        super().__init__(BASE)
        assert sha256(WORK / 'stats.json') == sha256(BASE / 'stats.json')
        self.plan = json.loads((WORK / 'plan.json').read_text())
        self.arm = arm
        self.use_monthly = arm == 'monthly_k8' and monthly
        self.monthly_loaded = {}

    def item(self, c, si):
        batch = super().item(c, si)
        if self.use_monthly:
            if c not in self.monthly_loaded:
                sc = self.plan['scenarios'][c]
                if 'holdout' in sc['group']:
                    raise ValueError('Holdout monthly labels must not be used in training or selection')
                self.monthly_loaded[c] = np.load(WORK / 'monthly_targets' / (key(sc) + '.npy'), mmap_mode='r')
            a = self.monthly_loaded[c][:, si, :11].copy()
            mask = np.isfinite(a).all(-1)
            batch['monthly_mask'] = torch.from_numpy(mask)
            batch['monthly_true'] = torch.from_numpy(np.nan_to_num(a))
        return batch
