"""Harvest targets and original balanced crop/management sampling."""
from pathlib import Path
from types import SimpleNamespace
import calendar, json
import numpy as np
from torch.utils.data import Dataset
from .common import GridStore, load_stats
WORK = Path('runs/unconfigured/data')
DRIVER_WORK = Path('inputs/soc_improvement_v1')
GROUPS = ['earlier_20', 'synthetic_expanded_train', 'realistic_expanded_train']
SEED = 20261009

class Data:
    def __init__(self):
        self.plan=json.loads((WORK/'plan.json').read_text());self.sites=np.asarray(self.plan['sites']);self.ns=len(self.sites)
        self.stats=load_stats(WORK/'stats.json')
        self.target=np.load(WORK/'target.npy',mmap_mode='r');self.crop=np.load(WORK/'crop.npy',mmap_mode='r')
        self.harvest=np.load(WORK/'harvest_day.npy',mmap_mode='r')
        self.weather=np.load(DRIVER_WORK/'cache/weather.npy',mmap_mode='r')
        static=np.load(DRIVER_WORK/'cache/static_synthetic.npy',mmap_mode='r')
        self.static=np.asarray((static-self.stats['static_mean'])/self.stats['static_std'],np.float32)
        self.year=np.stack([GridStore.year_encoding(SimpleNamespace(years=np.arange(2000,2024)),y) for y in range(24)])
        doy=np.arange(1,367,dtype=np.float32);self.doy=np.stack([np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)],-1)
        self.loaded={}
    def annual(self,c,si,yi):
        if c not in self.loaded:
            sc=self.plan['scenarios'][c]
            self.loaded[c]=(np.load(DRIVER_WORK/f'cache/management/{sc["family"]}/{sc["id"]:03d}.npy',mmap_mode='r'),
                np.load(DRIVER_WORK/'cache/offsets'/f'{sc["key"]}.npy',mmap_mode='r'))
        events,offsets=self.loaded[c];t=si*24+yi;ev=events[offsets[t]:offsets[t+1]]
        seq=np.zeros((366,16),np.float32)
        seq[:,:3]=(self.weather[si,yi]-self.stats['weather_mean'])/self.stats['weather_std'];seq[:,-2:]=self.doy
        code=ev['code'];flag=code>=0;d=ev['day'].astype(np.int64)-1
        np.add.at(seq,(d[flag],3+code[flag]),1.0)
        np.maximum.at(seq,(d[~flag],13),ev['rate'][~flag]);seq[:,13]/=self.stats['nitrogen_scale']
        return seq
    def item(self,index,history=False):
        c=int(index)//(self.ns*24);t=int(index)%(self.ns*24);si=t//24;yi=t%24
        crop=int(self.crop[c,si,yi]);assert crop<3 and yi>0
        seq=self.annual(c,si,yi);mask=np.zeros(366,np.float32);mask[:int(self.harvest[c,si,yi])]=1
        if history:
            # Previous year is 2000 + yi - 1.
            prevlen=366 if calendar.isleap(1999+yi) else 365
            prev=self.annual(c,si,yi-1)[prevlen-365:prevlen] if crop==2 else np.zeros((365,16),np.float32)
            seq=np.concatenate([prev,seq]);mask=np.r_[np.ones(365,np.float32) if crop==2 else np.zeros(365,np.float32),mask]
            seq=np.concatenate([seq,np.r_[np.zeros(365,np.float32),np.ones(366,np.float32)][:,None]],axis=1)
        val=float(self.target[c,si,yi]);target=(val-self.stats['yield_crop_mean'][crop])/self.stats['yield_crop_std'][crop]
        return {'sequence':seq,'harvest_mask':mask,'init_cond':self.static[si], 'year_enc':self.year[yi],
            'crop_class':np.int64(crop),'yield':np.float32(target),'yield_true':np.float32(val),
            'point_id':np.int32(self.sites[si]),'year':np.int16(2000+yi),'scenario_index':np.int16(c)}
    def indices(self,sites,scenarios):
        wanted=np.zeros(self.ns,bool);wanted[np.searchsorted(self.sites,sites)]=True
        arrays=[]
        for c in scenarios:
            valid=np.isfinite(self.target[c]) & wanted[:,None] & (self.crop[c]<3);valid[:,0]=False
            arrays.append((np.flatnonzero(valid)+c*self.ns*24).astype(np.uint32))
        return np.concatenate(arrays)
    def train_indices(self,epoch,count):
        rng=np.random.default_rng(self.plan['seed']+epoch); crops=rng.integers(0,3,count); groups=rng.choice(3,count,p=[.3,.5,.2]);out=np.empty(count,np.uint32)
        for g in range(3):
            for k in range(3):
                keep=(groups==g)&(crops==k);bucket=np.load(WORK/'buckets'/f'{g}_{k}.npy',mmap_mode='r')
                out[keep]=bucket[rng.integers(0,len(bucket),int(keep.sum()))]
        return out


class Samples(Dataset):
    def __init__(self,data,indices,history=False):self.data=data;self.indices=indices;self.history=history
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.data.item(self.indices[i],self.history)
