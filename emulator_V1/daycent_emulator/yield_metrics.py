"""Original yield prediction, selection, and reporting formulas."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
WORK = Path('runs/unconfigured')
CROPS = ['corn', 'soy', 'wheat']

def move(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def metrics(y,p):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);e=p-y;sst=np.sum((y-y.mean())**2)
    return {'n':len(y),'r2':float(1-np.dot(e,e)/sst) if sst>0 else None,
        'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.mean(abs(e))),'bias':float(e.mean())}


def score_frame(f,data):
    sc=data.plan['scenarios'];out=[]
    for (c,k),g in f.groupby(['scenario_index','crop']):
        out.append({'group':sc[c]['group'],'scenario':sc[c]['key'],'crop':CROPS[k],**metrics(g.yield_true,g.yield_pred)})
    crop=[]
    for k,g in f.groupby('crop'):crop.append({'crop':CROPS[k],**metrics(g.yield_true,g.yield_pred)})
    gnames=np.array([sc[int(c)]['group'] for c in f.scenario_index]);cells=[]
    for group in np.unique(gnames):
        for k in range(3):
            m=(gnames==group)&(f.crop.to_numpy()==k)
            if m.any():cells.append(float(np.mean(((f.yield_pred.to_numpy()[m]-f.yield_true.to_numpy()[m])/data.stats['yield_crop_std'][k])**2)))
    return {'balanced_normalized_mse':float(np.mean(cells)), 'per_crop':crop,'per_scenario_crop':out}


@torch.no_grad()
def predict(model,dl,data,device):
    model.eval();parts=[];mean=np.asarray(data.stats['yield_crop_mean']);std=np.asarray(data.stats['yield_crop_std'])
    for b in dl:
        k=b['crop_class'].numpy();p=model(move(b,device))['yield_pred'].cpu().numpy()*std[k]+mean[k]
        assert np.isfinite(p).all()
        parts.append(pd.DataFrame({'point_id':b['point_id'].numpy(),'year':b['year'].numpy(),
            'crop':k,'scenario_index':b['scenario_index'].numpy(),'yield_true':b['yield_true'].numpy(),'yield_pred':p.astype(np.float32)}))
    return pd.concat(parts,ignore_index=True)


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.cpu().numpy().tobytes())
    return h.hexdigest()


def write_reports(arm,frames,data):
    run=WORK;f=pd.concat(frames,ignore_index=True);sc=data.plan['scenarios']
    f['group']=[sc[c]['group'] for c in f.scenario_index];f['scenario']=[sc[c]['key'] for c in f.scenario_index]
    for keys,name in [(['group'],'summary'),(['group','crop'],'per_crop'),(['scenario','group','crop'],'per_scenario_crop'),(['group','year','crop'],'per_year_crop')]:
        rows=[]
        for labels,g in f.groupby(keys):
            if not isinstance(labels,tuple):labels=(labels,)
            row=dict(zip(keys,labels))
            if 'crop' in row:row['crop']=CROPS[row['crop']]
            rows.append({**row,**metrics(g.yield_true,g.yield_pred)})
        pd.DataFrame(rows).to_csv(run/f'{name}.csv',index=False)
    return f
