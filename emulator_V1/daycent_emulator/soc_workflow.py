"""Selected width-512 SOC model, monthly supervision, and K8 training."""
from pathlib import Path
import hashlib, json, os, shutil, time
import numpy as np
import pandas as pd
import torch
from .soc_data import Data as OriginalData, key
from .soc_model import SOCModel as Model, year_batch, state_error
from . import soc_loop as base
from .common import write_json, sha256, soc_metrics as metrics
from .config import effective_plan
ROOT = Path('runs/unconfigured')
FOLLOWUP = ROOT / 'data'

class Data(OriginalData):
    def __init__(self,arm,monthly=True):
        super().__init__('monthly_k8',monthly=monthly);self.arm=arm
        self.plan=effective_plan(self.plan,'soc')


def batch_for(model,b,state,y,stats):
    return year_batch(b,state,y,stats,'monthly_k8')


def training_states(model,b,stats,k):
    # Kept separate for executable checks of forward state boundaries.
    state=b['prev0']
    for start in range(0,24,k):
        if start:
            state=b['true'][:,start-1]
        for y in range(start,start+k):
            out=model(batch_for(model,b,state,y,stats));state=out['soc_state_pred']
            yield start,y,out


def backward(model,b,stats,k,divisor=1):
    k=model.configure_epoch(model.training_epoch)
    total=0.0;loss=0.0
    for start,y,out in training_states(model,b,stats,k):
        pred=out['soc_state_pred'];loss=loss+state_error(pred,b['true'][:,y],stats,model.somsc_loss_weight).mean()
        err=state_error(out['monthly_state_pred'],b['monthly_true'][:,y],stats,model.somsc_loss_weight);mask=b['monthly_mask'][:,y]
        loss=loss+model.monthly_loss_weight*(err*mask).sum()/mask.sum().clamp_min(1)
        if (y+1)%k==0:
            loss=loss/24
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SOC loss')
            (loss/divisor).backward();total+=float(loss.detach());loss=0.0
    return total


def predict(model,b,stats,teacher=False):
    state=b['prev0'];outputs=[]
    for y in range(24):
        if teacher and y:state=b['true'][:,y-1]
        state=model(batch_for(model,b,state,y,stats))['soc_state_pred'];outputs.append(state)
    result=torch.stack(outputs,1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite SOC predictions')
    return result


def new_model(data,device):
    config=dict(data.plan['architecture']);loss=data.plan['loss'];unroll=data.plan['arms'][data.arm]['unroll']
    torch.manual_seed(data.plan['seed']);m=Model(data.arm,unroll=unroll,monthly_loss_weight=loss['monthly_weight'],somsc_loss_weight=loss['somsc_weight'],**config).to(device)
    h=hashlib.sha256()
    for name,p in sorted(m.state_dict().items()):h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    meta={'input_dim':config['input_dim'],'init_dim':config['init_dim'],'architecture':config,
        'args':{'hidden_dim':config['hidden_dim'],'lstm_layers':config['lstm_layers'],'target':'soc','soc_init_state':True,'no_year_enc':False,'soc_unroll':unroll},
        'initialization':'random_from_scratch','seed':data.plan['seed'],'experiment_arm':data.arm,
        'initial_weight_sha256':h.hexdigest(),'pretrained_checkpoint_loaded':False,
        'parameter_count':sum(p.numel() for p in m.parameters()),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(device),'TF32':True,
        'cumulative_loss_weight':0,'monthly_loss_weight':loss['monthly_weight'],'loss':loss,'training_policy':{'policy':f'reset{unroll}'},'warmup_epochs':0,
        'input_stats_sha256':sha256(FOLLOWUP/'stats.json'),'run_plan_sha256':sha256(ROOT/'plan.json')}
    return m,meta


def model_from(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False);loss=ck.get('loss',{'monthly_weight':.5,'somsc_weight':.5})
    m=Model(ck['experiment_arm'],unroll=ck['args']['soc_unroll'],monthly_loss_weight=loss['monthly_weight'],somsc_loss_weight=loss['somsc_weight'],**ck['architecture']).to(device);m.load_state_dict(ck['model']);return m,ck


def save_checkpoint(path,metadata,model,arm,epoch,selection):
    ck=dict(metadata,model={k:v.detach().cpu() for k,v in model.state_dict().items()},experiment_arm=arm,epoch=epoch,selection=selection)
    ck['args']=dict(ck['args'],soc_unroll=model.configure_epoch(max(epoch,1)))
    tmp=path.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(path)


def probe(arm,device):
    d=Data(arm);pairs=base.sample_pairs(d,arm,1);k=d.plan['arms'][arm]['unroll'];records=[];chosen=None
    for micro in [d.plan['training']['micro_batch']]:
        m=None;b=None
        try:
            m,meta=new_model(d,device);b=base.to_device(next(iter(base.loader(d,pairs[:micro],micro,0))),device)
            torch.cuda.reset_peak_memory_stats(device);start=time.time();values=[]
            for ep in [1,21]:
                m.configure_epoch(ep);m.zero_grad(set_to_none=True);values.append(backward(m,b,d.stats,k))
                assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            value=values[-1]
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),d.plan['training']['gradient_clip']);assert torch.isfinite(norm)
            torch.cuda.synchronize();records.append({'micro_batch':micro,'seconds':time.time()-start,'peak_bytes':torch.cuda.max_memory_allocated(device),'loss':value,'phase_losses':values,'epochs_probed':[1,21]});chosen=micro;break
        except torch.cuda.OutOfMemoryError:
            records.append({'micro_batch':micro,'OOM':True});del m,b;torch.cuda.empty_cache()
    if chosen is None:raise RuntimeError('Configured SOC micro_batch does not fit; lower it in config.yml and prepare a new run')
    # Test causality after a real update, since the initial residual head is zero.
    opt=torch.optim.Adam(m.parameters(),lr=d.plan['training']['lr']);opt.step()
    del b,opt;torch.cuda.empty_cache()
    sample=base.to_device(next(iter(base.loader(d,pairs[:2],2,0))),device);m.eval()
    with torch.no_grad():
        original=predict(m,sample,d.stats);inputs={k:v for k,v in sample.items() if k not in ['true','monthly_true','monthly_mask']}
        assert torch.equal(original,predict(m,inputs,d.stats))
        changed={k:v.clone() for k,v in inputs.items()};changed['sequence'][:,1:]*=1.3
        assert torch.equal(original[:,:1],predict(m,changed,d.stats)[:,:1])
        clone=Model(arm,**meta['architecture']).to(device);clone.load_state_dict(m.state_dict());clone.eval();assert torch.equal(original,predict(clone,sample,d.stats))
    result={'state':'complete','passed':True,'arm':arm,'micro_batch':chosen,'effective_batch':d.plan['training']['effective_batch'],'benchmarks':records,
        'autonomous_without_PBM_targets':True,'future_year_causality':True,'checkpoint_reload':True,'initialization':meta,
        'source_monthly_audit':str(FOLLOWUP/'prepare_status.json')}
    write_json(ROOT/'probe.json',result);print(json.dumps(result),flush=True)


def summarize(arm):
    d=Data(arm,monthly=False);p=d.plan;out=ROOT;parts={};per=[];annual=[]
    for sc in p['scenarios']:
        with np.load(out/'predictions'/f'{key(sc)}.npz') as z:
            true=z['true'][:,:,:3].sum(-1);init=z['init'][:,:3].sum(-1);dt=true-np.concatenate([init[None],true[:-1]])
            for mode in ['closed_loop','teacher_forced']:
                pools=z[mode];pred=pools[:,:,:3].sum(-1);dp=pred-np.concatenate([init[None],pred[:-1] if mode=='closed_loop' else true[:-1]])
                parts.setdefault((sc['group'],mode),[]).append((true,pred,init,dt,dp,pools))
                row={'arm':arm,'group':sc['group'],'scenario':key(sc),'mode':mode}
                row.update({'final_cumulative_'+k:v for k,v in metrics(pred[-1]-init,true[-1]-init).items()})
                row.update({'annual_change_'+k:v for k,v in metrics(dp,dt).items()})
                for i,name in enumerate(['active_soil','slow_soil','passive_soil','slow_surface']):row.update({name+'_final_'+k:v for k,v in metrics(pools[-1,:,i],z['true'][-1,:,i]).items()})
                per.append(row)
    summary=[]
    for (group,mode),values in parts.items():
        true,pred,init,dt,dp,pools=[np.concatenate([x[j] for x in values],axis=0 if j==2 else 1) for j in range(6)]
        row={'arm':arm,'group':group,'mode':mode,'sites':len(p['test_sites']),'scenarios':len(values)}
        for label,a,b in [('final_stock',pred[-1],true[-1]),('final_cumulative',pred[-1]-init,true[-1]-init),('annual_change',dp,dt)]:row.update({label+'_'+k:v for k,v in metrics(a,b).items()})
        row['negative_predicted_pool_entries']=int((pools<0).sum());summary.append(row)
        for y in range(24):annual.append({'arm':arm,'group':group,'mode':mode,'year':2000+y,**metrics(pred[y]-init,true[y]-init)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(per).to_csv(out/'per_scenario.csv',index=False);pd.DataFrame(annual).to_csv(out/'per_year.csv',index=False)
    write_json(out/'results.json',{'state':'complete','arm':arm,'summary':summary})


def setup():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'stats.json'
    if not path.exists():
        tmp=path.with_name(f'stats.{os.getpid()}.tmp');shutil.copy2(FOLLOWUP/'stats.json',tmp);tmp.replace(path)
    base.WORK=ROOT;base.Data=Data;base.new_model=new_model;base.model_from=model_from
    base.training_backward=backward;base.predict=predict;base.save_checkpoint=save_checkpoint;base.summarize=summarize
