"""Original optimizer loop, balanced validation, sampler, and rollout evaluation.

soc_workflow.setup binds the selected model and loss functions below.
"""
from pathlib import Path
import json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import write_json
from .soc_data import Pairs, key
WORK = Path('runs/unconfigured')

def runtime(device,training_seed=20261008):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(device)
    torch.manual_seed(training_seed);np.random.seed(training_seed)
    return torch.device(device)


def loader(data,pairs,batch_size,workers=2):
    return DataLoader(Pairs(data,pairs),batch_size=batch_size,shuffle=False,num_workers=workers,
                      pin_memory=True,persistent_workers=workers>0)


def to_device(b,device):return {k:v.to(device,non_blocking=True) for k,v in b.items()}


def score_selection(model,data,device):
    p=data.plan;site_indices=np.searchsorted(data.sites,p['selection_sites'])
    result=[];model.eval()
    with torch.inference_mode():
        for c in p['selection_scenario_indices']:
            pairs=[(c,int(si)) for si in site_indices];errors=[];final=[]
            for batch in loader(data,pairs,128,workers=0):
                b=to_device(batch,device);pred=predict(model,b,data.stats)
                sd=torch.as_tensor(data.stats['pool_delta_std'],device=device)
                pool=((pred-b['true'])/sd).square().mean((1,2))
                total=((pred[:,:,:3].sum(-1)-b['true'][:,:,:3].sum(-1))/float(data.stats['somsc_delta_std'])).square().mean(1)
                errors.extend((pool+p.get('loss',{}).get('somsc_weight',0.5)*total).cpu().tolist())
                final.extend((pred[:,-1,:3].sum(-1)-b['true'][:,-1,:3].sum(-1)).cpu().tolist())
            result.append({'scenario':key(p['scenarios'][c]),'group':p['scenarios'][c]['group'],
                           'loss':float(np.mean(errors)),'final_stock_rmse':float(np.sqrt(np.mean(np.square(final)))),
                           'final_stock_bias':float(np.mean(final))})
    groups={g:float(np.mean([r['loss'] for r in result if r['group']==g])) for g in sorted({r['group'] for r in result})}
    return {'loss':float(np.mean(list(groups.values()))),'group_loss':groups,'per_scenario':result}


def sample_pairs(data,arm,epoch):
    p=data.plan;t=p['training'];n=t['steps_per_epoch']*t['effective_batch']
    # Site draws are identical across arms, including the horizon comparison.
    sites=np.random.default_rng(p['seed']+epoch*1000).choice(p['train_sites'],n)
    rng=np.random.default_rng(p['seed']+epoch*1000+1)
    if p['arms'][arm]['coverage']=='earlier20':
        cs=rng.choice(p['scenario_groups']['earlier_20'],n)
    else:
        names=list(t['expanded_sampling']);probs=list(t['expanded_sampling'].values())
        group=rng.choice(len(names),n,p=probs);cs=np.empty(n,np.int64)
        for j,name in enumerate(names):
            use=group==j;cs[use]=rng.choice(p['scenario_groups'][name],int(use.sum()))
    assert not set(cs)&set(p['scenario_groups']['synthetic_holdout']+p['scenario_groups']['realistic_holdout'])
    assert not (set(sites)&set(p['test_sites']) or set(sites)&set(p['validation_sites']))
    return np.stack([cs,np.searchsorted(data.sites,sites)],1)


def train(arm,device):
    data=Data(arm);p=data.plan;t=p['training'];k=p['arms'][arm]['unroll']
    micro=json.loads((WORK/'probe.json').read_text())['micro_batch'];acc=t['effective_batch']//micro
    assert t['effective_batch']%micro==0
    out=WORK;out.mkdir(parents=True,exist_ok=True)
    # setup() already places the scaler in this flat run directory.
    m,base=new_model(data,device)
    write_json(out/'initialization.json',{k:v for k,v in base.items() if k!='args'})
    opt=torch.optim.Adam(m.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['epochs'],eta_min=t.get('min_lr',t['lr']*0.02))
    started=time.time();history=[];first=1;best=float('inf');best_epoch=0
    def status(state,**kw):
        write_json(out/'status.json',{'state':state,'arm':arm,'pid':os.getpid(),'device':str(device),
             'updated_epoch':time.time(),'elapsed_seconds':round(time.time()-started),
             'epochs_total':t['epochs'],'steps_per_epoch':t['steps_per_epoch'],
             'micro_batch':micro,'effective_batch':t['effective_batch'],'unroll':k,**kw})
    status('running',phase='initial validation',epoch=0)
    try:
        if (out/'last.pt').exists():
            ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
            m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
            torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu(),device)
            history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];first=ck['epoch']+1
        else:
            initial=score_selection(m,data,device);best=initial['loss']
            write_json(out/'initial_validation.json',initial)
            save_checkpoint(out/'best.pt',base,m,arm,0,initial)
            print('INITIAL validation',best,initial['group_loss'],flush=True)
        for ep in range(first,t['epochs']+1):
            epstart=time.time();m.train();k=m.configure_epoch(ep);pairs=sample_pairs(data,arm,ep)
            dl=loader(data,pairs,micro,2);total=0.0;updates=0;norm=0.0
            opt.zero_grad(set_to_none=True)
            for j,batch in enumerate(dl):
                b=to_device(batch,device)
                total+=training_backward(m,b,data.stats,k,acc)
                if (j+1)%acc==0:
                    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),t['gradient_clip'])
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite training gradient')
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    if updates%16==0:
                        status('running',phase='training',epoch=ep,step=updates,best_epoch=best_epoch,
                               train_loss=total/(j+1),best_validation_loss=best)
            assert updates==t['steps_per_epoch']
            del dl
            status('running',phase='24-year validation',epoch=ep,step=updates,best_epoch=best_epoch)
            val=score_selection(m,data,device);sched.step()
            if val['loss']<best:
                best=val['loss'];best_epoch=ep;save_checkpoint(out/'best.pt',base,m,arm,ep,val)
            row={'epoch':ep,'train_loss':total/(updates*acc),'validation_loss':val['loss'],
                 'validation_group_loss':val['group_loss'],'best_epoch':best_epoch,
                 'seconds':time.time()-epstart,'learning_rate':opt.param_groups[0]['lr'],
                 'last_gradient_norm_before_clip':float(norm),'training_unroll':k,'state_boundary_policy':base['training_policy']['policy']}
            history.append(row);write_json(out/'history.json',history)
            write_json(out/'latest_validation.json',val)
            ck={'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                'epoch':ep,'history':history,'best':best,'best_epoch':best_epoch,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(device)}
            torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            print(json.dumps(row),flush=True)
        write_json(out/'training_results.json',{'state':'complete','arm':arm,'best_epoch':best_epoch,
                    'best_validation_loss':best,'history':history,'total_updates':t['epochs']*t['steps_per_epoch'],
                    'annual_targets_presented':t['epochs']*t['steps_per_epoch']*t['effective_batch']*24})
        status('trained',phase='awaiting final evaluation',epoch=t['epochs'],best_epoch=best_epoch)
    except BaseException as e:
        status('failed',error=f'{type(e).__name__}: {e}');raise


def evaluate(arm,device):
    data=Data(arm,monthly=False);p=data.plan;out=WORK;out.mkdir(parents=True,exist_ok=True)
    pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
    held=p['test_sites'];site_indices=np.searchsorted(data.sites,held);nsc=len(p['scenarios'])
    m,_=model_from(out/'best.pt',device);m.eval()
    order=sorted(range(nsc),key=lambda c:(p['scenarios'][c]['family']!='realistic',
                     'holdout' not in p['scenarios'][c]['group'],p['scenarios'][c]['id']))
    start=time.time();done=0
    def status(state,**extra):
        write_json(out/'evaluation_status.json',{'state':state,'pid':os.getpid(),'device':str(device),
                   'updated_epoch':time.time(),'completed':done,'total':nsc,'elapsed_seconds':round(time.time()-start),**extra})
    try:
        for c in order:
            sc=p['scenarios'][c];path=pred_dir/f'{key(sc)}.npz'
            if path.exists():done+=1;continue
            status('running',active=key(sc))
            auto=[];teacher=[];truth=[]
            with torch.inference_mode():
                for batch in loader(data,[(c,int(si)) for si in site_indices],256,2):
                    b=to_device(batch,device)
                    auto.append(predict(m,b,data.stats).cpu().numpy())
                    teacher.append(predict(m,b,data.stats,teacher=True).cpu().numpy())
                    truth.append(batch['true'].numpy())
            values={'site':np.asarray(held),'years':np.arange(2000,2024),'init':data.init[site_indices],
                    'true':np.concatenate(truth).transpose(1,0,2),
                    'closed_loop':np.concatenate(auto).transpose(1,0,2),
                    'teacher_forced':np.concatenate(teacher).transpose(1,0,2)}
            with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**values)
            path.with_suffix('.tmp').replace(path);done+=1
            print('EVAL',arm,key(sc),done,nsc,round(time.time()-start),flush=True)
        summarize(arm);status('complete')
        state=json.loads((out/'status.json').read_text());state.update(state='complete',phase='training and test evaluation complete',updated_epoch=time.time())
        write_json(out/'status.json',state)
    except BaseException as e:
        status('failed',error=f'{type(e).__name__}: {e}');raise
