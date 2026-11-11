"""Selected yield model: season alignment, antecedent context, paired loss."""
from pathlib import Path
import calendar, json, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import EVENT_TYPES, write_json, sha256
from .config import effective_plan, reference_recipe
from .yield_model import YieldModel
from . import yield_data as yd
from . import yield_metrics as shared
ROOT = Path('runs/unconfigured')

class Data(yd.Data):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.plan=effective_plan(self.plan,'yield')
        self.allowed_mask=np.isin(np.arange(198),sum([self.plan["scenario_groups"][g] for g in yd.GROUPS],[]))
        self.train_mask=np.isin(self.sites,self.plan["train_sites"])
        self.year_starts=np.r_[0,np.cumsum([366 if calendar.isleap(y) else 365 for y in range(2000,2024)])]
    def item(self,index,history=False):
        temporal=('history' in self.arm or 'season' in self.arm)
        b=super().item(index,temporal)
        if temporal:
            c=int(index)//(self.ns*24);si=(int(index)//24)%self.ns;yi=int(index)%24
            if b['crop_class']!=2:
                days=366 if calendar.isleap(1999+yi) else 365
                b['sequence'][:365,:16]=self.annual(c,si,yi-1)[days-365:days]
                b['harvest_mask'][:365]=1
        if 'season' in self.arm:
            end=int(np.flatnonzero(b['harvest_mask'])[-1])+1
            cols=[3+EVENT_TYPES.index(k) for k in ['corn_planting','soybean_planting','winterwheat_planting','ryegrass_planting']]
            plant=np.flatnonzero((b['sequence'][:end-1,cols]>0).any(1))
            if not len(plant):raise ValueError(f'No planting in available season: {index}')
            start=int(plant[-1]);length=end-start
            seq=np.zeros_like(b['sequence']);seq[:length]=b['sequence'][start:end];b['sequence']=seq
            b['harvest_mask']=np.zeros(len(seq),np.float32);b['harvest_mask'][:length]=1
            b['season_length']=np.int64(length)
        # Driver-only antecedent context: last 730 available days, strictly before planting.
        end=int(self.year_starts[yi])+int(self.harvest[c,si,yi])
        planting=end-int(b['season_length']);begin=max(0,planting-730)
        pieces=[]
        for hy in range(max(0,yi-3),yi+1):
            lo=max(begin,int(self.year_starts[hy]));hi=min(planting,int(self.year_starts[hy+1]))
            if lo<hi:pieces.append(self.annual(c,si,hy)[lo-self.year_starts[hy]:hi-self.year_starts[hy]])
        antecedent=np.concatenate(pieces) if pieces else np.zeros((0,16),np.float32)
        assert len(antecedent)==planting-begin and len(antecedent)<=730
        features=[]
        for part in np.array_split(antecedent,4):
            if len(part)==0:features.extend([0.0]*16);continue
            weather=part[:,:3]*self.stats['weather_std']+self.stats['weather_mean']
            base=0 if b['crop_class']==2 else 10
            features.extend([part[:,0].mean(),part[:,1].mean(),weather[:,2].sum()/100,
                np.maximum(weather[:,:2].mean(1)-base,0).sum()/1000])
            features.extend(part[:,3:13].sum(0).tolist())
            features.extend([part[:,13].sum(),len(part)/200])
        b['antecedent_summary']=np.asarray(features,np.float32)
        assert b['antecedent_summary'].shape==(64,)
        return b
    def train_indices(self,epoch,count):
        anchors=super().train_indices(epoch,count//2);si=(anchors//24)%self.ns;yi=anchors%24;cs=anchors//(self.ns*24)
        allowed=np.asarray(sum([self.plan['scenario_groups'][g] for g in yd.GROUPS],[]))
        crop=self.crop[cs,si,yi]
        possible=(self.crop[allowed[:,None],si,yi]==crop)&np.isfinite(self.target[allowed[:,None],si,yi])&(allowed[:,None]!=cs)
        counts=possible.sum(0);assert (counts>0).all(),'No independent management partner'
        rng=np.random.default_rng(self.plan['seed']+100000+epoch);rank=(rng.random(len(anchors))*counts).astype(int)
        choice=(np.cumsum(possible,axis=0,dtype=np.int16)>rank).argmax(0)
        partner=allowed[choice]*self.ns*24+si*24+yi
        out=np.empty(count,np.uint32);out[0::2]=anchors;out[1::2]=partner
        return out


def model(arm,architecture=None):
    return YieldModel(**(architecture or reference_recipe('yield')['architecture']))


def audit_indices(d,indices):
    c=indices//(d.ns*24);si=(indices//24)%d.ns;yi=indices%24
    assert d.allowed_mask[c].all() and d.train_mask[si].all() and (yi>0).all()
    assert np.isfinite(d.target[c,si,yi]).all()
    assert np.array_equal(si[::2],si[1::2]) and np.array_equal(yi[::2],yi[1::2])
    assert (c[::2]!=c[1::2]).all()
    assert np.array_equal(d.crop[c[::2],si[::2],yi[::2]],d.crop[c[1::2],si[1::2],yi[1::2]])


def loader(data,indices,batch=256,workers=4):
    return DataLoader(yd.Samples(data,indices),batch_size=batch,num_workers=workers,pin_memory=True,persistent_workers=workers>0)


def seed(arm,training_seed=None):
    yd.SEED=reference_recipe('yield')['training_seed'] if training_seed is None else training_seed
    random.seed(yd.SEED);np.random.seed(yd.SEED);torch.manual_seed(yd.SEED);torch.cuda.manual_seed_all(yd.SEED)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True


def check(arm,device):
    d=Data(arm);cfg=d.plan['training'];seed(arm,d.plan['seed']);batch=cfg['batch_size']
    indices=d.train_indices(1,batch);audit_indices(d,indices);b=next(iter(loader(d,indices,batch,0)));b=shared.move(b,device)
    # DataLoader iterator creation consumes Torch RNG; match train initialization.
    seed(arm,d.plan['seed'])
    m=model(arm,d.plan['architecture']).to(device);initial=shared.parameter_hash(m);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    started=time.time()
    for step in range(3):
        opt.zero_grad(set_to_none=True);pred=m(b)['yield_pred'];loss=(pred-b['yield']).square().mean()
        assert torch.equal(b['crop_class'][0::2],b['crop_class'][1::2]) and torch.equal(b['point_id'][0::2],b['point_id'][1::2]) and torch.equal(b['year'][0::2],b['year'][1::2])
        assert torch.all(b['scenario_index'][0::2]!=b['scenario_index'][1::2])
        loss=loss+d.plan['loss']['pair_weight']*((pred[0::2]-pred[1::2])-(b['yield'][0::2]-b['yield'][1::2])).square().mean()
        assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip']);opt.step()
    torch.cuda.synchronize()
    m.eval()
    with torch.no_grad():
        x=m(b)['yield_pred'];alter={k:v.clone() for k,v in b.items()};ignore=b['harvest_mask']==0
        alter['sequence'][ignore]=100*torch.randn_like(alter['sequence'][ignore]);y=m(alter)['yield_pred']
        assert torch.allclose(x,y,atol=2e-5,rtol=2e-5)
        clean={k:v for k,v in b.items() if k not in ['yield','yield_true','point_id','scenario_index','year']}
        assert torch.equal(x,m(clean)['yield_pred'])
        copy=model(arm,d.plan['architecture']).to(device);copy.load_state_dict(m.state_dict());copy.eval();assert torch.allclose(x,copy(b)['yield_pred'],atol=2e-5,rtol=2e-5)
    allowed=set(sum([d.plan['scenario_groups'][g] for g in yd.GROUPS],[]))
    assert set((indices//(d.ns*24)).tolist()).issubset(allowed)
    assert set(d.sites[(indices//24)%d.ns]).issubset(d.plan['train_sites'])
    result={'passed':True,'arm':arm,'device':str(device),'gpu':torch.cuda.get_device_name(device),'torch':str(torch.__version__),
        'three_steps_seconds':time.time()-started,'initial_hash':initial,'parameters':sum(v.numel() for v in m.parameters()),'batch_size':batch,
        'finite_backward':True,'no_post_harvest_driver_dependence':True,'reload_parity':True,'label_and_id_removal_invariant':True,'config':cfg,'sampled_sites_and_scenarios_allowed':True,
        'sequence_length':int(b['sequence'].shape[1]),'source_input_verification':str(yd.WORK/'verification.json')}
    write_json(ROOT/'probe.json',result);print(json.dumps(result),flush=True)


def management_scores(frame):
    p=frame.merge(frame,on=['point_id','year','crop'],suffixes=('_a','_b'))
    p=p[p.scenario_index_a<p.scenario_index_b]
    rows=[]
    for crop,g in p.groupby('crop'):
        rows.append({'crop':shared.CROPS[crop],**shared.metrics(g.yield_true_b-g.yield_true_a,g.yield_pred_b-g.yield_pred_a)})
    return {'validation_only':True,'pairing':'all unordered same-site/year/crop allowed-scenario pairs','metrics':rows}


def train(arm,device):
    d=Data(arm);seed(arm,d.plan['seed']);run=ROOT;run.mkdir(parents=True,exist_ok=True);cfg=d.plan['training']
    epochs,steps,batch=cfg['epochs'],cfg['steps_per_epoch'],cfg['batch_size']
    pair_weight=d.plan['loss']['pair_weight']
    m=model(arm,d.plan['architecture']).to(device);initial=shared.parameter_hash(m)
    assert initial==json.loads((run/'probe.json').read_text())['initial_hash']
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=cfg['min_lr'])
    history=[];first=1;best=float('inf');started=time.time()
    provenance={'arm':arm,'initialization':'random_from_scratch','initial_hash':initial,'pretrained_checkpoint_loaded':False,
        'torch':str(torch.__version__),'numpy':str(np.__version__),'gpu':torch.cuda.get_device_name(device),'seed':d.plan['seed'],
        'input_stats_sha256':sha256(yd.WORK/'stats.json'),'source_plan_sha256':sha256(yd.WORK/'plan.json'),'run_plan_sha256':sha256(ROOT/'plan.json'),
        'TF32':True,'epochs':epochs,'steps_per_epoch':steps,'batch_size':batch,'pair_loss_weight':pair_weight,'config':cfg,'architecture':d.plan['architecture']}
    write_json(run/'provenance.json',provenance)
    if (run/'last.pt').exists():
        ck=torch.load(run/'last.pt',map_location=device,weights_only=False);assert ck['initial_hash']==initial and ck['arm']==arm
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);history=ck['history'];best=ck['best'];first=ck['epoch']+1
        torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu(),device)
    assert set(d.plan['selection_sites'])<=set(d.plan['validation_sites'])
    assert not (set(d.plan['train_sites']) & (set(d.plan['validation_sites'])|set(d.plan['test_sites'])))
    assert d.allowed_mask[d.plan['selection_scenario_indices']].all()
    valix=d.indices(d.plan['selection_sites'],d.plan['selection_scenario_indices']);val=loader(d,valix)
    def status(state,**kw):write_json(run/'status.json',dict(state=state,arm=arm,pid=os.getpid(),updated_epoch=time.time(),elapsed_seconds=time.time()-started,**kw))
    for ep in range(first,epochs+1):
        start=time.time();m.train();ix=d.train_indices(ep,steps*batch);audit_indices(d,ix);dl=loader(d,ix,batch);total=0
        for step,b in enumerate(dl,1):
            b=shared.move(b,device);opt.zero_grad(set_to_none=True);pred=m(b)['yield_pred'];loss=(pred-b['yield']).square().mean()
            loss=loss+pair_weight*((pred[0::2]-pred[1::2])-(b['yield'][0::2]-b['yield'][1::2])).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip']);opt.step();total+=float(loss)
            if step==1 or step%32==0:status('training',epoch=ep,epochs=epochs,step=step,steps=steps,train_loss=total/step)
        del dl
        status('validating',epoch=ep,epochs=epochs)
        frame=shared.predict(m,val,d,device);scores=shared.score_frame(frame,d);value=scores['balanced_normalized_mse']
        ck={'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'arm':arm,'epoch':ep,'initial_hash':initial,'validation':scores,'provenance':provenance,'architecture':d.plan['architecture']}
        if value<best:
            best=value;write_json(run/'validation_management.json',management_scores(frame));torch.save(ck,run/'best.tmp');(run/'best.tmp').replace(run/'best.pt');frame.to_csv(run/'validation_predictions.csv',index=False)
        row={'epoch':ep,'train_loss':total/steps,'seconds':time.time()-start,'selection_loss':value,'per_crop':scores['per_crop'],'best_validation':best,'learning_rate':opt.param_groups[0]['lr'],'audited_training_draws':len(ix)};history.append(row)
        write_json(run/'history.json',history);write_json(run/f'validation_epoch_{ep:02d}.json',scores);print(json.dumps(row),flush=True);sched.step()
        ck.update(optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,best=best,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(device))
        torch.save(ck,run/'last.tmp');(run/'last.tmp').replace(run/'last.pt')
        status('training',epoch=ep,epochs=epochs,epoch_complete=True,latest_validation=scores['per_crop'])
    status('trained',epoch=epochs,best_validation=best)


def evaluate(arm,device):
    d=Data(arm);seed(arm,d.plan['seed']);run=ROOT;ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);m=model(arm,ck.get('architecture',d.plan['architecture'])).to(device);m.load_state_dict(ck['model']);m.eval()
    shared.WORK=ROOT;(run/'predictions').mkdir(exist_ok=True);p=d.plan
    order=list(dict.fromkeys(p['scenario_groups']['realistic_holdout']+p['scenario_groups']['synthetic_holdout']+list(range(198))))
    frames=[];started=time.time()
    for c in order:
        sc=p['scenarios'][c];path=run/'predictions'/f'{sc["key"]}.npz'
        if path.exists():
            with np.load(path) as a:frame=shared.pd.DataFrame(dict(a))
        else:
            ix=d.indices(p['test_sites'],[c]);dl=loader(d,ix,256,2);frame=shared.predict(m,dl,d,device);del dl
            with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**{k:frame[k].to_numpy() for k in frame.columns})
            path.with_suffix('.tmp').replace(path)
        frames.append(frame);write_json(run/'evaluation_status.json',{'state':'running','arm':arm,'completed':len(frames),'total':198,'latest':sc['key'],'checkpoint_epoch':ck['epoch'],'updated_epoch':time.time(),'elapsed_seconds':time.time()-started})
        if len(frames)==22 or len(frames)%25==0:shared.write_reports(arm,frames,d)
        print('EVAL',arm,len(frames),sc['key'],flush=True)
    shared.write_reports(arm,frames,d);write_json(run/'evaluation_status.json',{'state':'complete','arm':arm,'completed':198,'total':198,'checkpoint_epoch':ck['epoch'],'updated_epoch':time.time(),'elapsed_seconds':time.time()-started})
    write_json(run/'status.json',{'state':'complete','arm':arm,'updated_epoch':time.time()})
