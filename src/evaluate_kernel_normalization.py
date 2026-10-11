#!/usr/bin/env python3
"""Instrument frozen terminal SK banks after fidelity checks, never retrain."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from seg3d.data import BraTSVolumeDataset
from seg3d.model_zoo import build_model
from seg3d.reporting import case_region_metrics
from kernel_normalization_diagnostics import Capture, channel_diagnostics, cpu_copy, rms

REGIONS=('wt','tc','et')


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def forward(model,image,target,position,branch,reroute,precision,observed):
    model.clear_intervention()
    sel=model.multi_kernel_selectors[position]
    sel.set_intervention('drop_branch',branch=branch,renormalize=reroute)
    dtype=torch.bfloat16 if precision=='bf16' else torch.float32
    with Capture(sel) if observed else nullcontext() as cap:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            logits=model(image.to(dtype=dtype))
    # Preserve the original native-dtype sigmoid threshold; casting first changes
    # bf16 values around 0.5 and would create a false reproduction failure.
    masks=cpu_copy(logits.sigmoid()>=.5)
    near=cpu_copy((logits.sigmoid()-.5).abs()<=.01)
    metrics=[case_region_metrics(logits[i],target[i]) for i in range(len(image))]
    weights=[cpu_copy(s.selection_snapshot()['channel_weights']) for s in model.multi_kernel_selectors]
    result={'logits':cpu_copy(logits),'masks':masks,'near':near,'metrics':metrics,'weights':weights,
            'capture':cap.data if observed else None}
    return result


def metric_error(a,b):
    errors=[]
    for region in REGIONS:
        if bool(a['gt_positive_'+region]) != (b['gt_positive_'+region] in (True,'True')):
            raise ValueError('Reference eligibility mismatch')
        left,right=a['dice_'+region],b['dice_'+region]
        if left is not None:
            errors.append(abs(float(left)-float(right)))
        elif right not in (None,''):
            raise ValueError('Empty-reference Dice mismatch')
    return max(errors,default=0.)


def load_sources(base,expected):
    result={};hashes={}
    for reroute,folder in [(False,'stage_selector_interventions_final_epoch'),(True,'stage_selector_interventions_reroute_final_epoch')]:
        path=base/'metrics'/folder/'cases.csv';hashes[str(path)]=digest(path);rows={}
        with path.open() as f:
            for r in csv.DictReader(f):
                key=(r['condition'],r['case_id'])
                if key in rows or r['checkpoint']!='checkpoints/latest.pt':raise ValueError('Duplicate/nonterminal source')
                if r['renormalize']!=str(reroute):raise ValueError('Wrong source policy')
                rows[key]=r
        conditions={k[0] for k in rows}
        if len(conditions)!=13 or any({c for cond,c in rows if cond==name}!=expected for name in conditions):raise ValueError('Incomplete source grid')
        result[reroute]=rows
    return result,hashes


def source_condition(stage,branch,reroute):
    return f'drop_decoder_{stage}_{branch}'+('_and_reroute' if reroute else '')


def pair_case_rows(a,b,ids,stage,branch,precision):
    for n,case in enumerate(ids):
        delta=b['logits'][n].double()-a['logits'][n].double()
        row={'case_id':case,'patient_id':case.rsplit('-',1)[0],'stage':stage,'branch':branch,'precision':precision,
             'logit_difference_rms':rms(delta),'logit_difference_max':float(delta.abs().max())}
        for i,region in enumerate(REGIONS):
            for prefix,pack in [('drop',a),('renorm',b)]:
                row[prefix+'_dice_'+region]=pack['metrics'][n]['dice_'+region]
                row[prefix+'_predicted_voxels_'+region]=pack['metrics'][n]['predicted_voxels_'+region]
            row['gt_positive_'+region]=a['metrics'][n]['gt_positive_'+region]
            row['mask_disagreement_'+region]=float((a['masks'][n,i]!=b['masks'][n,i]).float().mean())
            row['drop_near_threshold_'+region]=float(a['near'][n,i].float().mean())
            row['renorm_near_threshold_'+region]=float(b['near'][n,i].float().mean())
        for index,(x,y) in enumerate(zip(a['weights'],b['weights'])):
            row['selector_'+str(index)+'_weight_rms']=rms(y[n].double()-x[n].double())
        yield row


def evaluate(root,base,out,selection,checkpoint_sha):
    if out.exists():raise FileExistsError('Distinct attempt output required')
    out.mkdir(parents=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    config=json.loads((base/'config.json').read_text());run=json.loads((base/'run.json').read_text())
    fold=int(run['fold']);checkpoint=base/'checkpoints/latest.pt'
    if digest(checkpoint)!=checkpoint_sha:raise ValueError('Checkpoint changed after matrix gate')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['epoch']!=99 or state['config']!=config:raise ValueError('Nonterminal/incompatible checkpoint')
    model=build_model(config['model'],in_channels=4,out_channels=3,base_channels=config.get('base_channels',16),**config.get('model_kwargs',{})).cuda().eval()
    model.load_state_dict(state['model']);del state
    data=BraTSVolumeDataset(config['data_dir'],fold,'val')
    sources,source_hashes=load_sources(base,set(data.case_ids))
    specs=[(p,int(s),b) for p,s in enumerate(model.selector_decoder_indices) for b in model.multi_kernel_selectors[p].branch_names]
    selected=next(r for r in selection['selection'] if r['fold']==fold)
    subsets=selected['original_batch_indices']
    loader=DataLoader(data,batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    fidelity=[];baseline_checks=[]
    save(out/'progress.json',{'phase':'fidelity','fold':fold,'time':time.time()})
    # Each original-composition batch is checked for all interventions. Thresholds
    # depend only on repeated identical inference, never on mechanism outcomes.
    pilot=DataLoader(data,batch_sampler=[list(range(i*4,min(i*4+4,len(data)))) for i in subsets],num_workers=0,pin_memory=True)
    for bi,(image,target,ids) in zip(subsets,pilot):
        image=image.cuda();target=target.cuda()
        model.clear_intervention()
        baseline_metrics=[]
        for repeat_index in range(2):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                baseline_logits=model(image.to(torch.bfloat16))
            baseline_metrics.append([case_region_metrics(baseline_logits[i],target[i]) for i in range(len(ids))])
            del baseline_logits
        noise=max(metric_error(x,y) for x,y in zip(*baseline_metrics))
        error=max(metric_error(baseline_metrics[0][i],sources[False][('learned',case)]) for i,case in enumerate(ids))
        baseline_checks.append({'batch':bi,'repeat_dice_max':noise,'historical_dice_max':error})
        save(out/'learned_fidelity.json',{'checks':baseline_checks})
        if noise>1e-4 or error>max(1e-8,5*noise):raise ValueError('Learned baseline fidelity failed')
        for pos,stage,branch in specs:
            for renorm in (False,True):
                x=forward(model,image,target,pos,branch,renorm,'bf16',False)
                y=forward(model,image,target,pos,branch,renorm,'bf16',False)
                z=forward(model,image,target,pos,branch,renorm,'bf16',True)
                noise=float((x['logits'].float()-y['logits'].float()).abs().max())
                observer=float((x['logits'].float()-z['logits'].float()).abs().max())
                repeat=max(metric_error(a,b) for a,b in zip(x['metrics'],y['metrics']))
                hook=max(metric_error(a,b) for a,b in zip(x['metrics'],z['metrics']))
                historical=max(metric_error(z['metrics'][i],sources[renorm][(source_condition(stage,branch,renorm),case)]) for i,case in enumerate(ids))
                tolerance=max(1e-8,5*repeat)
                row={'batch':bi,'stage':stage,'branch':branch,'renormalize':renorm,'repeat_logit_max':noise,
                     'hook_logit_max':observer,'repeat_dice_max':repeat,'hook_dice_max':hook,
                     'historical_dice_max':historical,'dice_tolerance':tolerance}
                fidelity.append(row)
                save(out/'fidelity.json',{'status':'checking','checks':fidelity})
                if repeat>1e-4 or historical>tolerance or hook>tolerance or observer>max(1e-7,5*noise):
                    raise ValueError('Fidelity gate failed; inspect fidelity.json before mechanism interpretation')
                del x,y,z
    if not fidelity:raise ValueError('No engineering cases')
    tolerances={(r['stage'],r['branch'],r['renormalize']):max(t['dice_tolerance'] for t in fidelity if (t['stage'],t['branch'],t['renormalize'])==(r['stage'],r['branch'],r['renormalize'])) for r in fidelity}
    save(out/'fidelity.json',{'status':'passed','checks':fidelity,'boundary':'Native-dtype primary runtime and original batch composition; bounded repeat-noise tolerance.'})
    all_rows=[];channels_count=0;sensitivity_cache={};precision_rows=[]
    selected_set=set(selected['studies'])
    original_tf32={'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32}
    # FP32 uses batch one, paired with a fresh bf16 batch-one control. Preserve
    # the primary batch-four results separately rather than confound precision.
    modes=[('primary_bf16','bf16',loader),
           ('sensitivity_bf16','bf16',DataLoader(Subset(data,[i for i,c in enumerate(data.case_ids) if c in selected_set]),batch_size=1,num_workers=0)),
           ('sensitivity_fp32','fp32',DataLoader(Subset(data,[i for i,c in enumerate(data.case_ids) if c in selected_set]),batch_size=1,num_workers=0))]
    for label,precision,batches in modes:
        if label!='primary_bf16':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with (out/(label+'_channels.jsonl')).open('x') as channel_file, (out/(label+'_cases.jsonl')).open('x') as case_file:
            for bi,(image,target,ids) in enumerate(batches):
                image=image.cuda();target=target.cuda()
                for pos,stage,branch in specs:
                    a=forward(model,image,target,pos,branch,False,precision,True)
                    b=forward(model,image,target,pos,branch,True,precision,True)
                    if label=='primary_bf16':
                        for reroute,pack in [(False,a),(True,b)]:
                            for i,case in enumerate(ids):
                                e=metric_error(pack['metrics'][i],sources[reroute][(source_condition(stage,branch,reroute),case)])
                                if e>tolerances[(stage,branch,reroute)]:raise ValueError(f'Full-grid fidelity failure {case} stage={stage} branch={branch}: {e}')
                    for row in channel_diagnostics(a['capture'],b['capture'],ids):
                        channel_file.write(json.dumps({'stage':stage,'branch':branch,**row},allow_nan=False)+'\n');channels_count+=1
                    rows=list(pair_case_rows(a,b,ids,stage,branch,precision))
                    for row in rows:
                        row['mode']=label
                        case_file.write(json.dumps(row,allow_nan=False)+'\n')
                        all_rows.append(row)
                    for policy,pack in [('drop',a),('renorm',b)]:
                        for i,case in enumerate(ids):
                            key=(stage,branch,policy,case)
                            if label=='sensitivity_bf16':
                                sensitivity_cache[key]={k:pack[k][i] for k in ('logits','masks','metrics')}
                            elif label=='sensitivity_fp32':
                                old=sensitivity_cache.pop(key)
                                comparison={'case_id':case,'stage':stage,'branch':branch,'policy':policy,
                                            'batch_size':1,'fp32_minus_bf16_logit_rms':rms(pack['logits'][i].double()-old['logits'].double())}
                                for j,region in enumerate(REGIONS):
                                    comparison['mask_disagreement_'+region]=float((old['masks'][j]!=pack['masks'][i,j]).float().mean())
                                    comparison['bf16_dice_'+region]=old['metrics']['dice_'+region]
                                    comparison['fp32_dice_'+region]=pack['metrics'][i]['dice_'+region]
                                precision_rows.append(comparison)
                    del a,b
                channel_file.flush();case_file.flush()
                save(out/'progress.json',{'phase':label,'fold':fold,'batch_completed':bi+1,'total_batches':len(batches),'time':time.time()})
    cells=[]
    for label,_,_ in modes:
        for _,stage,branch in specs:
            rows=[r for r in all_rows if r['mode']==label and r['stage']==stage and r['branch']==branch]
            cell={'mode':label,'stage':stage,'branch':branch,'studies':len(rows)}
            for policy in ('drop','renorm'):
                scores=[]
                for region in REGIONS:
                    values=[r[policy+'_dice_'+region] for r in rows if r['gt_positive_'+region]]
                    cell[policy+'_dice_'+region]=sum(values)/len(values) if values else None
                    if values:scores.append(cell[policy+'_dice_'+region])
                cell[policy+'_region_macro_dice']=sum(scores)/3 if len(scores)==3 else None
            cell['mask_disagreement_mean']=sum(sum(r['mask_disagreement_'+g] for g in REGIONS)/3 for r in rows)/len(rows)
            cells.append(cell)
    expected=len(data)*12+len(selected_set)*12*2
    if len(all_rows)!=expected:raise ValueError('Incomplete output study grid')
    if sensitivity_cache or len(precision_rows)!=len(selected_set)*24:raise ValueError('Incomplete matched precision comparisons')
    save(out/'precision_comparison.json',{'comparisons':precision_rows,'boundary':'Both runtimes use batch one; primary runtime uses batch four separately.'})
    save(out/'cells.json',{'cells':cells,'primary_endpoint':'study mean within positive-reference region, equal regions','sensitivity_boundary':'Fixed patient subset; no aggregate macro when any region has no positive reference.'})
    if digest(checkpoint)!=checkpoint_sha:raise ValueError('Checkpoint changed during audit')
    summary={'status':'complete','fold':fold,'base_seed':config['seed'],'effective_seed':config['seed']+fold,
             'checkpoint_sha256':checkpoint_sha,'source_hashes':source_hashes,'primary_studies':len(data),
             'sensitivity_studies':selected['studies'],'primary_cells':12,'paired_case_rows':len(all_rows),
             'channel_rows':channels_count,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
             'primary_tf32_settings':original_tf32,'sensitivity_tf32_settings':{'matmul':False,'cudnn':False},
             'artifacts':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='progress.json'}}
    save(out/'complete.json',summary)
    save(out/'progress.json',{'phase':'complete','fold':fold,'time':time.time()})
    print(json.dumps({'status':'complete','fold':fold,'paired_case_rows':len(all_rows),'channels':channels_count}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fold-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    a=p.parse_args();evaluate(a.root,a.fold_dir,a.out,json.loads(a.selection.read_text()),a.checkpoint_sha256)
