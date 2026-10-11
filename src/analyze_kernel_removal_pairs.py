#!/usr/bin/env python3
"""Read-only, exact-study paired removal audit of a completed stage matrix."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

BRANCHES = ('k3', 'k5', 'dilated_k3', 'anisotropic')
REGIONS = ('wt', 'tc', 'et')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows, expected):
    result = {}
    for row in rows:
        if row['case_id'] in result:
            raise ValueError('Duplicate study')
        if row['checkpoint'] != 'checkpoints/latest.pt':
            raise ValueError('Nonterminal record')
        result[row['case_id']] = row
    if set(result) != expected:
        raise ValueError('Study membership mismatch')
    return result


def paired_metric(left, right):
    """Equal-region means over the same positive-reference studies."""
    if set(left) != set(right):
        raise ValueError('Unpaired studies')
    lm, rm, differences = [], [], []
    for region in REGIONS:
        lvalues, rvalues = [], []
        for case in sorted(left):
            a, b = left[case], right[case]
            flag = a['gt_positive_' + region]
            if flag not in ('True', 'False') or flag != b['gt_positive_' + region]:
                raise ValueError('Reference eligibility mismatch')
            if a['target_voxels_' + region] != b['target_voxels_' + region]:
                raise ValueError('Reference voxel count mismatch')
            if flag == 'True':
                x, y = float(a['dice_' + region]), float(b['dice_' + region])
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
                    raise ValueError('Invalid Dice')
                lvalues.append(x)
                rvalues.append(y)
        if not lvalues:
            raise ValueError('No eligible studies')
        lm.append(fmean(lvalues))
        rm.append(fmean(rvalues))
        differences.append(fmean(x-y for x, y in zip(lvalues, rvalues)))
    return fmean(lm), fmean(rm), fmean(differences)


def analyze(root, matrix, destination):
    if destination.exists():
        raise FileExistsError('Use a new output directory')
    summary_path = matrix / 'analysis/summary.json'
    summary = json.loads(summary_path.read_text())
    plan_path = matrix / 'queue/plan.json'
    if (matrix / 'queue/combined_plan.json').exists():
        plan_path = matrix / 'queue/combined_plan.json'
    plan = json.loads(plan_path.read_text())
    splits_path = root / 'data/processed/brats2023_gli_crop4/splits.json'
    splits = json.loads(splits_path.read_text())
    source_cells_path = matrix / 'analysis/stage_branch_cells.csv'
    with source_cells_path.open() as f:
        source_cells = list(csv.DictReader(f))
    cached = {(int(r['base_seed']), int(r['fold']), int(r['stage']), r['branch']): r for r in source_cells}
    seeds = plan['protocol']['base_seeds']
    expected_keys = {(s, f, d, b) for s in seeds for f in range(5) for d in (1,2,3) for b in BRANCHES}
    if set(cached) != expected_keys or len(source_cells) != len(expected_keys) or summary['models'] != len(seeds)*5:
        raise ValueError('Incomplete/duplicate source matrix')
    hashes = {str(q.relative_to(root)): sha(q) for q in (summary_path, plan_path, splits_path, source_cells_path)}
    cells = []
    for seed in seeds:
        cp = root / plan['configs'][str(seed)]
        config = json.loads(cp.read_text())
        hashes[str(cp.relative_to(root))] = sha(cp)
        for fold in range(5):
            queue = root / plan.get('state_queues', {}).get(str(seed), str((matrix/'queue').relative_to(root)))
            state_path = queue / f'seed{seed}_fold{fold}.json'
            if json.loads(state_path.read_text())['phase'] != 'complete':
                raise ValueError('Incomplete source run')
            base = root / config['results_root'] / config['dataset'] / config['trainer'] / f'fold_{fold}'
            expected = set(splits[fold]['val'])
            policies = {}
            for name, folder, reroute in [('drop','stage_selector_interventions_final_epoch',False), ('renorm','stage_selector_interventions_reroute_final_epoch',True)]:
                path = base / 'metrics' / folder / 'cases.csv'
                relative = str(path.relative_to(root))
                digest = sha(path)
                if summary['source_hashes'].get(relative) != digest:
                    raise ValueError('Source CSV differs from completed analysis: ' + relative)
                hashes[relative] = digest
                grouped = defaultdict(list)
                with path.open() as f:
                    for row in csv.DictReader(f):
                        if row['renormalize'] != str(reroute):
                            raise ValueError('Wrong policy metadata')
                        grouped[row['condition']].append(row)
                suffix = '_and_reroute' if reroute else ''
                names = {'learned'} | {f'drop_decoder_{d}_{b}{suffix}' for d in (1,2,3) for b in BRANCHES}
                if set(grouped) != names:
                    raise ValueError('Incomplete conditions')
                policies[name] = {k:indexed(v, expected) for k,v in grouped.items()}
            baseline, other, _ = paired_metric(policies['drop']['learned'],policies['renorm']['learned'])
            if abs(baseline-other) > 1e-12:
                raise ValueError('Learned baseline mismatch')
            for case in expected:
                for region in REGIONS:
                    a=policies['drop']['learned'][case]['dice_'+region]
                    b=policies['renorm']['learned'][case]['dice_'+region]
                    if a != b:
                        raise ValueError('Study-level learned baseline mismatch')
            for stage in (1,2,3):
                for branch in BRANCHES:
                    name = f'drop_decoder_{stage}_{branch}'
                    left, right = policies['drop'][name], policies['renorm'][name+'_and_reroute']
                    # Check intervention eligibility against the baseline, too.
                    paired_metric(policies['drop']['learned'], left)
                    d, r, paired_difference = paired_metric(left,right)
                    old = cached[(seed,fold,stage,branch)]
                    if any(abs(float(old[k])-v)>1e-12 for k,v in [('drop_effect',baseline-d),('renormalized_drop_effect',baseline-r)]):
                        raise ValueError('Recomputed effect differs from reported cell')
                    if abs(((baseline-r)-(baseline-d))-paired_difference)>1e-12:
                        raise ValueError('Paired difference/contrast mismatch')
                    cells.append(dict(base_seed=seed,fold=fold,stage=stage,branch=branch,
                        studies=len(expected),learned_dice=baseline,drop_dice=d,renormalized_drop_dice=r,
                        drop_effect=baseline-d,renormalized_drop_effect=baseline-r,
                        renorm_minus_drop_effect=paired_difference,abs_policy_difference=abs(paired_difference)))
    stages=[]
    for stage in (1,2,3):
        rows=[c for c in cells if c['stage']==stage]
        values=[c['abs_policy_difference'] for c in rows]
        stages.append(dict(stage=stage,cells=len(rows),median_absolute_difference=median(values),
                           maximum_absolute_difference=max(values),mean_absolute_difference=fmean(values)))
    winners=[]
    for seed in seeds:
        for fold in range(5):
            for stage in (1,2,3):
                rows=[r for r in cells if (r['base_seed'],r['fold'],r['stage'])==(seed,fold,stage)]
                drop=max(rows,key=lambda r:r['drop_effect'])
                renorm=max(rows,key=lambda r:r['renormalized_drop_effect'])
                winners.append(dict(base_seed=seed,fold=fold,stage=stage,drop_max_branch=drop['branch'],
                    renorm_max_branch=renorm['branch'],same_max_branch=drop['branch']==renorm['branch']))
    result=dict(models=summary['models'],cells=len(cells),base_seeds=seeds,
        endpoint='Within each region, study-mean Dice over positive-reference studies; equal WT/TC/ET mean. Difference = renormalized removal effect minus unnormalized removal effect.',
        stage_summaries=stages,winners=winners,largest_differences=sorted(cells,key=lambda r:-r['abs_policy_difference'])[:12],
        source_hashes=hashes,analysis_code_sha256=sha(Path(__file__)),
        validation='All study identities, reference counts/eligibility, terminal path and policy labels checked; input CSV hashes match completed matrix; 120/180 primary cells rebuilt against its table.',
        boundary='Post-review descriptive analysis. Cells/seeds are dependent, not new cohorts. No tensor mechanism, precision effect, clinical threshold or causal explanation established.')
    destination.mkdir(parents=True)
    for name, rows in [('paired_cells.csv',cells),('winner_pairs.csv',winners)]:
        with (destination/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (destination/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Paired removal-policy effects','',result['endpoint'],'',
           '| Stage | Cells | Median absolute Dice difference | Maximum |','|---|---:|---:|---:|']
    lines += [f"| {r['stage']} | {r['cells']} | {r['median_absolute_difference']:.8f} | {r['maximum_absolute_difference']:.8f} |" for r in stages]
    lines += ['', 'Largest paired differences:', '', '| Seed | Fold | Stage | Branch | Drop effect | Renorm effect | Difference |', '|---:|---:|---:|---|---:|---:|---:|']
    lines += [f"| {r['base_seed']} | {r['fold']} | {r['stage']} | {r['branch']} | {r['drop_effect']:.8f} | {r['renormalized_drop_effect']:.8f} | {r['renorm_minus_drop_effect']:.8f} |" for r in result['largest_differences']]
    lines += ['',result['boundary'],'']
    (destination/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'models':result['models'],'cells':len(cells),'stages':stages,'largest':result['largest_differences'][0]},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    analyze(args.root.resolve(),(args.root/args.matrix).resolve(),(args.root/args.out).resolve())
