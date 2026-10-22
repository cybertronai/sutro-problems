"""CPU-only fresh-confirmation counts, FP32 ensemble and paired draw-level audit."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics

import analyze

ROOT = Path(__file__).resolve().parent
T95_DF10 = 2.2281388519649385


def draw_interval(values):
    if len(values) != 11:
        return None
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF10 * sd/(11**.5)
    return {'n_draws':11,'mean':mean,'sample_sd':sd,'t_critical_df10':T95_DF10,
            'approximate_95pct_t_interval':[mean-half,mean+half],
            'scope':'Draw-level t interval conditional on this fixed MNIST pool and frozen procedure; overlapping query images are not110000 independent Bernoulli trials. No universal accuracy ceiling.'}


def audit(root=ROOT, plan_name='confirmation-plan.json', manifest_name=None):
    plan = analyze.load(root/plan_name)
    launch_path = root/'confirmation-controller-status.json'
    launch = analyze.load(launch_path) if launch_path.exists() else {}
    if manifest_name is None:
        manifest_name = launch.get('input_manifest_path') or plan.get('input_manifest') or plan.get('fresh_inputs',{}).get('input_manifest')
    if not manifest_name:
        raise ValueError('Supply the frozen fresh input manifest, or record its path in launch status')
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = root/manifest_path
    digest = analyze.sha(manifest_path)
    expected_digest = launch.get('input_manifest_sha256') or plan.get('input_manifest_sha256') or plan.get('fresh_inputs',{}).get('input_manifest_sha256')
    attached = dict(plan,input_manifest=str(manifest_path),input_manifest_sha256=expected_digest)
    manifest, manifest_errors, archive_audits = analyze.audit_input_manifest(root,attached)
    global_errors = list(manifest_errors)
    for path,key in [('canonical/mnist.py','scorer_sha256'),('canonical/run_modal.py','runner_sha256'),('study.py','study_producer_sha256'),('input_bundles.py','input_bundle_helper_sha256')]:
        if not (root/path).exists() or analyze.sha(root/path) != plan.get(key):
            global_errors.append('Frozen producer/helper/canonical hash mismatch: '+path)
    if plan.get('phase')!='confirmation' or plan.get('frozen') is not True or plan.get('draw_seeds')!=list(range(2610222101,2610222112)) or [len(s) for s in plan.get('shards',[])]!=[3,3,3,2] or [s for shard in plan.get('shards',[]) for s in shard]!=plan.get('draw_seeds'):
        global_errors.append('Invalid frozen phase/fresh seed/shard semantics')
    roles=[role for arm in plan['arms'] for role in arm['roles']]
    if len(roles)!=len(set(roles)) or not {'selected','baseline450'}<=set(roles):
        global_errors.append('Missing or duplicated required confirmation roles')
    if plan.get('max_fit_seconds')!=300 or plan.get('max_concurrent_containers')!=4 or plan.get('acceptance',{}).get('required_correct')!=108460 or plan.get('acceptance',{}).get('total')!=110000:
        global_errors.append('Invalid frozen budget/target semantics')
    for path,expected in plan.get('development_raw_sha256',{}).items():
        if not analyze.existing(root/path).exists() or analyze.sha(root/path)!=expected:
            global_errors.append('Frozen development outcome changed: '+path)
    if not expected_digest or digest != expected_digest:
        global_errors.append('Fresh manifest attachment hash absent or mismatched')
    if launch.get('plan_sha256') != analyze.sha(root/plan_name):
        global_errors.append('Launch/frozen plan hash mismatch or absent')
    if not plan.get('save_probabilities'):
        global_errors.append('Frozen confirmation must require probabilities')
    references, arm_reports, role_map = {}, [], {}
    for arm in plan['arms']:
        for role in arm['roles']:
            role_map[role] = arm['name']
        errors, shards, rows = [], [], []
        source = root/arm['source_path']
        if analyze.sha(source) != arm['source_sha256']:
            errors.append('Frozen source hash mismatch')
        for index,seeds in enumerate(plan['shards'],1):
            path = analyze.existing(root/'confirmation'/f'{arm["name"]}__shard{index}.json')
            if not path.exists():
                shards.append({'index':index,'finalized':False,'path':str(path.relative_to(root))})
                errors.append(f'Awaiting shard{index}')
                continue
            raw = analyze.load(path)
            se = []
            for key,expected in [('arm',f'{arm["name"]}__shard{index}'),('parameters',arm),('phase','confirmation'),('source_sha256',arm['source_sha256']),('scorer_sha256',plan['scorer_sha256']),('runner_sha256',plan['runner_sha256']),('input_manifest_sha256',digest),('data_bundle_dict',manifest['dict_name'])]:
                if raw.get(key) != expected:
                    se.append('Raw '+key+' mismatch')
            if (raw.get('completed') is True or 'warmup_bundle' in raw) and raw.get('warmup_bundle') != manifest['entries'][f'fashion:{plan["warmup_seed"]}']:
                se.append('Warmup bundle identity mismatch')
            if not raw.get('completed'):
                se.append('Failed/incomplete execution: '+str(raw.get('error','No error supplied')))
            if [r.get('draw_seed') for r in raw.get('rows',[])] != seeds:
                se.append('Missing, duplicated or reordered shard draws')
            for row in raw.get('rows',[]):
                seed = row.get('draw_seed')
                if analyze.labels_ok(row.get('test_labels')):
                    references.setdefault(seed,row)
                expected = manifest.get('entries',{}).get(f'mnist:{seed}')
                if expected is None:
                    se.append('Unexpected confirmation seed')
                    continue
                result = analyze.audit_row(row,arm,references,expected,digest,plan['scorer_sha256'],True,True)
                errors.extend(f'draw{seed}: '+e for e in result['errors'])
                ps = [p for p in result['prefixes'] if p['members']==arm['members']]
                if len(ps)!=1:
                    errors.append(f'draw{seed}: frozen prefix absent')
                else:
                    errors.extend(f'draw{seed}: '+e for e in ps[0]['errors'])
                    if not ps[0]['within_budget']:
                        errors.append(f'draw{seed}: cumulative algorithm exceeds300seconds')
                rows.append(result)
            errors.extend(f'shard{index}: '+e for e in se)
            shards.append({'index':index,'path':str(path.relative_to(root)),'raw_sha256':analyze.sha(path),
                           'raw_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'finalized':type(raw.get('completed')) is bool,'completed':raw.get('completed'),'errors':se})
        prefixes=[]
        for row in rows:
            ps=[p for p in row['prefixes'] if p['members']==arm['members']]
            if len(ps)==1:
                prefixes.append({'draw_seed':row['seed'],**ps[0]})
        if sorted(r['seed'] for r in rows)!=plan['draw_seeds']:
            errors.append('Confirmation requires each of11 frozen draws exactly once')
        valid = not errors and not global_errors and len(prefixes)==11
        correct=sum(p['rebuilt_correct'] for p in prefixes);total=sum(p['total'] for p in prefixes)
        accuracies=[p['accuracy_pct'] for p in prefixes]
        arm_reports.append({'name':arm['name'],'roles':arm['roles'],'parameters':arm,'shards':shards,'rows':rows,'prefix_results':prefixes,
                            'all_shards_finalized':all(s['finalized'] for s in shards),'valid_confirmation':valid,'errors':sorted(set(errors)),
                            'pooled_correct':correct,'observed_total':total,'observed_accuracy_pct':100*correct/total if total else None,
                            'target_met':correct>=108460 if valid else None,'accuracy_draw_interval':draw_interval(accuracies) if valid else None,
                            'mean_full_call_seconds':statistics.mean(p['full_call_seconds'] for p in prefixes) if prefixes else None,
                            'mean_cumulative_algorithm_seconds':statistics.mean(p['budget_seconds'] for p in prefixes) if prefixes else None,
                            'max_cumulative_algorithm_seconds':max((p['budget_seconds'] for p in prefixes),default=None)})
    byname={a['name']:a for a in arm_reports};comparisons=[]
    requested=[('selected','baseline450')]
    requested += [(p['regularized_role'],p['control_role']) for p in plan['matched_control_audit']['comparisons']]
    for left,right in requested:
        a,b=byname[role_map[left]],byname[role_map[right]]
        paired=None;differences=[]
        if a['valid_confirmation'] and b['valid_confirmation']:
            amap={p['draw_seed']:p for p in a['prefix_results']};bmap={p['draw_seed']:p for p in b['prefix_results']}
            differences=[{'draw_seed':seed,'correct_difference':amap[seed]['rebuilt_correct']-bmap[seed]['rebuilt_correct'],
                          'accuracy_difference_pp':amap[seed]['accuracy_pct']-bmap[seed]['accuracy_pct']} for seed in plan['draw_seeds']]
            paired=draw_interval([d['accuracy_difference_pp'] for d in differences])
        comparisons.append({'left_role':left,'right_role':right,'same_deduped_algorithm':a['name']==b['name'],
                            'paired_difference_by_draw':differences,'paired_accuracy_difference_pp_interval':paired})
    return {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'confirmation',
            'plan_sha256':analyze.sha(root/plan_name),'input_manifest_sha256':digest,'global_errors':global_errors,
            'input_archive_audit':archive_audits,'all_shards_finalized':all(a['all_shards_finalized'] for a in arm_reports),
            'all_arms_valid':all(a['valid_confirmation'] for a in arm_reports),'arms':arm_reports,'role_to_algorithm':role_map,'comparisons':comparisons,
            'scope':'Frozen fresh11-draw confirmation, including predeclared best-regularized secondary comparison when applicable. No source, prefix or model reselection after fresh data. Labels/counts, immutable input archives and ordered FP32 probability aggregation audited; not official canonical qualification or an upper-bound proof.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--plan',default='confirmation-plan.json')
    parser.add_argument('--input-manifest')
    args=parser.parse_args()
    result=audit(args.root,args.plan,args.input_manifest)
    (args.root/'confirmation-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_shards_finalized':result['all_shards_finalized'],'all_arms_valid':result['all_arms_valid'],
                      'roles':result['role_to_algorithm'],'errors':result['global_errors']},indent=2))
