"""CPU-only adversarial confirmation fixtures; never launch a study or learner."""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.util
import json
import shutil
import statistics
import sys
import tempfile
import numpy as np

from independent_analyzer_tests import encode

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    aggregate = module('tested_confirmation', ROOT/'analyze_confirmation.py')
    controller = module('tested_controller', ROOT/'confirm.py')
    freeze = module('tested_freezer', ROOT/'freeze_confirmation.py')
    import analyze
    initial = json.loads((ROOT/'plan.json').read_text())
    extension = json.loads((ROOT/'extension-plan.json').read_text())
    original_manifest = json.loads((ROOT/extension['input_manifest']).read_text())
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='sutro-independent-confirmation-') as temporary:
        root = Path(temporary)
        for name in ['canonical', 'sources', 'inputs']:
            shutil.copytree(ROOT/name, root/name)
        for name in ['study.py', 'input_bundles.py', 'plan.json']:
            shutil.copyfile(ROOT/name, root/name)
        seeds = list(range(2610222101, 2610222112))
        warm = 2610222190
        manifest = copy.deepcopy(original_manifest)
        manifest.update(entries={}, local_archives={})
        for dataset, seed in [('fashion', warm)]+[('mnist',s) for s in seeds]:
            # Synthetic schema fixture: reuse known archived arrays, never claim these are real fresh draws.
            oldkey = 'fashion:2610221190' if dataset=='fashion' else 'mnist:2610221101'
            meta = copy.deepcopy(original_manifest['entries'][oldkey])
            meta.update(seed=seed, key=f"v1:{initial['scorer_sha256']}:{dataset}:{seed}")
            label = f'{dataset}:{seed}'
            manifest['entries'][label]=meta
            manifest['local_archives'][label]=original_manifest['local_archives'][oldkey]
        manifest_path = root/'fresh-manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        mh = sha(manifest_path)
        specs = [('selected', ['selected','best_regularized_control'], extension['arms'][0]),
                 ('baseline450',['baseline450'],initial['arms'][0]),
                 ('best_regularized',['best_regularized'], next(a for a in extension['arms'] if a['name']=='noguard_dropout_01_7200'))]
        arms = []
        for name,roles,original in specs:
            p=copy.deepcopy(original);p.update(name=name,roles=roles,members=1,prefixes=[1])
            if 'source_path' not in p:
                p['source_path']='sources/'+original['name']+'.py'
            arms.append(p)
        plan = {'phase':'confirmation','frozen':True,'save_probabilities':True,
                'draw_seeds':seeds,'warmup_seed':warm,'shards':[seeds[:3],seeds[3:6],seeds[6:9],seeds[9:]],
                'max_fit_seconds':300,'max_concurrent_containers':4,'arms':arms,
                'scorer_sha256':initial['scorer_sha256'],'runner_sha256':initial['runner_sha256'],
                'study_producer_sha256':sha(root/'study.py'),'input_bundle_helper_sha256':sha(root/'input_bundles.py'),
                'input_manifest':'fresh-manifest.json','input_manifest_sha256':mh,
                'acceptance':{'accuracy_target_pct':98.6,'required_correct':108460,'total':110000,'every_draw_cumulative_algorithm_seconds_max':300},
                'matched_control_audit':{'comparisons':[{'regularized_role':'best_regularized','control_role':'best_regularized_control'}]}}
        plan_path = root/'confirmation-plan.json';plan_path.write_text(json.dumps(plan))
        launch = {'plan_sha256':sha(plan_path),'input_manifest_sha256':mh,'input_manifest_path':str(manifest_path)}
        (root/'confirmation-controller-status.json').write_text(json.dumps(launch))
        jobs = controller.prepare_jobs(root, plan, manifest)
        assert len(jobs)==12
        assert all(j['runner_sha256']==plan['runner_sha256'] and len(j['data_bundles'])==1+len(j['draw_seeds']) for j in jobs)
        outcomes.append({'case':'three_algorithms_four_shards_job_schema','passed':True,'jobs':12,'roles':4})
        (root/'confirmation').mkdir()
        records = {}
        truth=np.load(root/manifest['local_archives'][f'mnist:{seeds[0]}'],allow_pickle=False)['qy'].tolist()
        for arm in arms:
            for index, shard in enumerate(plan['shards'],1):
                name=f"{arm['name']}__shard{index}"
                raw={'arm':name,'phase':'confirmation','parameters':arm,'source_sha256':arm['source_sha256'],
                     'scorer_sha256':plan['scorer_sha256'],'runner_sha256':plan['runner_sha256'],
                     'input_manifest_sha256':mh,'data_bundle_dict':manifest['dict_name'],
                     'warmup_bundle':manifest['entries'][f'fashion:{warm}'],'completed':True,'rows':[]}
                for seed in shard:
                    position=seeds.index(seed)
                    misses=(120+2*position) if arm['name']=='selected' else 150 if arm['name']=='baseline450' else (110+2*position)
                    prediction=list(truth)
                    for i in range(misses):prediction[i]=(prediction[i]+1)%10
                    probs=np.full((10000,10),np.float32(.01));probs[np.arange(10000),prediction]=np.float32(.91)
                    meta=manifest['entries'][f'mnist:{seed}']
                    member={'member':1,'model_seed':11,'prediction':list(prediction),'correct':10000-misses,'total':10000,
                            'training_errors':0,'training_total':10000,'full_call_seconds':1.,
                            'probability_accumulation_argmax_seconds':.01,'algorithm_seconds':1.01,**encode(probs)}
                    raw['rows'].append({'draw_seed':seed,'data_bundle_key':meta['key'],'data_bundle_npz_sha256':meta['npz_sha256'],
                                        'input_manifest_sha256':mh,'source_release_sha256':plan['scorer_sha256'],
                                        'array_sha256':{k:v['sha256'] for k,v in meta['arrays'].items()},
                                        'train_sha256':meta['train_sha256'],'query_sha256':meta['query_sha256'],
                                        'test_labels':list(truth),'members':[member],
                                        'prefixes':[{'members':1,'prediction':list(prediction),'correct':10000-misses,'total':10000,
                                                     'accuracy_pct':100*(10000-misses)/10000,'full_call_seconds':1.,'cumulative_algorithm_seconds':1.01}]})
                records[name]=raw
                (root/'confirmation'/(name+'.json')).write_text(json.dumps(raw))
        target=root/'confirmation/selected__shard4.json'

        def audit(name, mutate=None, valid=False, setup=None, cleanup=None):
            record=copy.deepcopy(records['selected__shard4'])
            if mutate:mutate(record)
            target.write_text(json.dumps(record))
            if setup:setup()
            report=aggregate.audit(root, manifest_name='fresh-manifest.json')
            assert report['all_arms_valid']==valid,(name,report['global_errors'])
            selected=next(a for a in report['arms'] if a['name']=='selected')
            if valid:
                assert selected['valid_confirmation'] and selected['target_met'] is True
                assert selected['observed_total']==110000 and selected['pooled_correct']==108570
                assert selected['accuracy_draw_interval']['n_draws']==11
                assert report['role_to_algorithm']['best_regularized_control']=='selected'
                assert len(report['comparisons'][0]['paired_difference_by_draw'])==11
            else:
                assert not selected['valid_confirmation'] and selected['target_met'] is None
            outcomes.append({'case':name,'passed':True,'all_arms_valid':report['all_arms_valid'],
                             'global_errors':report['global_errors'],'selected_errors':selected['errors']})
            if cleanup:cleanup()

        audit('valid_frozen_eleven_draw_confirmation_and_role_aliases',valid=True)
        audit('missing_confirmation_draw',lambda x:x['rows'].pop())
        audit('duplicate_confirmation_draw',lambda x:x['rows'].__setitem__(1,copy.deepcopy(x['rows'][0])))
        audit('missing_frozen_prefix',lambda x:x['rows'][0]['prefixes'].clear())
        def overtime(x):
            x['rows'][0]['members'][0].update(algorithm_seconds=301.,probability_accumulation_argmax_seconds=300.)
            x['rows'][0]['prefixes'][0]['cumulative_algorithm_seconds']=301.
        audit('over300_algorithm_despite_fast_fit',overtime)
        audit('query_truth_drift',lambda x:x['rows'][0]['test_labels'].__setitem__(0,(x['rows'][0]['test_labels'][0]+1)%10))
        audit('coordinate_hash_drift',lambda x:x['rows'][0].update(train_sha256='0'*64))
        audit('source_identity_drift',lambda x:x.update(source_sha256='0'*64))
        def no_probs(x):
            for k in list(x['rows'][0]['members'][0]):
                if k.startswith('probabilities'):x['rows'][0]['members'][0].pop(k)
        audit('required_probability_missing',no_probs)
        def bad_probability(x):
            p=np.full((10000,10),np.float32(.01));p[:,0]=np.float32(.91)
            x['rows'][0]['members'][0].update(encode(p))
        audit('probability_argmax_mismatch',bad_probability)
        audit('new_cumulative_clock_missing',lambda x:x['rows'][0]['prefixes'][0].pop('cumulative_algorithm_seconds'))
        audit('warmup_bundle_identity_drift',lambda x:x['warmup_bundle'].update(npz_sha256='0'*64))
        audit('missing_whole_shard',setup=lambda:target.unlink())
        for filename in ['canonical/mnist.py','canonical/run_modal.py','study.py','input_bundles.py']:
            p=root/filename;data=p.read_bytes()
            audit('frozen_'+filename.replace('/','_')+'_drift',setup=lambda p=p,data=data:p.write_bytes(data+b'\n'),cleanup=lambda p=p,data=data:p.write_bytes(data))
        source=root/arms[0]['source_path'];data=source.read_bytes()
        audit('frozen_learner_file_drift',setup=lambda:source.write_bytes(data+b'\n'),cleanup=lambda:source.write_bytes(data))
        # A known draw-level interval, independently calculated using sample SD and eleven observations.
        interval=aggregate.draw_interval(list(range(11)))
        expected_half=2.2281388519649385*statistics.stdev(range(11))/(11**.5)
        assert interval['mean']==5 and interval['n_draws']==11
        assert interval['approximate_95pct_t_interval']==[5-expected_half,5+expected_half]
        outcomes.append({'case':'draw_level_t_interval_not_query_binomial','passed':True,'n':11})
        # Test freezer aliases and one-shot protection on an isolated mocked finalized development audit.
        freezer_root=root/'freeze-fixture';freezer_root.mkdir()
        for name in ['sources','canonical']:shutil.copytree(ROOT/name,freezer_root/name)
        for name in ['plan.json','study.py','input_bundles.py']:shutil.copyfile(ROOT/name,freezer_root/name)
        selected=copy.deepcopy(extension['arms'][0]);regularized=copy.deepcopy(next(a for a in extension['arms'] if a['name']=='noguard_dropout_01_7200'));baseline=copy.deepcopy(initial['arms'][0])
        descriptions=[{'name':p['name'],'parameters':p,'source_path':p.get('source_path','sources/'+p['name']+'.py'),'raw_path':'synthetic/'+p['name']+'.json','raw_sha256':'0'*64} for p in [selected,regularized,baseline]]
        chosen={'arm':selected['name'],'members':1,'pooled_correct':29700,'total':30000}
        reg={'arm':regularized['name'],'members':1,'pooled_correct':29600,'total':30000}
        synthetic={'selection_locked':False,'selected':chosen,'arms':descriptions,'provisional_ranking':[chosen,reg], 'plans':[]}
        original_analyze=freeze.analyze.analyze
        freeze.analyze.analyze=lambda root:copy.deepcopy(synthetic)
        try:
            frozen,_=freeze.freeze(freezer_root)
            aliases={role:arm['name'] for arm in frozen['arms'] for role in arm['roles']}
            assert len(frozen['arms'])==3 and aliases['selected']==aliases['best_regularized_control']
            assert aliases['best_regularized']!=aliases['selected'] and aliases['baseline450']!=aliases['selected']
            try:freeze.freeze(freezer_root)
            except RuntimeError:pass
            else:raise AssertionError('A second winner freeze overwrote the first')
        finally:freeze.analyze.analyze=original_analyze
        outcomes.append({'case':'matched_control_alias_dedup_and_one_shot_freeze','passed':True,'physical_algorithms':3,'roles':4})
    report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','all_passed':True,
            'hashes':{name:sha(ROOT/name) for name in ['analyze.py','analyze_confirmation.py','confirm.py','freeze_confirmation.py']},
            'scope':'Disposable synthetic eleven-seed schemas, reused known archived arrays and synthetic prediction probabilities. No real fresh draws/results/source/plan modified, no learner execution, GPU or network call.',
            'cases':outcomes}
    (ROOT/'independent-confirmation-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':len(outcomes),'saved':str(ROOT/'independent-confirmation-fixtures.json')},indent=2))


if __name__=='__main__':main()
