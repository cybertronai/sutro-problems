"""Independent adversarial CPU fixtures; no learner import or GPU execution."""
from pathlib import Path
import base64
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import shutil
import tempfile
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def encode(p):
    raw = np.asarray(p, dtype='<f4').tobytes()
    return {'probabilities_codec': 'zlib-base64-f32-le', 'probabilities_shape': [10000, 10],
            'probabilities_sha256': hashlib.sha256(raw).hexdigest(),
            'probabilities': base64.b64encode(zlib.compress(raw)).decode()}


def main():
    analyzer = module('tested_analyzer', ROOT/'analyze.py')
    initial = json.loads((ROOT/'plan.json').read_text())
    extension = json.loads((ROOT/'extension-plan.json').read_text())
    manifest = json.loads((ROOT/extension['input_manifest']).read_text())
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='sutro-independent-analyzer-') as temporary:
        root = Path(temporary)
        shutil.copytree(ROOT/'canonical', root/'canonical')
        shutil.copytree(ROOT/'sources', root/'sources')
        shutil.copytree(ROOT/'inputs', root/'inputs')
        shutil.copyfile(ROOT/extension['input_manifest'], root/extension['input_manifest'])
        for p, name in [(initial, 'plan.json'), (extension, 'extension-plan.json')]:
            (root/name).write_text(json.dumps(p))
            directory = root/('development' if p['phase']=='development' else 'development-extension')
            directory.mkdir()
            for arm in p['arms']:
                raw = {'arm': arm['name'], 'phase': p['phase'], 'parameters': arm,
                       'source_sha256': arm['source_sha256'], 'scorer_sha256': p['scorer_sha256'],
                       'runner_sha256': p['runner_sha256'], 'completed': False,
                       'error': 'Deliberate terminal fixture failure, never a real learner execution', 'rows': []}
                if p['phase']!='development':
                    raw.update(input_manifest_sha256=p['input_manifest_sha256'], data_bundle_dict=manifest['dict_name'],
                               warmup_bundle=manifest['entries'][f"fashion:{p['warmup_seed']}"])
                (directory/(arm['name']+'.json')).write_text(json.dumps(raw))
        # Only this candidate is a passing fixture; all other predeclared outcomes are terminal failures.
        target = extension['arms'][0]
        target.update(members=2, prefixes=[2])
        (root/'extension-plan.json').write_text(json.dumps(extension))
        raw = {'arm': target['name'], 'phase': extension['phase'], 'parameters': target,
               'source_sha256': target['source_sha256'], 'scorer_sha256': extension['scorer_sha256'],
               'runner_sha256': extension['runner_sha256'], 'completed': True, 'rows': [],
               'input_manifest_sha256': extension['input_manifest_sha256'], 'data_bundle_dict': manifest['dict_name'],
               'warmup_bundle': manifest['entries'][f"fashion:{extension['warmup_seed']}"]}
        for seed in extension['draw_seeds']:
            label = f'mnist:{seed}'
            meta = manifest['entries'][label]
            with np.load(root/manifest['local_archives'][label], allow_pickle=False) as arrays:
                truth = arrays['qy'].tolist()
            probs = np.full((10000, 10), np.float32(.01))
            probs[np.arange(10000), truth] = np.float32(.91)
            members = []
            for index in [1, 2]:
                members.append({'member': index, 'model_seed': 11+997*(index-1), 'prediction': list(truth),
                                'correct': 10000, 'total': 10000, 'training_errors': 0, 'training_total': 10000,
                                'full_call_seconds': 1., 'probability_accumulation_argmax_seconds': .01,
                                'algorithm_seconds': 1.01, **encode(probs)})
            raw['rows'].append({'draw_seed': seed, 'data_bundle_key': meta['key'],
                               'data_bundle_npz_sha256': meta['npz_sha256'], 'input_manifest_sha256': extension['input_manifest_sha256'],
                               'runner_sha256': extension['runner_sha256'], 'source_release_sha256': extension['scorer_sha256'],
                               'array_sha256': {k:v['sha256'] for k,v in meta['arrays'].items()},
                               'train_sha256': meta['train_sha256'], 'query_sha256': meta['query_sha256'],
                               'test_labels': list(truth), 'members': members,
                               'prefixes': [{'members': 2, 'prediction': list(truth), 'correct': 10000, 'total': 10000,
                                             'accuracy_pct': 100., 'full_call_seconds': 2., 'cumulative_algorithm_seconds': 2.02}]})
        path = root/'development-extension'/(target['name']+'.json')

        def audit(name, mutate=None, expected_selected=False, setup=None, cleanup=None):
            record = copy.deepcopy(raw)
            if mutate:
                mutate(record)
            path.write_text(json.dumps(record))
            if setup:
                setup()
            report = analyzer.analyze(root)
            selected = report['selected'] is not None
            assert selected == expected_selected, (name, report['selected'], report['global_errors'])
            if selected:
                assert report['selected']['arm'] == target['name'] and report['selected']['members'] == 2
            candidate = next(x for x in report['all_prefix_candidates'] if x['arm']==target['name'] and x['members']==2)
            outcomes.append({'case': name, 'passed': True, 'selection_available': selected,
                             'target_eligible': candidate['eligible'], 'reasons': candidate['reasons']})
            if cleanup:
                cleanup()

        audit('valid_complete_probability_ensemble', expected_selected=True)
        audit('missing_draw', lambda x: x['rows'].pop())
        audit('duplicate_draw', lambda x: x['rows'].__setitem__(2, copy.deepcopy(x['rows'][0])))
        audit('missing_selected_prefix', lambda x: x['rows'][2]['prefixes'].clear())
        def over_budget(x):
            for m in x['rows'][2]['members']:
                m['algorithm_seconds'] = 151.; m['probability_accumulation_argmax_seconds'] = 150.
            x['rows'][2]['prefixes'][0]['cumulative_algorithm_seconds'] = 302.
        audit('over_budget_despite_fast_fit_only_time', over_budget)
        audit('missing_new_algorithm_clock', lambda x: x['rows'][2]['prefixes'][0].pop('cumulative_algorithm_seconds'))
        audit('coordinate_hash_drift', lambda x: x['rows'][2].update(query_sha256='0'*64))
        audit('archive_identity_drift', lambda x: x['rows'][2].update(data_bundle_npz_sha256='0'*64))
        def truth_drift(x):
            x['rows'][2]['test_labels'][0] = (x['rows'][2]['test_labels'][0]+1)%10
        audit('query_truth_drift', truth_drift)
        def no_probs(x):
            for key in list(x['rows'][2]['members'][1]):
                if key.startswith('probabilities'):
                    x['rows'][2]['members'][1].pop(key)
        audit('required_probability_missing', no_probs)
        def bad_probabilities(x):
            p = np.full((10000,10), np.float32(.01)); p[:,0]=np.float32(.91)
            x['rows'][2]['members'][1].update(encode(p))
        audit('member_probability_argmax_mismatch', bad_probabilities)
        def bad_ensemble(x):
            p = x['rows'][2]['prefixes'][0]
            p['prediction'][0] = (p['prediction'][0]+1)%10
            p['correct'] = sum(a==b for a,b in zip(p['prediction'],x['rows'][2]['test_labels']))
            p['accuracy_pct'] = p['correct']/100
        audit('ensemble_argmax_mismatch_with_consistent_reported_counts', bad_ensemble)
        def bad_norm(x):
            member = x['rows'][2]['members'][1]
            data = zlib.decompress(base64.b64decode(member['probabilities']))
            member.update(encode(np.frombuffer(data,dtype='<f4').reshape(10000,10)*.5))
        audit('nonnormalized_probabilities', bad_norm)
        audit('raw_source_identity_drift', lambda x: x.update(source_sha256='0'*64))
        learner = root/target['source_path']; learner_bytes=learner.read_bytes()
        audit('frozen_source_file_drift', setup=lambda:learner.write_bytes(learner_bytes+b'\n'), cleanup=lambda:learner.write_bytes(learner_bytes))
        other = root/'development'/(initial['arms'][1]['name']+'.json'); other_bytes=other.read_bytes()
        audit('any_predeclared_outcome_missing_locks_selection', setup=lambda:other.unlink(), cleanup=lambda:other.write_bytes(other_bytes))
        # Compression may change archive file bytes but must preserve the exact raw JSON payload identity.
        def compress():
            Path(str(path)+'.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0));path.unlink()
        def uncompress():
            Path(str(path)+'.gz').unlink()
        audit('lossless_gzip_raw_retention', expected_selected=True, setup=compress, cleanup=uncompress)
        # A healthy rival must not become a selected winner by silently dropping corrupted evidence.
        baseline=initial['arms'][0]
        baseline_path=root/'development'/(baseline['name']+'.json')
        baseline_original=baseline_path.read_bytes()
        rival=copy.deepcopy(raw)
        rival.update(arm=baseline['name'],phase=initial['phase'],parameters=baseline,
                     source_sha256=baseline['source_sha256'])
        for row in rival['rows']:
            row['members']=row['members'][:1]
            row['prefixes'][0].update(members=1,full_call_seconds=1.,cumulative_algorithm_seconds=1.01)
        baseline_path.write_text(json.dumps(rival))
        path.write_text(json.dumps(raw))
        # Legitimate terminal execution failure may leave the independent healthy rival eligible.
        failed=copy.deepcopy(raw);failed.update(completed=False,error='Retained ordinary fixture execution failure')
        path.write_text(json.dumps(failed))
        report=analyzer.analyze(root)
        assert not report['selection_locked'] and report['selected']['arm']==baseline['name']
        outcomes.append({'case':'ordinary_execution_failure_does_not_lock_healthy_candidate','passed':True})
        path.write_text(json.dumps(raw))
        learner.write_bytes(learner_bytes+b'\n')
        report=analyzer.analyze(root)
        assert report['selection_locked'] and report['selected'] is None
        outcomes.append({'case':'corrupt_source_globally_locks_even_with_healthy_rival','passed':True})
        learner.write_bytes(learner_bytes)
        baseline_path.write_bytes(baseline_original)
        path.write_text(json.dumps(raw))
        early_path=root/'development-extension'/(extension['arms'][1]['name']+'.json')
        early_original=early_path.read_bytes()
        early=json.loads(early_original);early.pop('warmup_bundle')
        early['error']='Legitimate fixture failure before warmup bundle returned'
        early_path.write_text(json.dumps(early))
        report=analyzer.analyze(root)
        assert not report['selection_locked'] and report['selected']['arm']==target['name']
        outcomes.append({'case':'early_execution_failure_without_warmup_not_integrity_corruption','passed':True})
        early_path.write_bytes(early_original)
    report = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'analyzer_sha256': sha(ROOT/'analyze.py'), 'status': 'PASS', 'all_passed': True,
              'scope': 'Disposable synthetic terminal-outcome fixtures using real archived canonical inputs and source identity. No real outcome, plan, source or learner execution changed; no GPU or network call.',
              'cases': outcomes}
    (ROOT/'independent-analyzer-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':len(outcomes),'saved':str(ROOT/'independent-analyzer-fixtures.json')},indent=2))


if __name__=='__main__':
    main()
