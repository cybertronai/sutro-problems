"""Frozen paired D6 development screen. Research only; no canonical score claims."""
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import input_bundles
spec = importlib.util.spec_from_file_location("canonical_runner", ROOT / "canonical/run_modal.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
image = (runner.image.add_local_dir(ROOT / "canonical", "/root/canonical", copy=True)
         .add_local_file(ROOT / "input_bundles.py", "/root/input_bundles.py", copy=True))
app = modal.App("sutro-d6-accuracy-20261022", image=image)
results = modal.Dict.from_name("sutro-d6-accuracy-20261022-results", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=3600, max_containers=4, single_use_containers=True)
def screen(job):
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    import time
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from input_bundles import decode_arrays
    bundle_store = modal.Dict.from_name(job["data_bundle_dict"])
    def draw_bundle(dataset, seed):
        expected = job["data_bundles"][f"{dataset}:{seed}"]
        return decode_arrays(bundle_store[expected["key"]], expected)
    assert hashlib.sha256(Path('/root/mnist.py').read_bytes()).hexdigest() == job['scorer_sha256']
    assert hashlib.sha256(Path('/root/canonical/run_modal.py').read_bytes()).hexdigest() == job['runner_sha256']
    os.environ['TRITON_CACHE_DIR'] = '/tmp/d6-triton'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/d6-inductor'
    p = Path('/root/d6learner.py')
    p.write_text(job['source'])
    spec = importlib.util.spec_from_file_location('d6learner', p)
    method = importlib.util.module_from_spec(spec)
    sys.modules['d6learner'] = method
    spec.loader.exec_module(method)
    result = {'arm': job['arm'], 'parameters': job['parameters'], 'phase': job['phase'],
              'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
              'scorer_sha256': job['scorer_sha256'], 'source_release_sha256': job['scorer_sha256'],
              'runner_sha256': job['runner_sha256'],
              'input_manifest_sha256': job['input_manifest_sha256'],
              'timing_scope': 'cumulative_algorithm_seconds includes synchronized _predict, probability accumulation and member/prefix argmax; excludes truth comparison and evidence serialization. full_call_seconds retains _predict-only time for comparison with the initial screen.',
              'data_bundle_dict': job['data_bundle_dict'],
              'gpu': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'numpy': str(np.__version__), 'rows': [],
              'scope': 'Unsandboxed accuracy research using canonical released draws. No secret release map, pixels, query truth, prior model or pool is passed to the learner. This is not an official latency/energy qualification.'}
    try:
        warm = draw_bundle('fashion', job['warmup_seed'])
        result['warmup_bundle'] = job['data_bundles'][f"fashion:{job['warmup_seed']}"]
        method.SEED = 11
        t = time.perf_counter()
        method._predict(*(torch.as_tensor(x, device='cuda') for x in warm[:3]))
        torch.cuda.synchronize()
        result['warmup_seconds'] = time.perf_counter() - t
        del warm
        for seed in job['draw_seeds']:
            tx, ty, qx, qy = draw_bundle('mnist', seed)
            bundle_meta = job['data_bundles'][f'mnist:{seed}']
            inputs = tuple(torch.as_tensor(x, device='cuda') for x in (tx, ty, qx))
            truth = torch.as_tensor(qy, device='cuda')
            aggregate = torch.zeros(len(qy), 10, device='cuda')
            row = {'draw_seed': seed, 'data_bundle_key': bundle_meta['key'],
                   'data_bundle_npz_sha256': bundle_meta['npz_sha256'],
                   'runner_sha256': job['runner_sha256'],
              'input_manifest_sha256': job['input_manifest_sha256'],
              'timing_scope': 'cumulative_algorithm_seconds includes synchronized _predict, probability accumulation and member/prefix argmax; excludes truth comparison and evidence serialization. full_call_seconds retains _predict-only time for comparison with the initial screen.',
                   'source_release_sha256': job['scorer_sha256'],
                   'array_sha256': {name: value['sha256'] for name,value in bundle_meta['arrays'].items()},
                   'train_sha256': hashlib.sha256(tx.tobytes()+ty.tobytes()).hexdigest(),
                   'query_sha256': hashlib.sha256(qx.tobytes()).hexdigest(),
                   'test_labels': qy.tolist(), 'members': [], 'prefixes': []}
            elapsed = 0.0
            algorithm_elapsed = 0.0
            for member in range(job['members']):
                method.SEED = 11 + 997 * member
                torch.cuda.manual_seed(500000 + member)
                torch.cuda.synchronize()
                start = time.perf_counter()
                probs, training_errors = method._predict(*inputs)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                elapsed += seconds
                assert bool(torch.isfinite(probs).all()) and bool((probs >= 0).all())
                torch.cuda.synchronize()
                postprocess_start = time.perf_counter()
                aggregate.add_(probs)
                pred = probs.argmax(1)
                prefix_pred = aggregate.argmax(1) if member+1 in job['prefixes'] else None
                torch.cuda.synchronize()
                postprocess_seconds = time.perf_counter() - postprocess_start
                algorithm_elapsed += seconds + postprocess_seconds
                member_record = {'member': member+1, 'model_seed': method.SEED,
                    'correct': int((pred == truth).sum()), 'total': len(qy),
                    'training_errors': int(training_errors), 'training_total': len(ty),
                    'full_call_seconds': seconds, 'probability_accumulation_argmax_seconds': postprocess_seconds,
                    'algorithm_seconds': seconds + postprocess_seconds,
                    'prediction': pred.cpu().tolist()}
                if job.get('save_probabilities'):
                    import base64
                    import zlib
                    raw_probs = probs.float().cpu().numpy().astype('<f4', copy=False).tobytes()
                    member_record.update(probabilities_codec='zlib-base64-f32-le',
                        probabilities_shape=[len(qy),10], probabilities_sha256=hashlib.sha256(raw_probs).hexdigest(),
                        probabilities=base64.b64encode(zlib.compress(raw_probs)).decode())
                row['members'].append(member_record)
                if member+1 in job['prefixes']:
                    pred = prefix_pred
                    row['prefixes'].append({'members': member+1, 'correct': int((pred == truth).sum()),
                        'total': len(qy), 'accuracy_pct': float((pred == truth).double().mean()*100),
                        'full_call_seconds': elapsed, 'cumulative_algorithm_seconds': algorithm_elapsed,
                        'prediction': pred.cpu().tolist()})
                if algorithm_elapsed > job['max_fit_seconds']:
                    row['budget_exhausted'] = True
                    break
            result['rows'].append(row)
        result['completed'] = True
    except BaseException:
        import traceback
        result['completed'] = False
        result['error'] = traceback.format_exc()
    payload = json.dumps(result)
    results[job['phase'] + ':' + job['arm']] = payload
    return payload


@app.function(cpu=4, memory=4096, timeout=1800, max_containers=1, single_use_containers=True)
def prepare_canonical_inputs(request):
    """CPU only: generate each missing canonical draw once and persist its arrays."""
    import datetime
    import hashlib
    from pathlib import Path
    import sys
    sys.path.insert(0, '/root')
    import mnist
    from input_bundles import DICT_NAME, bundle_key, decode_arrays, encode_arrays
    assert hashlib.sha256(Path('/root/mnist.py').read_bytes()).hexdigest() == request['scorer_sha256']
    store = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    entries = {}
    generated = []
    for dataset, seed in request['draws']:
        key = bundle_key(dataset, seed, request['scorer_sha256'])
        if key not in store:
            # No learner is imported in this function. Query truth is archived
            # only in trusted controller storage; no selection occurs here.
            arrays = mnist.draw(dataset, seed=seed)
            payload = encode_arrays(dataset, seed, arrays, request['scorer_sha256'], mnist.VERSION)
            assert store.put(key, payload, skip_if_exists=True), 'Concurrent preparation; keep winning immutable bundle'
            generated.append(key)
        payload = store[key]
        decode_arrays(payload)
        assert payload['metadata']['key'] == key
        entries[f'{dataset}:{seed}'] = payload['metadata']
    return {'format':'sutro-d6-input-manifest-v1','dict_name':DICT_NAME,
            'scorer_sha256':request['scorer_sha256'],
            'source_release_sha256':request['scorer_sha256'],
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'entries':entries,'generated_keys':generated,
            'scope':'Trusted controller canonical arrays. Query truth is never passed to a learner. Arrays are shared byte-for-byte across all future arms using this manifest.',
            'storage':'Modal Dict, primitive keys and compressed NPZ bytes; each entry below5MiB, well below documented100MiB maximum. Local archives retain evidence beyond Dict inactivity expiration.'}


def load_input_manifest(plan, *, verify_remote=True):
    """Reusable CPU preflight. Refuses missing, drifted, or incomplete bundles."""
    path = ROOT / plan['input_manifest']
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == plan['input_manifest_sha256'], 'Input manifest hash mismatch'
    manifest = input_bundles.validate_manifest(json.loads(raw), plan)
    if verify_remote:
        store = modal.Dict.from_name(manifest['dict_name'])
        for dataset,seed in input_bundles.required_entries(plan):
            expected = manifest['entries'][f'{dataset}:{seed}']
            payload = store[expected['key']]
            input_bundles.decode_arrays(payload, expected)
    return manifest


def build_jobs(plan, *, verify_remote=True):
    """Reusable CPU job construction for development and frozen confirmation."""
    for filename,key in [('mnist.py','scorer_sha256'),('run_modal.py','runner_sha256')]:
        assert hashlib.sha256((ROOT/'canonical'/filename).read_bytes()).hexdigest() == plan[key], f'Canonical {filename} drift'
    manifest = load_input_manifest(plan, verify_remote=verify_remote)
    if plan['phase'].startswith('confirmation'):
        assert plan.get('save_probabilities') is True, 'Confirmation must retain probabilities'
    jobs = []
    for arm in plan['arms']:
        path = ROOT / arm.get('source_path', 'sources/' + arm['name'] + '.py')
        source = path.read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == arm['source_sha256']
        jobs.append({'arm':arm['name'],'parameters':arm,'source':source,
                     'scorer_sha256':plan['scorer_sha256'],'runner_sha256':plan['runner_sha256'],'phase':plan['phase'],
                     'draw_seeds':plan['draw_seeds'],'warmup_seed':plan['warmup_seed'],
                     'members':arm['members'],'prefixes':arm['prefixes'],
                     'max_fit_seconds':plan.get('max_fit_seconds',300),
                     'save_probabilities':plan.get('save_probabilities',False),
                     'data_bundle_dict':manifest['dict_name'],
                     'data_bundles':manifest['entries'],
                     'input_manifest_sha256':plan['input_manifest_sha256']})
    return jobs


def prepare_inputs_main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Prepare frozen canonical arrays on a CPU container; no GPU call.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--local-dir', required=True)
    args = parser.parse_args(argv)
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    manifest_path = ROOT / args.manifest
    assert not manifest_path.exists(), 'Refuse to overwrite an existing input manifest'
    local = ROOT / args.local_dir
    local.mkdir(parents=True, exist_ok=True)
    request = {'scorer_sha256':plan['scorer_sha256'],'draws':input_bundles.required_entries(plan)}
    with modal.enable_output(), app.run():
        manifest = prepare_canonical_inputs.remote(request)
    input_bundles.validate_manifest(manifest, plan)
    store = modal.Dict.from_name(manifest['dict_name'])
    for label,entry in manifest['entries'].items():
        payload = store[entry['key']]
        input_bundles.decode_arrays(payload, entry)
        destination = local / (label.replace(':','-') + '.npz')
        if destination.exists():
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == entry['npz_sha256']
        else:
            destination.write_bytes(payload['npz'])
        entry['local_archive'] = str(destination.relative_to(ROOT))
    # Local archive paths belong to the manifest only, not the Dict metadata.
    # Keep job comparison with immutable Dict metadata exact by separating them.
    manifest['local_archives'] = {label:entry.pop('local_archive') for label,entry in manifest['entries'].items()}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest':args.manifest,'input_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      'entries':len(manifest['entries']),'GPU_jobs_launched':0},indent=2))


def main():
    import argparse
    if len(sys.argv)>1 and sys.argv[1]=='prepare-inputs':
        return prepare_inputs_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='plan.json')
    parser.add_argument('--output', default='development')
    parser.add_argument('--status', default='controller-status.json')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    jobs = build_jobs(plan)
    raw = ROOT / args.output
    raw.mkdir(exist_ok=True)
    assert not list(raw.glob('*.json')), 'Never replace completed or failed screen outcomes'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/args.status).write_text(json.dumps({'running':True,'started_utc':started,
        'input_manifest_sha256':plan['input_manifest_sha256']},indent=2)+'\n')
    with modal.enable_output(), app.run():
        for payload in screen.map(jobs, order_outputs=False):
            result = json.loads(payload)
            (raw/f"{result['arm']}.json").write_text(json.dumps(result,indent=1)+'\n')
            print(json.dumps({'arm':result['arm'],'completed':result['completed'],
                'prefixes': [[(p['members'],p['correct'],p['full_call_seconds']) for p in r['prefixes']] for r in result['rows']],
                'error':result.get('error') }),flush=True)
    (ROOT/args.status).write_text(json.dumps({'running':False,'started_utc':started,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input_manifest_sha256':plan['input_manifest_sha256']},indent=2)+'\n')


if __name__ == '__main__':
    main()
