"""One-time transition of future study jobs to frozen shared input archives."""
from pathlib import Path
p=Path(__file__).with_name('study.py')
s=p.read_text()
s=s.replace('ROOT = Path(__file__).resolve().parent\n', 'ROOT = Path(__file__).resolve().parent\nsys.path.insert(0, str(ROOT))\nimport input_bundles\n')
s=s.replace('image = runner.image.add_local_dir(ROOT / "canonical", "/root/canonical", copy=True)', 'image = (runner.image.add_local_dir(ROOT / "canonical", "/root/canonical", copy=True)\n         .add_local_file(ROOT / "input_bundles.py", "/root/input_bundles.py", copy=True))')
s=s.replace('    import mnist\n    assert hashlib.sha256', '    from input_bundles import decode_arrays\n    bundle_store = modal.Dict.from_name(job["data_bundle_dict"])\n    def draw_bundle(dataset, seed):\n        expected = job["data_bundles"][f"{dataset}:{seed}"]\n        return decode_arrays(bundle_store[expected["key"]], expected)\n    assert hashlib.sha256')
s=s.replace("              'scorer_sha256': job['scorer_sha256'], 'gpu': torch.cuda.get_device_name(),", "              'scorer_sha256': job['scorer_sha256'], 'source_release_sha256': job['scorer_sha256'],\n              'input_manifest_sha256': job['input_manifest_sha256'],\n              'data_bundle_dict': job['data_bundle_dict'],\n              'gpu': torch.cuda.get_device_name(),")
s=s.replace("        warm = mnist.draw('fashion', seed=job['warmup_seed'])", "        warm = draw_bundle('fashion', job['warmup_seed'])\n        result['warmup_bundle'] = job['data_bundles'][f\"fashion:{job['warmup_seed']}\"]")
s=s.replace("            tx, ty, qx, qy = mnist.draw('mnist', seed=seed)", "            tx, ty, qx, qy = draw_bundle('mnist', seed)\n            bundle_meta = job['data_bundles'][f'mnist:{seed}']")
s=s.replace("            row = {'draw_seed': seed, 'train_sha256':", "            row = {'draw_seed': seed, 'data_bundle_key': bundle_meta['key'],\n                   'data_bundle_npz_sha256': bundle_meta['npz_sha256'],\n                   'input_manifest_sha256': job['input_manifest_sha256'],\n                   'source_release_sha256': job['scorer_sha256'],\n                   'array_sha256': {name: value['sha256'] for name,value in bundle_meta['arrays'].items()},\n                   'train_sha256':")
start=s.index('def main():\n')
s=s[:start]+'''@app.function(cpu=4, memory=4096, timeout=1800, max_containers=1, single_use_containers=True)
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
    manifest = load_input_manifest(plan, verify_remote=verify_remote)
    if plan['phase'].startswith('confirmation'):
        assert plan.get('save_probabilities') is True, 'Confirmation must retain probabilities'
    jobs = []
    for arm in plan['arms']:
        path = ROOT / arm.get('source_path', 'sources/' + arm['name'] + '.py')
        source = path.read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == arm['source_sha256']
        jobs.append({'arm':arm['name'],'parameters':arm,'source':source,
                     'scorer_sha256':plan['scorer_sha256'],'phase':plan['phase'],
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
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n')
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
        'input_manifest_sha256':plan['input_manifest_sha256']},indent=2)+'\\n')
    with modal.enable_output(), app.run():
        for payload in screen.map(jobs, order_outputs=False):
            result = json.loads(payload)
            (raw/f"{result['arm']}.json").write_text(json.dumps(result,indent=1)+'\\n')
            print(json.dumps({'arm':result['arm'],'completed':result['completed'],
                'prefixes': [[(p['members'],p['correct'],p['full_call_seconds']) for p in r['prefixes']] for r in result['rows']],
                'error':result.get('error') }),flush=True)
    (ROOT/args.status).write_text(json.dumps({'running':False,'started_utc':started,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input_manifest_sha256':plan['input_manifest_sha256']},indent=2)+'\\n')


if __name__ == '__main__':
    main()
'''
assert "mnist.draw('mnist'" not in s and "mnist.draw('fashion'" not in s
p.write_text(s)
