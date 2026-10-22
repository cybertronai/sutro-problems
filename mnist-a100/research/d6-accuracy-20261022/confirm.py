"""Explicit-launch controller for a frozen D6 confirmation plan; no selection/tuning."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_jobs(root, plan, manifest):
    from input_bundles import validate_manifest
    validate_manifest(manifest, plan)
    assert plan['phase'] == 'confirmation' and plan['frozen'] and plan['save_probabilities'] is True
    assert plan['draw_seeds'] == list(range(2610222101, 2610222112))
    assert [len(s) for s in plan['shards']] == [3,3,3,2]
    assert [seed for shard in plan['shards'] for seed in shard] == plan['draw_seeds']
    assert plan['max_concurrent_containers'] == 4 and plan['max_fit_seconds'] == 300
    assert sha(root/'canonical/mnist.py') == plan['scorer_sha256']
    assert sha(root/'canonical/run_modal.py') == plan['runner_sha256']
    assert sha(root/'study.py') == plan['study_producer_sha256']
    assert sha(root/'input_bundles.py') == plan['input_bundle_helper_sha256']
    assert isinstance(manifest['dict_name'], str) and manifest['dict_name']
    jobs = []
    for arm in plan['arms']:
        source = (root/arm['source_path']).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == arm['source_sha256']
        assert len(source.encode()) == arm['source_bytes']
        assert arm['prefixes'] == [arm['members']]
        for index, seeds in enumerate(plan['shards'], 1):
            entries = {f'fashion:{plan["warmup_seed"]}':manifest['entries'][f'fashion:{plan["warmup_seed"]}']}
            entries.update({f'mnist:{seed}':manifest['entries'][f'mnist:{seed}'] for seed in seeds})
            for entry in entries.values():
                assert len(entry['npz_sha256']) == 64 and entry['npz_bytes'] > 0
                assert entry['source_release_sha256'] == plan['scorer_sha256']
                assert set(entry['arrays']) == {'tx','ty','qx','qy'}
            jobs.append({'arm':f'{arm["name"]}__shard{index}','parameters':arm,'source':source,
                         'scorer_sha256':plan['scorer_sha256'],'phase':'confirmation',
                         'runner_sha256':plan['runner_sha256'],
                         'draw_seeds':seeds,'warmup_seed':plan['warmup_seed'],'members':arm['members'],
                         'prefixes':arm['prefixes'],'max_fit_seconds':300,'save_probabilities':True,
                         'data_bundle_dict':manifest['dict_name'],'data_bundles':entries})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'confirmation-plan.json')
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--expected-input-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'confirmation')
    args = parser.parse_args()
    assert sha(args.plan) == args.expected_plan_sha256, 'Frozen plan hash differs'
    assert sha(args.input_manifest) == args.expected_input_manifest_sha256, 'Frozen input manifest hash differs'
    plan, manifest = json.loads(args.plan.read_text()), json.loads(args.input_manifest.read_text())
    jobs = prepare_jobs(ROOT, plan, manifest)
    import study
    attached_plan = dict(plan, input_manifest=str(args.input_manifest.resolve()),
                         input_manifest_sha256=args.expected_input_manifest_sha256)
    # CPU-only live Dict preflight decodes every immutable NPZ against the frozen manifest.
    assert study.load_input_manifest(attached_plan, verify_remote=True) == manifest
    for job in jobs:
        job['input_manifest_sha256'] = args.expected_input_manifest_sha256
    assert not args.output.exists() or not list(args.output.glob('*.json')), 'Never overwrite confirmation outcomes'
    args.output.mkdir(exist_ok=True)
    status = ROOT/'confirmation-controller-status.json'
    assert not status.exists(), 'A confirmation launch was already attempted; preserve failures, do not replace'
    launch = {'running':True,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan_sha256':args.expected_plan_sha256,'input_manifest_sha256':args.expected_input_manifest_sha256,
              'input_manifest_path':str(args.input_manifest.resolve()),
              'jobs':[{'arm':j['arm'],'draw_seeds':j['draw_seeds'],'source_sha256':j['parameters']['source_sha256'],'members':j['members']} for j in jobs],
              'max_concurrent_containers':4,'source_or_prefix_tuning_allowed':False}
    status.write_text(json.dumps(launch,indent=2)+'\n')
    # No GPU function is invoked before all integrity and live archive checks pass.
    try:
        with study.modal.enable_output(), study.app.run():
            for payload in study.screen.map(jobs, order_outputs=False):
                result = json.loads(payload)
                destination = args.output/(result['arm']+'.json')
                assert not destination.exists(), 'Refusing to overwrite a saved confirmation shard'
                destination.write_text(json.dumps(result,indent=1)+'\n')
                print(json.dumps({'arm':result['arm'],'completed':result['completed'],'draws':len(result.get('rows',[])),'error':result.get('error')}),flush=True)
        launch['completed'] = True
    except BaseException:
        import traceback
        launch['completed'] = False
        launch['error'] = traceback.format_exc()
        raise
    finally:
        launch.update(running=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status.write_text(json.dumps(launch,indent=2)+'\n')


if __name__ == '__main__':
    main()
