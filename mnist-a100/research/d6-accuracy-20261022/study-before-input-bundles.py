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
spec = importlib.util.spec_from_file_location("canonical_runner", ROOT / "canonical/run_modal.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
image = runner.image.add_local_dir(ROOT / "canonical", "/root/canonical", copy=True)
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
    import mnist
    assert hashlib.sha256(Path('/root/mnist.py').read_bytes()).hexdigest() == job['scorer_sha256']
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
              'scorer_sha256': job['scorer_sha256'], 'gpu': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'numpy': str(np.__version__), 'rows': [],
              'scope': 'Unsandboxed accuracy research using canonical released draws. No secret release map, pixels, query truth, prior model or pool is passed to the learner. This is not an official latency/energy qualification.'}
    try:
        warm = mnist.draw('fashion', seed=job['warmup_seed'])
        method.SEED = 11
        t = time.perf_counter()
        method._predict(*(torch.as_tensor(x, device='cuda') for x in warm[:3]))
        torch.cuda.synchronize()
        result['warmup_seconds'] = time.perf_counter() - t
        del warm
        for seed in job['draw_seeds']:
            tx, ty, qx, qy = mnist.draw('mnist', seed=seed)
            inputs = tuple(torch.as_tensor(x, device='cuda') for x in (tx, ty, qx))
            truth = torch.as_tensor(qy, device='cuda')
            aggregate = torch.zeros(len(qy), 10, device='cuda')
            row = {'draw_seed': seed, 'train_sha256': hashlib.sha256(tx.tobytes()+ty.tobytes()).hexdigest(),
                   'query_sha256': hashlib.sha256(qx.tobytes()).hexdigest(),
                   'test_labels': qy.tolist(), 'members': [], 'prefixes': []}
            elapsed = 0.0
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
                aggregate.add_(probs)
                pred = probs.argmax(1)
                member_record = {'member': member+1, 'model_seed': method.SEED,
                    'correct': int((pred == truth).sum()), 'total': len(qy),
                    'training_errors': int(training_errors), 'training_total': len(ty),
                    'full_call_seconds': seconds, 'prediction': pred.cpu().tolist()}
                if job.get('save_probabilities'):
                    import base64
                    import zlib
                    raw_probs = probs.float().cpu().numpy().astype('<f4', copy=False).tobytes()
                    member_record.update(probabilities_codec='zlib-base64-f32-le',
                        probabilities_shape=[len(qy),10], probabilities_sha256=hashlib.sha256(raw_probs).hexdigest(),
                        probabilities=base64.b64encode(zlib.compress(raw_probs)).decode())
                row['members'].append(member_record)
                if member+1 in job['prefixes']:
                    pred = aggregate.argmax(1)
                    row['prefixes'].append({'members': member+1, 'correct': int((pred == truth).sum()),
                        'total': len(qy), 'accuracy_pct': float((pred == truth).double().mean()*100),
                        'full_call_seconds': elapsed, 'prediction': pred.cpu().tolist()})
                if elapsed > job['max_fit_seconds']:
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='plan.json')
    parser.add_argument('--output', default='development')
    parser.add_argument('--status', default='controller-status.json')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    jobs = []
    for arm in plan['arms']:
        path = ROOT / 'sources' / f"{arm['name']}.py"
        source = path.read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == arm['source_sha256']
        jobs.append({'arm': arm['name'], 'parameters': arm, 'source': source,
                     'scorer_sha256': plan['scorer_sha256'], 'phase': plan['phase'],
                     'draw_seeds': plan['draw_seeds'], 'warmup_seed': plan['warmup_seed'],
                     'members': arm['members'], 'prefixes': arm['prefixes'], 'max_fit_seconds': 300})
    raw = ROOT / args.output
    raw.mkdir(exist_ok=True)
    assert not list(raw.glob('*.json')), 'Never replace completed or failed initial screen outcomes'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/args.status).write_text(json.dumps({'running':True,'started_utc':started},indent=2)+'\n')
    with modal.enable_output(), app.run():
        for payload in screen.map(jobs, order_outputs=False):
            result = json.loads(payload)
            (raw/f"{result['arm']}.json").write_text(json.dumps(result,indent=1)+'\n')
            print(json.dumps({'arm':result['arm'],'completed':result['completed'],
                'prefixes': [[(p['members'],p['correct'],p['full_call_seconds']) for p in r['prefixes']] for r in result['rows']],
                'error':result.get('error') }),flush=True)
    (ROOT/args.status).write_text(json.dumps({'running':False,'started_utc':started,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')


if __name__ == '__main__':
    main()
