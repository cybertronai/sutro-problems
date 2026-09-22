"""Full-split accuracy, kernel cross-checks, and paired-idle A100 measurements."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time

import numpy as np
import pynvml as nv
import torch

from data import official
import model


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='raw')
    p.add_argument('--output', default='generated/rerun')
    p.add_argument('--dimensions', type=int, nargs='+', default=[100, 80])
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--repeats', type=int, default=60)
    p.add_argument('--idle-seconds', type=float, default=10)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    def processes():
        return [{'pid': x.pid, 'bytes': x.usedGpuMemory} for x in nv.nvmlDeviceGetComputeRunningProcesses(h)]
    initial = processes()
    assert not initial, ('GPU is already occupied', initial)
    assert str(nv.nvmlDeviceGetName(h)) == 'NVIDIA A100-SXM4-40GB'
    torch.cuda.init()
    before = processes()
    probe = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    probe.fill_(7)
    torch.cuda.synchronize()
    during = processes()
    assert len(during) == 1 and during[0]['bytes'] - sum(x['bytes'] for x in before) >= probe.numel()
    own_pid = during[0]['pid']
    del probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    released = processes()
    assert during[0]['bytes'] - sum(x['bytes'] for x in released) >= 128 * 1024 * 1024
    def guard():
        other = [x for x in processes() if x['pid'] != own_pid]
        assert not other, ('Other GPU processes', other)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    arrays, manifest = official(args.raw)
    x, y, q = [torch.from_numpy(a).cuda() for a in arrays[:3]]
    labels = arrays[3]
    doc = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'config': {'L1': 8, 'L2': 5, 'blocks': 9, 'block_size': 14, 'stride': 7,
                   'features': 2304, 'mixture_components_per_class': 8,
                   'dimensions': args.dimensions, 'rounds': args.rounds,
                   'repeats': args.repeats, 'idle_seconds': args.idle_seconds,
                   'tf32_matmul': True, 'tf32_cudnn': True},
        'hardware': {'gpu': str(nv.nvmlDeviceGetName(h)), 'uuid': str(nv.nvmlDeviceGetUUID(h)),
                     'driver': str(nv.nvmlSystemGetDriverVersion()),
                     'power_limit_w': nv.nvmlDeviceGetPowerManagementLimit(h) / 1000},
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                     'numpy': np.__version__},
        'process_ownership': {'worker_pid': os.getpid(), 'nvml_pid': own_pid,
                              'initial': initial, 'before_probe': before,
                              'during_probe': during, 'after_release': released},
        'data': manifest, 'source_sha256': {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob('*.py')},
        'scope': 'Every repetition learns both filter banks, extracts all 70000 features, fits PCA and all class mixtures, and predicts 10000 labels. Device-resident normalized input. Includes tensor allocation and host dispatch; excludes data download, normalization, transfer, compilation, warmup, validation and test-label scoring.',
        'runs': {}, 'samples': [], 'intervals': [], 'monitor_errors': [],
    }
    def save():
        (out / 'results.json').write_text(json.dumps(doc, indent=2) + '\n')
    print('GPU', doc['hardware'], flush=True)
    # Compile and validate before the measurement windows.
    for K in args.dimensions:
        print('Validating K=', K, flush=True)
        preds = []
        for _ in range(3):
            preds.append(model.train_predict(x, y, q, K=K).cpu().numpy())
        pred = preds[0]
        np.save(out / f'predictions-k{K}.npy', pred)
        correct = int(np.count_nonzero(pred == labels))
        ref = model.train_predict(x, y, q, K=K, backend='reference').cpu().numpy()
        np.save(out / f'reference-predictions-k{K}.npy', ref)
        check = {'correct': correct, 'total': 10000, 'errors': 10000 - correct,
                 'accuracy_pct': correct / 100, 'passes_99_percent': correct >= 9900,
                 'prediction_sha256': digest(pred),
                 'repeat_prediction_matches': [int(np.count_nonzero(a == pred)) for a in preds[1:]],
                 'reference_correct': int(np.count_nonzero(ref == labels)),
                 'reference_prediction_matches': int(np.count_nonzero(ref == pred))}
        assert all(a == 10000 for a in check['repeat_prediction_matches']), check
        # Test labels are never passed into train_predict.
        if K == args.dimensions[0]:
            rotated = model.train_predict(x, (y + 1) % 10, q, K=K).cpu().numpy()
            check['rotated_train_label_changed_predictions'] = int(np.count_nonzero(rotated != pred))
            check['rotated_train_label_correct'] = int(np.count_nonzero(rotated == labels))
            assert check['rotated_train_label_changed_predictions'] > 9000
        doc['runs'][str(K)] = {'validation': check, 'rounds': []}
        print(check, flush=True)
        save()
    # Check emitted features, including a non-default CUDA stream.
    W1 = model.pca_filters(model.patches(x[:255], 7), 8)
    M1 = model.stage(x[:250], W1, 7)
    W2 = model.pca_filters(model.patches(M1.reshape(-1, 28, 28)[:2000], 7), 5)
    sample = torch.cat((x[:512], q[:1024]))
    reference_features = model.pcanet_feats(sample, W1, W2, 8, 5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel_features = model.feats_cuda(sample, W1, W2, 8, 5)
    torch.cuda.current_stream().wait_stream(stream)
    doc['feature_validation'] = {
        'images': len(sample), 'values': reference_features.numel(),
        'mismatched_values': int(torch.count_nonzero(reference_features != kernel_features).item()),
        'max_abs_difference': float((reference_features - kernel_features).abs().max().item()),
        'nondefault_stream_checked': True,
    }
    print('Feature check:', doc['feature_validation'], flush=True)
    del W1, W2, M1, sample, reference_features, kernel_features
    gc.collect()
    torch.cuda.empty_cache()
    stop = threading.Event()
    lock = threading.Lock()
    def sample_power():
        a = time.perf_counter()
        watts = nv.nvmlDeviceGetPowerUsage(h) / 1000
        b = time.perf_counter()
        row = {'t': (a + b) / 2, 'power_w': watts,
               'temperature_c': nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
               'sm_mhz': nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)}
        with lock:
            doc['samples'].append(row)
    def monitor():
        next_guard = 0
        while not stop.is_set():
            try:
                sample_power()
                if time.perf_counter() >= next_guard:
                    guard()
                    next_guard = time.perf_counter() + 1
            except Exception as e:
                doc['monitor_errors'].append(repr(e))
            stop.wait(.02)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def stamp():
        a = time.perf_counter()
        e = nv.nvmlDeviceGetTotalEnergyConsumption(h)
        b = time.perf_counter()
        return {'t': (a + b) / 2, 'energy_mj': int(e)}
    def window(name, fn=None, seconds=None):
        guard()
        torch.cuda.synchronize()
        sample_power()
        start = stamp()
        if fn is None:
            time.sleep(seconds)
        else:
            fn()
        torch.cuda.synchronize()
        end = stamp()
        sample_power()
        with lock:
            samples = sorted(doc['samples'], key=lambda a: a['t'])
        ts = np.array([a['t'] for a in samples])
        ps = np.array([a['power_w'] for a in samples])
        inner = ts[(ts > start['t']) & (ts < end['t'])]
        times = np.r_[start['t'], inner, end['t']]
        powers = np.interp(times, ts, ps)
        dt = end['t'] - start['t']
        row = {'name': name, 'start': start, 'end': end, 'seconds': dt,
               'counter_j': (end['energy_mj'] - start['energy_mj']) / 1000,
               'sampled_j': float(np.trapezoid(powers, times))}
        row['counter_w'] = row['counter_j'] / dt
        row['sampled_w'] = row['sampled_j'] / dt
        doc['intervals'].append(row)
        guard()
        assert not doc['monitor_errors'], doc['monitor_errors']
        return row
    try:
        a = torch.randn(8192, 8192, device='cuda')
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        idle = window('sensor-idle', seconds=10)
        def burn():
            deadline = time.perf_counter() + 5
            while time.perf_counter() < deadline:
                for _ in range(20):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize()
        load = window('sensor-load', burn)
        doc['sensor_check'] = {'idle_w': idle['counter_w'], 'load_w': load['counter_w'],
                               'counter_sampled_relative_difference': abs(load['counter_j'] - load['sampled_j']) / load['counter_j']}
        assert load['counter_w'] > max(150, idle['counter_w'] + 50)
        assert doc['sensor_check']['counter_sampled_relative_difference'] < .1
        del a, b, c
        torch.cuda.empty_cache()
        for K in args.dimensions:
            baseline_pred = np.load(out / f'predictions-k{K}.npy')
            last = None
            def workload():
                nonlocal last
                for _ in range(args.repeats):
                    last = model.train_predict(x, y, q, K=K)
            model.train_predict(x, y, q, K=K)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for r in range(args.rounds):
                before = window(f'k{K}-r{r}-before', seconds=args.idle_seconds)
                active = window(f'k{K}-r{r}-active', workload)
                after = window(f'k{K}-r{r}-after', seconds=args.idle_seconds)
                row = {'round': r, 'repeats': args.repeats, 'task_ms': active['seconds'] * 1000 / args.repeats,
                       'gross_j': active['counter_j'] / args.repeats,
                       'active_seconds': active['seconds'],
                       'last_prediction_matches': int(np.count_nonzero(last.cpu().numpy() == baseline_pred))}
                for method in ('counter', 'sampled'):
                    idle_w = (before[method + '_w'] + after[method + '_w']) / 2
                    row['idle_w_' + method] = idle_w
                    row['adjusted_j_' + method] = (active[method + '_j'] - idle_w * active['seconds']) / args.repeats
                assert row['last_prediction_matches'] == 10000, row
                doc['runs'][str(K)]['rounds'].append(row)
                print('MEASURE', K, row, flush=True)
                save()
            doc['runs'][str(K)]['summary'] = {
                k: statistics.median(r[k] for r in doc['runs'][str(K)]['rounds'])
                for k in ('task_ms', 'gross_j', 'adjusted_j_counter', 'adjusted_j_sampled')}
            doc['runs'][str(K)]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            doc['runs'][str(K)]['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        before = window('control-before', seconds=args.idle_seconds)
        sham = window('control-idle-only', seconds=15)
        after = window('control-after', seconds=args.idle_seconds)
        doc['idle_only_control'] = {
            method + '_net_j': sham[method + '_j'] - .5 * (before[method + '_w'] + after[method + '_w']) * sham['seconds']
            for method in ('counter', 'sampled')}
        doc['completed'] = True
    finally:
        stop.set()
        thread.join()
        save()
        (out / 'nvidia-smi.txt').write_text(subprocess.run(['nvidia-smi', '-q'], text=True, capture_output=True).stdout)
        nv.nvmlShutdown()
    print('Completed:', {k: v['summary'] for k, v in doc['runs'].items()}, flush=True)


if __name__ == '__main__':
    main()
