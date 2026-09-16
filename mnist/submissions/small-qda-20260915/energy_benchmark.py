#!/usr/bin/env python3
"""Measure the unchanged PTX learner with settled idle and telemetry controls.

python energy_benchmark.py --original-payloads generated/payloads \
    --fresh-payloads generated/fresh-payloads --expected-uuid GPU-... \
    --output results/energy-a100.json

Inputs are exported by run.py. Each measured replay fits the model and predicts
1,000 labels. Transfers, preprocessing, allocation, compilation and capture are
excluded. Both NVML readouts share device telemetry; agreement is not external
power-meter calibration. All signed estimates and raw samples are retained.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import pynvml as nv
import torch

import gpu_benchmark_ptx as submitted
from energy_verify import analyze

HERE = Path(__file__).resolve().parent
EXPECTED_PTX_SHA256 = '38d36146e4303b7acd640ee6c04ba642c751182c4fb9501b9fbccd52333cba58'
PLAN = {
    'version': 1,
    'sequence': ['ptx', 'sham', 'ptx', 'ptx'],
    'ptx_replays': 1200000,
    'settle_seconds': 3,
    'idle_seconds': 10,
    'sham_seconds': 20,
    'power_sample_interval_seconds': 0.05,
    'process_check_interval_seconds': 1,
    'energy_dataset': 'original draw 0, seed 20261201',
    'positive_control': {'matrix_dimension': 4096, 'seconds': 10,
                         'dtype': 'float32', 'tf32': False,
                         'minimum_above_idle_w': 20},
    'reporting': 'All rounds, signed subtraction, per-round subtraction before median; no energy-based round selection.',
}
TRACE_COLUMNS = ['time_s', 'power_w', 'energy_time_s', 'energy_mj',
                 'sm_clock_mhz', 'memory_clock_mhz', 'temperature_c',
                 'gpu_util_percent', 'memory_util_percent', 'pstate']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-payloads', type=Path, required=True)
    parser.add_argument('--fresh-payloads', type=Path, required=True)
    parser.add_argument('--expected-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert sys.version_info[:2] == (3, 11), 'Use the documented Python 3.11 environment'
    assert torch.__version__ == '2.5.1+cu124' and np.__version__ == '2.1.2'
    assert importlib.metadata.version('pyptx') == '0.1.1'
    assert importlib.metadata.version('nvidia-ml-py') == '13.610.43'
    payloads, provenance = {}, {}
    for kind, path in [('original', args.original_payloads), ('fresh', args.fresh_payloads)]:
        payloads[kind], full = submitted.load_payloads(path)
        assert len(payloads[kind]) == 11, 'Supply the complete eleven-draw directory'
        evidence = HERE / ('evidence/accuracy' if kind == 'original' else 'evidence/fresh/accuracy')
        manifest = full['payload_manifest']
        for name in ('draw_manifest', 'prediction_manifest'):
            assert manifest[name + '_sha256'] == sha(evidence / (name + '.json'))
        draws = json.loads((evidence / 'draw_manifest.json').read_text())['draws']
        predictions = json.loads((evidence / 'prediction_manifest.json').read_text())['draws']
        for index, (_, payload) in enumerate(payloads[kind]):
            for array, expected, dtype in (
                ('x', draws[index]['train_input_sha256'], '<f4'),
                ('q', draws[index]['test_input_sha256'], '<f4'),
                ('frozen_predictions', predictions[index]['array_sha256'], '<i8'),
            ):
                assert hashlib.sha256(payload[array].astype(dtype).tobytes()).hexdigest() == expected
        provenance[kind] = {
            'manifest_sha256': full['payload_manifest_sha256'],
            'draw_manifest_sha256': manifest['draw_manifest_sha256'],
            'prediction_manifest_sha256': manifest['prediction_manifest_sha256'],
            'draws': [{'draw': record['draw'], 'seed': record['dataset_seed'],
                       'payload_sha256': record['sha256']} for record, _ in payloads[kind]],
        }
    assert [r['seed'] for r in provenance['original']['draws']] == list(range(20261201, 20261212))
    assert [r['seed'] for r in provenance['fresh']['draws']] == list(range(20262101, 20262112))
    epoch = time.perf_counter()
    now = lambda: time.perf_counter() - epoch
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(args.expected_uuid)
    assert nv.nvmlDeviceGetName(handle) == 'NVIDIA A100-SXM4-40GB'

    def processes():
        return [{'pid': p.pid, 'memory_bytes': p.usedGpuMemory}
                for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]

    before_context = processes()
    assert not before_context, 'GPU has another compute process; retry when idle'
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    uuid = uuid if uuid.startswith('GPU-') else 'GPU-' + uuid
    assert uuid == args.expected_uuid, 'CUDA and the requested NVML device differ'
    torch.cuda.init()
    torch.cuda.synchronize()
    before_probe = processes()
    assert len(before_probe) <= 1, before_probe
    probe = torch.empty(128 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    probe.fill_(3)
    torch.cuda.synchronize()
    after_probe = processes()
    assert len(after_probe) == 1, after_probe
    own_pid = after_probe[0]['pid']
    assert all(p['pid'] == own_pid for p in before_probe)
    assert after_probe[0]['memory_bytes'] - sum(p['memory_bytes'] for p in before_probe) >= probe.numel()
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_release = processes()
    assert all(p['pid'] == own_pid for p in after_release)
    assert after_probe[0]['memory_bytes'] - sum(p['memory_bytes'] for p in after_release) >= 128 * 1024 * 1024
    ownership = {'python_pid': os.getpid(), 'nvml_pid': own_pid,
                 'before_context': before_context, 'before_probe': before_probe,
                 'after_128MiB_allocation': after_probe, 'after_release': after_release}

    def guard():
        rows = processes()
        if any(p['pid'] != own_pid for p in rows):
            raise RuntimeError('Another GPU process: ' + json.dumps(rows))
        return rows

    guard()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    kernels = submitted.build_kernels()
    stats_ptx, score_ptx = [kernel.ptx() for kernel in kernels]
    ptx_sha = hashlib.sha256((stats_ptx + score_ptx).encode()).hexdigest()
    assert ptx_sha == EXPECTED_PTX_SHA256, 'Generated PTX differs from the submitted learner'

    def entry(ptx):
        return next(line for line in ptx.splitlines() if '.entry' in line).split('.entry')[1].split('(')[0].strip()

    x = torch.empty((1000, 9), device='cuda', dtype=torch.float32)
    y = torch.empty(1000, device='cuda', dtype=torch.int32)
    q = torch.empty_like(x)
    out = torch.empty(1000, device='cuda', dtype=torch.int32)
    prm = torch.empty(submitted.C * submitted.NPARAM, device='cuda')
    part = torch.empty(submitted.NB1 * submitted.C * submitted.SLOTS, device='cuda')
    counter = torch.zeros(2, device='cuda', dtype=torch.int32)
    stats_launcher = submitted.DriverLauncher(stats_ptx, entry(stats_ptx))
    score_launcher = submitted.DriverLauncher(score_ptx, entry(score_ptx))

    def task():
        stats_launcher.launch([x, y, q, part, prm, counter, out], submitted.BLOCK, submitted.NB1)
        score_launcher.launch([part, q, out], submitted.BLOCK2, submitted.NB2)

    def load(payload):
        x.copy_(torch.from_numpy(payload['x']))
        y.copy_(torch.from_numpy(payload['labels'].astype(np.int32)))
        q.copy_(torch.from_numpy(payload['q']))

    def poison():
        out.fill_(-1)
        part.fill_(float('nan'))
        prm.fill_(float('nan'))

    load(payloads['original'][0][1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            task()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        task()
    torch.cuda.synchronize()
    validation = {}
    for kind, draws in payloads.items():
        checks = []
        for index, (_, payload) in enumerate(draws):
            guard()
            load(payload)
            poison()
            task()
            eager = out.cpu().numpy().copy()
            replays = []
            for _ in range(3):
                poison()
                graph.replay()
                replays.append(bool(np.array_equal(out.cpu().numpy(), eager)))
            row = {'draw': index, 'seed': int(payload['dataset_seed']),
                   'matches': int((eager == payload['frozen_predictions']).sum()),
                   'correct': int((eager == payload['test_labels']).sum()),
                   'graph_replays_equal_eager': replays,
                   'predictions_int64_sha256': hashlib.sha256(eager.astype('<i8').tobytes()).hexdigest()}
            assert row['matches'] == 1000 and all(replays), row
            checks.append(row)
        validation[kind] = checks
    load(payloads['original'][0][1])
    poison()
    graph.replay()
    torch.cuda.synchronize()
    library_paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if 'libnvidia-ml.so' in line and line.split()[-1].startswith('/')})
    plan_sha = hashlib.sha256(json.dumps(PLAN, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    doc = {
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'plan': PLAN, 'plan_sha256': plan_sha, 'ptx_sha256': ptx_sha,
        'source_sha256': {name: sha(HERE / name) for name in
                         ('energy_benchmark.py', 'energy_verify.py', 'gpu_benchmark_ptx.py',
                          'run.py', 'reference.py', 'protocol.json', 'protocol_fresh.json', 'protocol_beacon.json')},
        'payloads': provenance, 'validation': validation, 'pid_ownership': ownership,
        'hardware': {'name': nv.nvmlDeviceGetName(handle), 'uuid': uuid,
                     'driver': nv.nvmlSystemGetDriverVersion(), 'vbios': nv.nvmlDeviceGetVbiosVersion(handle),
                     'memory_bytes': properties.total_memory, 'multiprocessors': properties.multi_processor_count,
                     'power_limit_w': nv.nvmlDeviceGetPowerManagementLimit(handle)/1000,
                     'default_power_limit_w': nv.nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000},
        'software': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'numpy': np.__version__, 'pyptx': importlib.metadata.version('pyptx'),
                     'nvidia-ml-py': importlib.metadata.version('nvidia-ml-py'),
                     'pynvml_sha256': sha(nv.__file__),
                     'nvml_libraries_sha256': {Path(path).name: sha(path) for path in library_paths}},
        'scope': 'Complete training and 1000 predictions, warm CUDA graph, original draw zero. Device-resident inputs. Includes host dispatch gaps; excludes resizing, transfers, allocation, compilation and capture. Settling, idle and matrix control are separate from task energy.',
        'trace_columns': TRACE_COLUMNS, 'trace': [], 'intervals': [], 'comparisons': [],
        'process_samples': [], 'monitor_errors': [], 'measurement_complete': False,
    }
    stop, lock = threading.Event(), threading.Lock()

    def save():
        # Compact JSON keeps the full raw trace reviewable without repeated keys.
        args.output.write_text(json.dumps(doc, separators=(',', ':')) + '\n')

    def sample():
        with lock:
            a = now(); power = nv.nvmlDeviceGetPowerUsage(handle)/1000; b = now()
            c = now(); energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); d = now()
            util = nv.nvmlDeviceGetUtilizationRates(handle)
            doc['trace'].append([(a+b)/2, power, (c+d)/2, energy,
                nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
                nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
                nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
                util.gpu, util.memory, nv.nvmlDeviceGetPerformanceState(handle)])

    def monitor():
        previous = -1
        while not stop.is_set():
            try:
                sample()
                if now() - previous >= PLAN['process_check_interval_seconds']:
                    doc['process_samples'].append({'time_s': now(), 'processes': guard()})
                    previous = now()
            except Exception as error:
                doc['monitor_errors'].append(repr(error))
            stop.wait(PLAN['power_sample_interval_seconds'])

    def stamp():
        a = now(); energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b = now()
        return {'time_s': (a+b)/2, 'energy_mj': energy, 'read_seconds': b-a}

    def interval(name, seconds=0, replays=0, matrices=None):
        guard()
        assert not doc['monitor_errors'], doc['monitor_errors']
        torch.cuda.synchronize()
        sample()
        start = stamp()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        actual = 0
        if replays:
            for i in range(replays):
                graph.replay()
                if i % 10000 == 0 and doc['monitor_errors']:
                    raise RuntimeError(doc['monitor_errors'])
            actual = replays
        elif matrices is not None:
            until = now() + seconds
            while now() < until:
                for _ in range(10):
                    torch.mm(matrices[0], matrices[1], out=matrices[2])
                    actual += 1
                torch.cuda.synchronize()
                assert not doc['monitor_errors'], doc['monitor_errors']
        else:
            time.sleep(seconds)
        end.record()
        torch.cuda.synchronize()
        finish = stamp()
        sample()
        row = {'name': name, 'start': start, 'end': finish, 'replays': actual,
               'cuda_ms_per_replay': begin.elapsed_time(end)/actual if actual else None}
        doc['intervals'].append(row)
        print(json.dumps(row), flush=True)
        return name

    def settled_idle(name):
        interval(name + '-settle', seconds=PLAN['settle_seconds'])
        return interval(name, seconds=PLAN['idle_seconds'])

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for i, kind in enumerate(PLAN['sequence']):
            name = f'{i}-{kind}'
            before = settled_idle(name + '-idle-before')
            middle = interval(name, replays=PLAN['ptx_replays'] if kind == 'ptx' else 0,
                              seconds=PLAN['sham_seconds'])
            after = settled_idle(name + '-idle-after')
            doc['comparisons'].append({'kind': kind, 'before': before, 'active': middle, 'after': after,
                                       'nominal_tasks': PLAN['ptx_replays']})
            if kind == 'ptx':
                assert np.array_equal(out.cpu().numpy(), payloads['original'][0][1]['frozen_predictions'])
            save()
        dim = PLAN['positive_control']['matrix_dimension']
        ma = torch.full((dim, dim), 1/64, device='cuda')
        mb, mc = torch.full_like(ma, 1/64), torch.empty_like(ma)
        torch.mm(ma, mb, out=mc)
        torch.cuda.synchronize()
        before = settled_idle('matmul-idle-before')
        middle = interval('matmul', seconds=PLAN['positive_control']['seconds'], matrices=(ma, mb, mc))
        after = settled_idle('matmul-idle-after')
        assert bool(torch.isfinite(mc).all()) and bool((mc == 1).all())
        doc['comparisons'].append({'kind': 'matmul', 'before': before, 'active': middle, 'after': after,
                                   'nominal_tasks': next(r['replays'] for r in doc['intervals'] if r['name'] == middle)})
        doc['positive_control_output_verified'] = True
        guard()
        assert not doc['monitor_errors'], doc['monitor_errors']
        doc['measurement_complete'] = True
    except BaseException as error:
        doc['failure'] = repr(error)
        raise
    finally:
        stop.set(); thread.join()
        doc['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()
        nv.nvmlShutdown()
    doc['summary'] = analyze(doc)
    save()
    print(json.dumps(doc['summary'], indent=2))
    if not doc['summary']['telemetry_sanity_passed']:
        raise SystemExit('Positive-control telemetry sanity check failed; raw result retained')


if __name__ == '__main__':
    main()
