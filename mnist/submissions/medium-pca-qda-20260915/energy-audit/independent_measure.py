"""Independent measurement of the unchanged submitted GPU learner.

Compare the cumulative energy counter with trapezoidal integration of sampled
board power. Paired idle windows and sham runs expose baseline subtraction
error. Both NVML readouts share hardware telemetry, not an external wattmeter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
import numpy as np
import torch
import pynvml as nv

sys.path.insert(0, '/workspace/source')
import gpu_benchmark as submitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payloads', default='/workspace/payloads')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    nv.nvmlInit()
    prop = torch.cuda.get_device_properties(0)
    uuid = str(prop.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    handle = nv.nvmlDeviceGetHandleByUUID(uuid)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    payloads, provenance = submitted.load_payloads(Path(args.payloads))
    x = torch.empty((10000, 81), device='cuda')
    q = torch.empty_like(x)
    y = torch.empty(10000, dtype=torch.int64, device='cuda')
    w0 = torch.tensor(submitted.initial_basis(), device='cuda')
    output = torch.empty(10000, dtype=torch.int64, device='cuda')

    def load(index):
        p = payloads[index][1]
        x.copy_(torch.from_numpy(p['x']))
        q.copy_(torch.from_numpy(p['q']))
        y.copy_(torch.from_numpy(p['labels'].astype(np.int64)))

    def guard():
        others = [{'pid': p.pid, 'used_gpu_memory': p.usedGpuMemory}
                  for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                  if p.pid not in {os.getpid(), int(os.environ['SUTRO_NVML_SELF_PID'])}]
        if others:
            raise RuntimeError('Other GPU processes: ' + json.dumps(others))

    guard()
    load(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            submitted.pca_qda_gpu(x, y, q, w0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    scratch = {}
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(submitted.pca_qda_gpu(x, y, q, w0, scratch))
    torch.cuda.synchronize()
    checks = []
    for i, (_, p) in enumerate(payloads):
        load(i)
        output.fill_(-1)
        for tensor in scratch.values():
            tensor.fill_(float('nan'))
        graph.replay()
        pred = output.cpu().numpy().copy()
        check = {'draw': i, 'matches': int((pred == p['frozen_predictions']).sum()),
                 'correct': int((pred == p['test_labels']).sum())}
        assert check['matches'] == 10000, check
        checks.append(check)
    load(0)
    graph.replay()
    torch.cuda.synchronize()
    doc = {'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'source_sha256': {n: hashlib.sha256((Path('/workspace/source')/n).read_bytes()).hexdigest()
                             for n in ('gpu_benchmark.py', 'reference.py')},
           'measurement_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'hardware': {'name': str(nv.nvmlDeviceGetName(handle)), 'uuid': uuid,
                        'memory_bytes': prop.total_memory, 'sm_count': prop.multi_processor_count,
                        'power_limit_w': nv.nvmlDeviceGetPowerManagementLimit(handle)/1000,
                        'driver': str(nv.nvmlSystemGetDriverVersion())},
           'software': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'numpy': np.__version__},
           'validation': checks,
           'scope': 'Complete device-resident training and prediction using unchanged pca_qda_gpu; no transfers, allocation, graph capture or area resize in timed window.',
           'telemetry_independence': 'Separate sampled-power integration and cumulative-energy arithmetic, sharing the GPU power sensors. Not external wall-socket measurement.',
           'nvml_pid_alias': int(os.environ['SUTRO_NVML_SELF_PID']),
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'scratch_logical_bytes': {k: v.numel()*v.element_size() for k,v in scratch.items()},
           'samples': [], 'intervals': [], 'comparisons': [], 'monitor_errors': []}
    stop = threading.Event()

    def sample():
        t0 = time.perf_counter()
        power = nv.nvmlDeviceGetPowerUsage(handle)/1000
        t1 = time.perf_counter()
        util = nv.nvmlDeviceGetUtilizationRates(handle)
        row = {'t': (t0+t1)/2, 'power_w': power, 'read_seconds': t1-t0,
               'sm_clock_mhz': nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
               'memory_clock_mhz': nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
               'temperature_c': nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
               'pstate': nv.nvmlDeviceGetPerformanceState(handle),
               'gpu_util_percent': util.gpu, 'memory_util_percent': util.memory}
        doc['samples'].append(row)

    def monitor():
        last_guard = 0
        while not stop.is_set():
            try:
                sample()
                if time.perf_counter()-last_guard > 1:
                    guard()
                    last_guard = time.perf_counter()
            except Exception as exc:
                doc['monitor_errors'].append(repr(exc))
            stop.wait(.05)

    def stamp():
        t0 = time.perf_counter()
        energy = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
        t1 = time.perf_counter()
        return {'t': (t0+t1)/2, 'energy_mj': int(energy), 'read_seconds': t1-t0}

    def interval(name, repeats=0, seconds=10):
        guard()
        torch.cuda.synchronize()
        sample()
        begin = stamp()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if repeats:
            a.record()
            for _ in range(repeats):
                graph.replay()
            b.record()
            torch.cuda.synchronize()
        else:
            time.sleep(seconds)
        end = stamp()
        sample()
        rows = sorted(doc['samples'], key=lambda r:r['t'])
        ts = np.array([r['t'] for r in rows])
        ps = np.array([r['power_w'] for r in rows])
        inner = (ts > begin['t']) & (ts < end['t'])
        selected_t = np.r_[begin['t'], ts[inner], end['t']]
        selected_p = np.interp(selected_t, ts, ps)
        joules = float(np.trapezoid(selected_p, selected_t))
        elapsed = end['t']-begin['t']
        counter = (end['energy_mj']-begin['energy_mj'])/1000
        row = {'name':name, 'repeats':repeats, 'start':begin, 'end':end,
               'seconds':elapsed, 'counter_j':counter, 'integrated_power_j':joules,
               'counter_w':counter/elapsed, 'integrated_power_w':joules/elapsed,
               'sample_count':int(inner.sum()),
               'cuda_ms':a.elapsed_time(b)/repeats if repeats else None}
        doc['intervals'].append(row)
        print(json.dumps({'interval':row}), flush=True)
        return row

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        # Longer active windows than the submitted benchmark; two idle-only
        # shams use 6000 nominal tasks solely to express subtraction noise.
        for i, (kind, repeats) in enumerate([('active',6000), ('sham',0), ('active',12000), ('sham',0), ('active',6000)]):
            before = interval(f'{i}-idle-before')
            active = interval(f'{i}-{kind}', repeats, seconds=20)
            after = interval(f'{i}-idle-after')
            denominator = repeats or 6000
            row = {'kind':kind, 'repeats':repeats, 'nominal_tasks':denominator,
                   'wall_ms_per_task':active['seconds']*1000/denominator,
                   'cuda_ms_per_task':active['cuda_ms']}
            for method in ('counter','integrated_power'):
                idle = (before[method+'_w']+after[method+'_w'])/2
                row[method+'_gross_mj_per_task'] = active[method+'_j']*1000/denominator
                row[method+'_idle_adjusted_mj_per_task'] = (active[method+'_j']-idle*active['seconds'])*1000/denominator
                row[method+'_idle_before_w'] = before[method+'_w']
                row[method+'_idle_after_w'] = after[method+'_w']
            doc['comparisons'].append(row)
            path.write_text(json.dumps(doc,indent=2)+'\n')
            print(json.dumps({'comparison':row}),flush=True)
        guard()
        assert not doc['monitor_errors'], doc['monitor_errors']
        graph.replay()
        assert np.array_equal(output.cpu().numpy(), payloads[0][1]['frozen_predictions'])
        doc['passed'] = True
    finally:
        stop.set()
        thread.join()
        doc['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        path.write_text(json.dumps(doc,indent=2)+'\n')
        nv.nvmlShutdown()


if __name__ == '__main__':
    main()
