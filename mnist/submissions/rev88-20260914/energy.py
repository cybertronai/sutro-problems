"""Paired-idle NVML measurement of repeated fresh GPU-resident tasks."""
from __future__ import annotations
import hashlib
import importlib.metadata
import math
import platform
import statistics
import time

import numpy as np


def measure(task, trials=3, target_seconds=5.0):
    import ctypes
    import glob
    import torch
    import pynvml as nv
    task.run(); torch.cuda.synchronize()
    before_pred, before_meta = task.outputs()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); task.run(); stop.record(); stop.synchronize()
    calibration_ms = start.elapsed_time(stop)
    repeats = max(1, min(1000, math.ceil(target_seconds * 1000 / calibration_ms)))
    nv.nvmlInit()
    runtime=None
    candidates=glob.glob('/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*')+glob.glob('/usr/local/cuda/lib64/libcudart.so*')+['libcudart.so.12']
    for candidate in candidates:
        try: runtime=ctypes.CDLL(candidate);break
        except OSError: pass
    if runtime is None: raise RuntimeError('Cannot verify CUDA/NVML device identity: libcudart missing')
    pci=ctypes.create_string_buffer(32)
    runtime.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    status=runtime.cudaDeviceGetPCIBusId(pci,32,torch.cuda.current_device())
    if status!=0: raise RuntimeError(f'cudaDeviceGetPCIBusId failed: {status}')
    handle=nv.nvmlDeviceGetHandleByPciBusId(pci.value)
    def decode(x): return x.decode() if isinstance(x,bytes) else str(x)
    def stamp():
        a=time.perf_counter(); e=int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b=time.perf_counter()
        return {"energy_mj":e,"time_s":(a+b)/2,"query_latency_s":b-a}
    def interval(a,b):
        dt=b['time_s']-a['time_s']; joules=(b['energy_mj']-a['energy_mj'])/1000
        return {'start':a,'end':b,'duration_s':dt,'energy_j':joules,'average_power_w':joules/dt}
    def idle():
        torch.cuda.synchronize(); time.sleep(3); a=stamp(); time.sleep(3)
        return {**interval(a,stamp()),'settle_seconds':3}
    def telemetry():
        return {'temperature_c':nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                'power_w':nv.nvmlDeviceGetPowerUsage(handle)/1000,
                'graphics_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_GRAPHICS),
                'memory_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM)}
    hardware={'gpu':decode(nv.nvmlDeviceGetName(handle)),'uuid':decode(nv.nvmlDeviceGetUUID(handle)),
              'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
              'power_limit_w':nv.nvmlDeviceGetPowerManagementLimit(handle)/1000,
              'driver':decode(nv.nvmlSystemGetDriverVersion()),'cuda_pci_bus_id':pci.value.decode(),
              'nvml_pci_bus_id':decode(nv.nvmlDeviceGetPciInfo(handle).busId),
              'nvml_handle_selected_by_cuda_pci_bus_id':True}
    assert 'A100' in hardware['gpu'],hardware
    records=[]
    try:
        for trial in range(trials):
            idle_before=idle(); telemetry_before=telemetry(); a=stamp(); start.record()
            for _ in range(repeats): task.run()
            stop.record(); stop.synchronize(); active=interval(a,stamp()); telemetry_after=telemetry(); idle_after=idle()
            idle_w=(idle_before['average_power_w']+idle_after['average_power_w'])/2
            adjusted=active['energy_j']-idle_w*active['duration_s']
            records.append({'trial':trial,'invocations':repeats,'idle_before':idle_before,'idle_after':idle_after,
                            'active':active,'telemetry_before':telemetry_before,'telemetry_after':telemetry_after,
                            'paired_idle_w':idle_w,'idle_adjusted_mj_per_task':adjusted*1000/repeats,
                            'unadjusted_mj_per_task':active['energy_j']*1000/repeats,
                            'cuda_ms_per_task':start.elapsed_time(stop)/repeats,
                            'wall_ms_per_task':active['duration_s']*1000/repeats})
            print('Energy trial',trial,records[-1]['idle_adjusted_mj_per_task'],'mJ/task',flush=True)
    finally: nv.nvmlShutdown()
    after_pred, after_meta=task.outputs()
    np.testing.assert_array_equal(before_pred,after_pred)
    # Learner exposes deterministic final state hashes; preserve both for audit.
    state_keys=[k for k in before_meta if 'sha256' in k or 'hash' in k]
    for key in state_keys: assert before_meta[key]==after_meta[key],key
    def summary(key):
        v=[r[key] for r in records]
        return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None,'values':v}
    return {'calibration_ms':calibration_ms,'repeats_per_trial':repeats,'trials':records,
            'summary':{k:summary(k) for k in ('idle_adjusted_mj_per_task','unadjusted_mj_per_task','cuda_ms_per_task','wall_ms_per_task')},
            'hardware':hardware,'software':{'python':platform.python_version(),'torch':str(torch.__version__),
                'numpy':np.__version__,'cuda':torch.version.cuda,'nvidia_ml_py':importlib.metadata.version('nvidia-ml-py')},
            'repeat_predictions_equal':True,'repeat_state_hashes_equal':True,'state_hash_keys_checked':state_keys,
            'scope':'Every task resets weights and momentum, normalizes raw9x9 inputs, trains from scratch, predicts allqueries; allinvocations included. GPUresident measurement excludes transfers, allocation, JIT/graphcapture, hostpreprocessing and verification.',
            'before_metadata':before_meta,'after_metadata':after_meta}
