"""Whole-task A100 CUDA-event timing and paired-idle NVML energy measurement.

The caller supplies an already compiled GPU-resident invocation that resets all
learned state and completes the entire ensemble's training and inference. This
module does not alter the learner. Compilation/capture, transfers, allocation,
and fingerprint validation occur outside the measured intervals.
"""
from __future__ import annotations
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import statistics
import time
import uuid


def measure(invoke, fingerprint, *, trials=3, minimum_active_seconds=10,
            idle_seconds=3, settle_seconds=3, expected_fingerprint=None,
            invocations_per_trial=None):
    import torch
    import pynvml as nv
    def require(value,message):
        if not value:
            raise ValueError(message)
    require(trials >= 3, 'At least three trials are required')
    require(torch.cuda.device_count() == 1, 'This harness requires exactly one visible CUDA GPU')
    props = torch.cuda.get_device_properties(0)
    require('A100' in props.name, 'An actual A100 is required')
    # Torch 2.5 exposes a UUID object without NVML's GPU- prefix.
    raw_uuid = str(getattr(props,'uuid','')).removeprefix('GPU-')
    cuda_uuid = 'GPU-'+str(uuid.UUID(raw_uuid))
    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByUUID(cuda_uuid)
        def decode(value):
            return value.decode() if isinstance(value,bytes) else str(value)
        require(decode(nv.nvmlDeviceGetUUID(handle)) == cuda_uuid, 'CUDA and NVML device UUIDs disagree')
        def optional(call):
            try:
                return call()
            except nv.NVMLError:
                return None
        def stamp():
            before = time.perf_counter()
            counter = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
            after = time.perf_counter()
            return dict(counter_mj=counter,time_seconds=(before+after)/2,
                        read_duration_seconds=after-before)
        def interval(first,last):
            duration = last['time_seconds']-first['time_seconds']
            energy = last['counter_mj']-first['counter_mj']
            require(duration > 0 and energy >= 0, 'Invalid NVML counter interval')
            return dict(start=first,end=last,duration_seconds=duration,energy_mj=energy,
                        average_power_w=energy/duration/1000)
        def idle():
            torch.cuda.synchronize()
            time.sleep(settle_seconds)
            first = stamp(); time.sleep(idle_seconds)
            return dict(**interval(first,stamp()),settle_seconds=settle_seconds)
        def telemetry():
            return dict(temperature_c=nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                power_w=nv.nvmlDeviceGetPowerUsage(handle)/1000,
                graphics_clock_mhz=nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_GRAPHICS),
                memory_clock_mhz=nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM),
                throttle_mask=optional(lambda:int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle))))
        hardware = dict(cuda_name=props.name,nvml_name=decode(nv.nvmlDeviceGetName(handle)),
            cuda_uuid=cuda_uuid,nvml_uuid=decode(nv.nvmlDeviceGetUUID(handle)),uuid_match=True,
            pci_bus_id=decode(nv.nvmlDeviceGetPciInfo(handle).busId),
            total_memory_bytes=int(props.total_memory),multiprocessors=int(props.multi_processor_count),
            compute_capability=f'{props.major}.{props.minor}',
            power_limit_w=nv.nvmlDeviceGetPowerManagementLimit(handle)/1000,
            mig_mode=optional(lambda:list(nv.nvmlDeviceGetMigMode(handle))),
            compute_processes=optional(lambda:[dict(pid=p.pid,used_memory_bytes=p.usedGpuMemory)
                for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')),'unknown'))
        software = dict(python=platform.python_version(),platform=platform.platform(),
            torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
            driver=decode(nv.nvmlSystemGetDriverVersion()),nvml=decode(nv.nvmlSystemGetNVMLVersion()),
            nvidia_ml_py=importlib.metadata.version('nvidia-ml-py'))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if expected_fingerprint is not None:
            # Long training tasks need no extra full calibration invocation:
            # every timed run is compared directly to the frozen accuracy run.
            require(invocations_per_trial is not None and invocations_per_trial>=1,
                    'Supply a positive repetition count with a frozen reference')
            expected=expected_fingerprint
            repeats=invocations_per_trial
            calibration_ms=None
        else:
            require(invocations_per_trial is None,'Fixed repetitions require a frozen reference')
            expected=fingerprint()
            torch.cuda.synchronize()
            start_event.record(); invoke(); end_event.record(); end_event.synchronize()
            calibration_ms=start_event.elapsed_time(end_event)
            require(fingerprint() == expected, 'Fresh complete-task calibration changed the expected result')
            repeats=max(1,math.ceil(minimum_active_seconds*1000/calibration_ms))
        records = []
        for index in range(trials):
            baseline_before = idle()
            telemetry_before = telemetry()
            first = stamp()
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(repeats):
                invoke()
            end_event.record(); end_event.synchronize()
            wall_end = time.perf_counter()
            last = stamp()
            telemetry_after = telemetry()
            active = interval(first,last)
            # No GPU validation work is placed between the active and idle periods.
            baseline_after = idle()
            actual = fingerprint()
            require(actual == expected, 'A timed full training invocation did not reproduce the expected result')
            paired_power = (baseline_before['average_power_w']+baseline_after['average_power_w'])/2
            adjusted = active['energy_mj']-paired_power*active['duration_seconds']*1000
            cuda_ms = start_event.elapsed_time(end_event)
            require(cuda_ms>=minimum_active_seconds*1000*.8,'Active trial too short for the declared protocol')
            wall_ms = (wall_end-wall_start)*1000
            require(.8 < cuda_ms/wall_ms < 1.2, 'CUDA and wall-clock timing disagree')
            records.append(dict(trial=index+1,invocations=repeats,cuda_ms_per_task=cuda_ms/repeats,
                wall_ms_per_task=wall_ms/repeats,wall_start_seconds=wall_start,wall_end_seconds=wall_end,
                active=active,idle_before=baseline_before,idle_after=baseline_after,
                telemetry_before=telemetry_before,telemetry_after=telemetry_after,paired_idle_power_w=paired_power,
                gross_energy_mj_per_task=active['energy_mj']/repeats,idle_adjusted_energy_mj_per_task=adjusted/repeats,
                before_only_adjusted_mj_per_task=(active['energy_mj']-baseline_before['average_power_w']*active['duration_seconds']*1000)/repeats,
                after_only_adjusted_mj_per_task=(active['energy_mj']-baseline_after['average_power_w']*active['duration_seconds']*1000)/repeats,
                repeated_full_state_fingerprint_matches=True))
            print(f'Whole-task trial {index+1}/{trials}: {cuda_ms/repeats:.2g} ms, {adjusted/repeats:.2g} mJ',flush=True)
        def summary(key):
            values=[row[key] for row in records]
            return dict(median=statistics.median(values),mean=statistics.mean(values),
                min=min(values),max=max(values),sample_standard_deviation=statistics.stdev(values))
        return dict(schema='sutro-complete-task-a100/1',hardware=hardware,software=software,
            calibration_ms=calibration_ms,invocations_per_trial=repeats,trials=records,
            summary={key:summary(key) for key in ('cuda_ms_per_task','wall_ms_per_task',
                'idle_adjusted_energy_mj_per_task','gross_energy_mj_per_task')},
            state_fingerprint=expected,
            protocol=dict(trials=trials,minimum_active_seconds=minimum_active_seconds,
                baseline='Mean of paired NVML cumulative-counter idle powers, applied to the same active counter interval',
                idle_seconds=idle_seconds,settle_seconds=settle_seconds,
                task='All members freshly initialized, fully trained, and evaluated; ensemble predictions produced',
                time_units='ms',energy_units='mJ',energy_scope='NVML whole GPU board; host CPU excluded',
                included=['full learned-state reset','input normalization','target construction',
                    'all training minibatches','all query predictions','ensemble arithmetic','graph launch overhead'],
                excluded=['input host/device transfers','allocation','compilation','graph capture',
                    'seed-only program compilation','fingerprint validation','cold start'],
                caveats=['GPU-resident repetitions may reuse caches','Paired idle power may drift',
                    'Integrated board telemetry does not measure isolated arithmetic energy']))
    finally:
        nv.nvmlShutdown()
