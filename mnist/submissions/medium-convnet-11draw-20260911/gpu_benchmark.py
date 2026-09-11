"""Measure the unchanged complete three-member learner after the accuracy gate.

The task includes fresh training, input/output copies, training diagnostics,
parameter hashing, checkpoint serialization, and the CPU float64 ensemble.
Reference outputs are used only by the host-side verifier, outside the learner
and measured intervals. No test labels are loaded anywhere in this command.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import modal

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import learner

NVML_VERSION='12.560.30'
image=(modal.Image.from_registry(learner.IMAGE_REF)
       .pip_install('numpy==2.2.6',f'nvidia-ml-py=={NVML_VERSION}')
       .add_local_file(str(HERE/'learner.py'),remote_path='/root/learner.py'))
app=modal.App('sutro-medium-convnet-complete-task-measurement')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1800,
              startup_timeout=300,min_containers=0,max_containers=1,
              buffer_containers=0,scaledown_window=2,retries=0)
def measure(payload,provenance,oracle_payload,reference,measurement_protocol):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import importlib.util
    import importlib.metadata
    import platform
    import statistics
    import time
    import numpy as np
    import torch
    import pynvml as nv
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('unchanged_learner','/root/learner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(Path('/root/learner.py'))==measurement_protocol['learner_sha256']
    assert measurement_protocol['warmup_tasks']==1 and measurement_protocol['measured_tasks']==3
    assert set(payload)=={'train_images','train_labels','test_images'}
    arrays={name:np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
            for name,value in payload.items()}
    for name,array in arrays.items():assert module.array_hash(array)==provenance['input_sha256'][name]
    # CPU-only oracle buffers are never passed to the learner or moved to CUDA.
    oracle={name:np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
            for name,value in oracle_payload.items()}
    expected={member['seed']:member for member in reference['members']}
    assert sorted(expected)==module.MEMBER_SEEDS==[101,102,103]
    for member in expected.values():assert member['provenance']==provenance
    nv.nvmlInit()
    assert torch.cuda.device_count()==nv.nvmlDeviceGetCount()==1, 'Expected one matching CUDA/NVML GPU'
    handle=nv.nvmlDeviceGetHandleByIndex(0)

    def safe(fn,*args):
        try:
            value=fn(*args)
            return value.decode() if isinstance(value,bytes) else value
        except nv.NVMLError as error:return {'unavailable':str(error)}

    def device_snapshot():
        memory=nv.nvmlDeviceGetMemoryInfo(handle)
        process_list=safe(nv.nvmlDeviceGetComputeRunningProcesses,handle)
        processes=process_list if isinstance(process_list,dict) else [
            {'pid':int(p.pid),'used_gpu_memory_bytes':int(p.usedGpuMemory)} for p in process_list]
        return {'at_utc':datetime.now(timezone.utc).isoformat(),
            'temperature_c':safe(nv.nvmlDeviceGetTemperature,handle,nv.NVML_TEMPERATURE_GPU),
            'graphics_clock_mhz':safe(nv.nvmlDeviceGetClockInfo,handle,nv.NVML_CLOCK_GRAPHICS),
            'sm_clock_mhz':safe(nv.nvmlDeviceGetClockInfo,handle,nv.NVML_CLOCK_SM),
            'memory_clock_mhz':safe(nv.nvmlDeviceGetClockInfo,handle,nv.NVML_CLOCK_MEM),
            'power_usage_mw':safe(nv.nvmlDeviceGetPowerUsage,handle),
            'performance_state':safe(nv.nvmlDeviceGetPerformanceState,handle),
            'clocks_throttle_reasons':safe(nv.nvmlDeviceGetCurrentClocksThrottleReasons,handle),
            'memory_used_bytes':int(memory.used),'memory_free_bytes':int(memory.free),
            'compute_processes':processes}

    def energy_stamp():
        before=time.perf_counter()
        value=int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
        after=time.perf_counter()
        return {'energy_mj':value,'time_s':(before+after)/2,
                'api_started_s':before,'api_finished_s':after,'api_duration_s':after-before}

    def interval(first,last):
        duration=last['time_s']-first['time_s'];energy=last['energy_mj']-first['energy_mj']
        assert duration>0 and energy>=0
        return {'first':first,'last':last,'duration_s':duration,'gross_energy_mj':energy,
                'average_power_w':energy/1000/duration}

    def idle_sample():
        torch.cuda.synchronize()
        time.sleep(measurement_protocol['idle_settle_seconds'])
        first=energy_stamp()
        time.sleep(measurement_protocol['idle_sample_seconds'])
        result=interval(first,energy_stamp())
        result['settle_seconds']=measurement_protocol['idle_settle_seconds']
        result['requested_sample_seconds']=measurement_protocol['idle_sample_seconds']
        return result

    def run_task():
        members=[];member_logits=[]
        for seed in module.MEMBER_SEEDS:
            # Unchanged function; even keep_checkpoint=False still computes its
            # tensor hashes and serializes/hashes the checkpoint before discarding bytes.
            result,discarded_checkpoint,logits=module.train_member(
                arrays,module.CONFIG,seed,provenance,keep_checkpoint=False)
            assert discarded_checkpoint==b''
            members.append(result);member_logits.append(logits)
        ensemble=np.mean(np.stack(member_logits),axis=0,dtype=np.float64)
        predictions=ensemble.argmax(1).astype(np.int64)
        return members,member_logits,ensemble,predictions

    def verify(output,label):
        members,values,ensemble,predictions=output
        checks=[]
        for result,logits in zip(members,values):
            seed=result['seed'];truth=expected[seed];oracle_logits=oracle[f'seed{seed}']
            check={'seed':seed,
                'reference_logit_hash_unchanged':module.array_hash(oracle_logits)==truth['logits_sha256'],
                'logits_bitwise_match':module.array_hash(logits)==truth['logits_sha256'],
                'prediction_hash_match':result['predictions_sha256']==truth['predictions_sha256'],
                'state_tensor_hashes_match':result['state_tensor_sha256']==truth['state_tensor_sha256'],
                'normalization_match':result['normalization']==truth['normalization'],
                'configuration_and_extent_match':result['config']==truth['config'] and result['epochs']==71}
            assert all(v for k,v in check.items() if k!='seed'),check
            assert result['parameter_count']==truth['parameter_count']==1404618
            assert np.array_equal(logits.view(np.uint32),oracle_logits.view(np.uint32))
            checks.append(check)
        ensemble_match=module.array_hash(ensemble)==reference['ensemble_logits_sha256']
        prediction_match=module.array_hash(predictions)==reference['predictions_sha256']
        assert ensemble_match and prediction_match
        assert module.array_hash(oracle['ensemble'])==reference['ensemble_logits_sha256']
        assert np.array_equal(ensemble.view(np.uint64),oracle['ensemble'].view(np.uint64))
        return {'label':label,'passed':True,'members':checks,
            'ensemble_bitwise_match':ensemble_match,'ensemble_prediction_hash_match':prediction_match,
            'checked_at_utc':datetime.now(timezone.utc).isoformat()}

    props=torch.cuda.get_device_properties(0)
    hardware={'gpu':torch.cuda.get_device_name(0),'device_uuid':safe(nv.nvmlDeviceGetUUID,handle),
        'driver':safe(nv.nvmlSystemGetDriverVersion),'nvml':safe(nv.nvmlSystemGetNVMLVersion),
        'cuda_capability':list(torch.cuda.get_device_capability(0)),
        'memory_total_bytes':int(props.total_memory),'multiprocessors':props.multi_processor_count,
        'power_limit_mw':safe(nv.nvmlDeviceGetPowerManagementLimit,handle),
        'default_power_limit_mw':safe(nv.nvmlDeviceGetPowerManagementDefaultLimit,handle),
        'max_sm_clock_mhz':safe(nv.nvmlDeviceGetMaxClockInfo,handle,nv.NVML_CLOCK_SM),
        'max_memory_clock_mhz':safe(nv.nvmlDeviceGetMaxClockInfo,handle,nv.NVML_CLOCK_MEM),
        'exclusive_container':'One Modal A100 allocation; max_containers=1; no concurrent benchmark jobs',
        'clocks_policy':'Unmodified platform defaults; no frequency or power-limit locking'}
    versions={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'nvidia_ml_py':importlib.metadata.version('nvidia-ml-py'),'container_image':module.IMAGE_REF}
    initial_hardware=device_snapshot()
    print('Complete-task warmup: three fresh71-epoch members and FP64 ensemble.',flush=True)
    last_output=run_task();torch.cuda.synchronize()
    warmup_validation=verify(last_output,'warmup')
    trials=[];retained_outputs=[]
    for trial in range(3):
        before_validation=verify(last_output,f'before_trial_{trial}:reference_and_previous_complete_task')
        before_hardware=device_snapshot();idle_before=idle_sample()
        torch.cuda.reset_peak_memory_stats()
        start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        first=energy_stamp()
        started=time.perf_counter();start_event.record()
        current=run_task()
        end_event.record();end_event.synchronize();finished=time.perf_counter()
        last=energy_stamp()
        cuda_ms=float(start_event.elapsed_time(end_event));wall_ms=(finished-started)*1000
        active=interval(first,last)
        peak={'allocated_bytes':int(torch.cuda.max_memory_allocated()),'reserved_bytes':int(torch.cuda.max_memory_reserved())}
        after_validation=verify(current,f'after_trial_{trial}:new_complete_task')
        idle_after=idle_sample();after_hardware=device_snapshot()
        idle_power=(idle_before['average_power_w']+idle_after['average_power_w'])/2
        adjusted=active['gross_energy_mj']-idle_power*active['duration_s']*1000
        row={'trial':trial,'tasks':1,'wall_ms_per_task':wall_ms,'cuda_event_ms_per_task':cuda_ms,
            'gross_energy_mj_per_task':active['gross_energy_mj'],
            'idle_adjusted_energy_mj_per_task':adjusted,'paired_idle_power_w':idle_power,
            'idle_before':idle_before,'idle_after':idle_after,'active':active,
            'adjusted_with_before_only_mj':active['gross_energy_mj']-idle_before['average_power_w']*active['duration_s']*1000,
            'adjusted_with_after_only_mj':active['gross_energy_mj']-idle_after['average_power_w']*active['duration_s']*1000,
            'wall_timer_started_s':started,'wall_timer_finished_s':finished,
            'wall_timer_includes_bounding_nvml_api_calls':False,'energy_interval_includes_boundary_api_latency':True,
            'peak_torch_memory':peak,'hardware_before':before_hardware,'hardware_after':after_hardware,
            'before_validation':before_validation,'after_validation':after_validation}
        trials.append(row)
        retained_outputs.append({'trial':trial,'member_results':current[0],
            'ensemble_logits_sha256':module.array_hash(current[2]),'predictions_sha256':module.array_hash(current[3])})
        last_output=current
        print(json.dumps({'trial':trial,'wall_ms':wall_ms,'cuda_ms':cuda_ms,
            'idle_adjusted_energy_mj':adjusted,'gross_energy_mj':active['gross_energy_mj'],
            'all_reference_checks_passed':True}),flush=True)
    summary={}
    for field in ('wall_ms_per_task','cuda_event_ms_per_task','idle_adjusted_energy_mj_per_task','gross_energy_mj_per_task'):
        vals=[t[field] for t in trials]
        summary[field+'_median']=statistics.median(vals)
        summary[field+'_mean']=statistics.mean(vals)
        summary[field+'_sample_sd']=statistics.stdev(vals)
        summary[field+'_min']=min(vals);summary[field+'_max']=max(vals)
    result={'schema_version':1,'protocol':measurement_protocol,'provenance':provenance,
        'hardware':hardware,'versions':versions,'initial_hardware':initial_hardware,
        'warmup_validation':warmup_validation,'trials':trials,'summary':summary,
        'validation':{'all_logits_predictions_and_state_hashes_match_draw00':True,'warmup_tasks':1,'measured_tasks':3,
            'oracles':'CPU only; never passed to learner; verification outside wall and energy intervals'},
        'scope':{'primary_time':'Complete-task wall milliseconds, excluding bounding NVML calls',
            'cuda_time':'CUDA event elapsed interval around the complete task, including GPU idle gaps caused by host work',
            'energy':'NVML whole-board cumulative mJ delta minus mean paired idle power times counter midpoint interval',
            'includes':['three fresh initializations','all71epochs per member','all allowed input and output copies',
                'per-epoch clean training diagnostics','parameter/output hashes','checkpoint serialization',
                'CPU float64 three-logit mean and argmax','CUDA timing event record/synchronization overhead'],
            'excludes':['Modal startup','image/package initialization','dataset decoding and preprocessing',
                'source/input validation in benchmark wrapper','reference comparisons','file writes','idle sampling intervals'],
            'energy_excludes':'Host CPU energy, even though host work contributes to elapsed task time',
            'repeat_boundary':'Fresh learner/optimizer/scheduler/RNG each task; framework handles/allocator caches are warm',
            'sample_scope':'One predeclared draw00 on one A100; accuracy was assessed separately over all11draws'},
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'test_labels_opened':False}
    nv.nvmlShutdown()
    return result,retained_outputs


@app.local_entrypoint()
def main(study: str='',output: str=''):
    import numpy as np
    root=Path(study) if study else HERE;destination=Path(output) if output else root/'benchmark'
    protocol=json.loads((root/'protocol.json').read_text());accuracy=json.loads((root/'accuracy.json').read_text())
    assert accuracy['meets_accuracy_target'] is True and accuracy['total_correct']>=64680
    assert accuracy['total_predictions']==66000 and accuracy['total_draws']==11
    assert accuracy['prediction_manifest_sha256']==sha(root/'prediction_manifest.json')
    assert accuracy['protocol_sha256']==sha(root/'protocol.json')
    assert protocol['config']==learner.CONFIG and protocol['member_seeds']==[101,102,103]
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest
    draw=json.loads((root/'draws/draw-00.json').read_text())
    reference=json.loads((root/'results/draw-00.json').read_text())
    frozen=json.loads((root/'prediction_manifest.json').read_text())['predictions'][0]
    assert frozen['draw_index']==0 and reference['dataset_seed']==draw['dataset_seed']==20261001
    assert sha(root/frozen['result_path'])==frozen['result_sha256']
    assert sha(root/reference['logits_file'])==reference['logits_file_sha256']==frozen['logits_sha256']
    assert sha(root/draw['allowed_archive'])==draw['allowed_archive_sha256']
    with np.load(root/draw['allowed_archive'],allow_pickle=False) as archive:
        assert set(archive.files)=={'train_images','train_labels','test_images'}
        arrays={name:archive[name] for name in archive.files}
    for name,array in arrays.items():assert learner.array_hash(array)==draw['arrays'][name]['sha256']
    with np.load(root/reference['logits_file'],allow_pickle=False) as archive:
        oracle={name:archive[name] for name in ('seed101','seed102','seed103','ensemble')}
    def encode(values):return {name:{'shape':v.shape,'dtype':str(v.dtype),'bytes':np.ascontiguousarray(v).tobytes()} for name,v in values.items()}
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise FileExistsError('Measurement output must be empty; no overwritten/retried trials')
    measurement_protocol={'schema_version':1,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'benchmark_source_sha256':sha(Path(__file__)),'learner_sha256':sha(HERE/'learner.py'),
        'accuracy_gate_sha256':sha(root/'accuracy.json'),'accuracy_correct':accuracy['total_correct'],
        'accuracy_total':66000,'accuracy_target_percent':'98','accuracy_gate_passed':True,
        'draw_index':0,'dataset_seed':20261001,'config':learner.CONFIG,'epochs':71,'schedule_epochs':100,
        'member_seeds':[101,102,103],'warmup_tasks':1,'measured_tasks':3,'max_containers':1,
        'idle_settle_seconds':3,'idle_sample_seconds':3,
        'idle_energy_formula':'active counter delta mJ - mean(before idle W, after idle W) * active counter-midpoint seconds *1000',
        'primary_summary':'Median across3trials; retain means, sample SD and raw trials',
        'reference_result_sha256':sha(root/'results/draw-00.json'),'reference_logits_sha256':sha(root/reference['logits_file']),
        'allowlist_archive_sha256':draw['allowed_archive_sha256'],'container_image':learner.IMAGE_REF,
        'numpy_version':'2.2.6','nvidia_ml_py_version':NVML_VERSION,
        'learner_policy':'Call frozen train_member unchanged three times; no optimizations or arithmetic changes',
        'clock_policy':'Platform defaults, unmodified',
        'stop_policy':'Any output/state mismatch aborts without measurements.json; no selective trial dropping'}
    write_new(destination/'measurement_protocol.json',measurement_protocol)
    result,outputs=measure.remote(encode(arrays),reference['members'][0]['provenance'],encode(oracle),reference,measurement_protocol)
    assert sha(Path(__file__))==measurement_protocol['benchmark_source_sha256'], 'Benchmark source changed during measurement'
    assert sha(HERE/'learner.py')==measurement_protocol['learner_sha256'], 'Learner source changed during measurement'
    assert result['protocol']==measurement_protocol and result['validation']['all_logits_predictions_and_state_hashes_match_draw00']
    result['benchmark_source_sha256']=sha(Path(__file__))
    write_new(destination/'measurements.json',result)
    write_new(destination/'task_outputs.json',{'protocol_sha256':sha(destination/'measurement_protocol.json'),'tasks':outputs})
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':raise SystemExit('Run using modal run '+__file__)
