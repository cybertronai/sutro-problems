"""Run original benchmark and an independent meter in one isolated A100 job."""
from pathlib import Path
import hashlib
import json
import sys
import modal

HERE = Path(__file__).resolve().parent
IMAGE = 'ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image = (modal.Image.from_registry(IMAGE)
         .pip_install('numpy==2.1.2', 'nvidia-ml-py==13.610.43')
         .add_local_dir(HERE/'source','/workspace/source')
         .add_local_dir(HERE/'payloads','/workspace/payloads')
         .add_local_file(HERE/'independent_measure.py','/workspace/independent_measure.py'))
app = modal.App('sutro-pca-qda-wandb-sxm40-20260915')

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,
              min_containers=0,max_containers=1,buffer_containers=0,
              scaledown_window=2,timeout=1200,retries=0)
def execute():
    import subprocess
    import contextlib
    import gc
    import os
    import runpy
    import traceback
    import torch
    import pynvml as nv
    assert torch.__version__.startswith('2.5.1'), torch.__version__
    out = Path('/tmp/audit-results')
    out.mkdir()
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    actual_name = str(nv.nvmlDeviceGetName(h))
    if actual_name != 'NVIDIA A100-SXM4-40GB':
        nv.nvmlShutdown()
        print('Hardware mismatch:', actual_name, flush=True)
        return [{'hardware_unavailable': True, 'requested': 'NVIDIA A100-SXM4-40GB', 'actual': actual_name}], {}
    def processes():
        return [{'pid':p.pid,'bytes':p.usedGpuMemory} for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]
    before_context = processes()
    assert not before_context, before_context
    torch.cuda.init()
    torch.cuda.synchronize()
    before_probe = processes()
    probe = torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    probe.fill_(3)
    torch.cuda.synchronize()
    after_probe = processes()
    assert len(after_probe)==1, after_probe
    reported_pid = after_probe[0]['pid']
    prior_bytes = sum(p['bytes'] for p in before_probe)
    assert after_probe[0]['bytes']-prior_bytes >= probe.numel(), (before_probe,after_probe)
    del probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_release = processes()
    assert after_probe[0]['bytes']-sum(p['bytes'] for p in after_release)>=128*1024*1024
    os.environ['SUTRO_NVML_SELF_PID']=str(reported_pid)
    records = [{'worker_pid':os.getpid(),'nvml_self_pid':reported_pid,
                'before_context':before_context,'before_probe':before_probe,
                'after_128MiB_probe':after_probe,'after_probe_release':after_release,
                'execution_mode':'Worker process. Controlled allocation/release establishes the NVML PID alias; guards still reject every other reported PID.'}]
    nv.nvmlShutdown()
    original = Path('/workspace/source/gpu_benchmark.py').read_text()
    old = 'if item.pid != os.getpid()]'
    assert original.count(old)==1
    instrumented = original.replace(old, "if item.pid not in {os.getpid(), int(os.environ['SUTRO_NVML_SELF_PID'])}]")
    instrumented_path = Path('/workspace/source/gpu_benchmark_instrumented.py')
    instrumented_path.write_text(instrumented)
    records[0]['instrumentation']='Only the process identity filter is changed to accept the experimentally established container PID alias. Learner, capture, replay counts, energy counter and idle-subtraction arithmetic are unchanged.'
    records[0]['original_runner_sha256']=hashlib.sha256(original.encode()).hexdigest()
    records[0]['instrumented_runner_sha256']=hashlib.sha256(instrumented.encode()).hexdigest()
    commands = [
        [sys.executable,str(instrumented_path),'/workspace/payloads',str(out/'original.json'),'3000','5'],
        [sys.executable,'/workspace/independent_measure.py','--output',str(out/'independent.json')],
    ]
    for name, command in zip(('original','independent'), commands):
        print('START',name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            code = 0
            original_argv = sys.argv
            sys.argv = command[1:]
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    runpy.run_path(command[1],run_name='__main__')
            except BaseException:
                traceback.print_exc(file=log)
                code = 1
            finally:
                sys.argv = original_argv
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        records.append({'name':name,'command':command,'returncode':code})
        print('FINISH',name,'returncode',code,flush=True)
        if code:
            print((out/(name+'.log')).read_text()[-4000:],flush=True)
    records.append({'nvidia_smi':subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True).stdout})
    return records, {p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}

@app.local_entrypoint()
def main():
    plan = {'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'source').iterdir()},
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'measurement_sha256':hashlib.sha256((HERE/'independent_measure.py').read_bytes()).hexdigest(),
            'gpu':'A100-40GB','max_containers':1,'timeout_seconds':1200,
            'original_protocol':'five rounds, 3000 replays, 5 s paired idle windows',
            'independent_protocol':'three 6000/12000/6000-replay rounds, 10 s paired idle windows, two 20 s idle-only shams; power sampling every 50 ms'}
    (HERE/'plan-sxm40.json').write_text(json.dumps(plan,indent=2)+'\n')
    records, files = execute.remote()
    destination = HERE/'results-sxm40'
    destination.mkdir(exist_ok=True)
    for name, data in files.items():
        (destination/name).write_bytes(data)
    (destination/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records[:-1],indent=2),flush=True)
