"""Cache sizing, fresh-task energy, and actual Nsight Compute counters."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import sys
import modal
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import data_reference as data
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
NCU_URL='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2025.1.1_2025.1.1.2-1_amd64.deb'
NCU_SHA='d386c53f5452ddfaec8de3d7240eba1e0d8ffb690aae6585c77d60199e9b2b05'
image=(modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
    .run_commands(f"python -c \"import urllib.request,hashlib,pathlib; p=pathlib.Path('/tmp/ncu.deb'); urllib.request.urlretrieve('{NCU_URL}',p); assert hashlib.sha256(p.read_bytes()).hexdigest()=='{NCU_SHA}'\"",
                  'dpkg-deb -x /tmp/ncu.deb /','rm /tmp/ncu.deb')
    .env({'PYTHONPATH':'/workspace/base:/workspace/cache'}))
if modal.is_local():
    for name in ('learner.py','energy.py','data_reference.py'):
        image=image.add_local_file(BASE/name,'/workspace/base/'+name)
    image=image.add_local_file(HERE/'fixture.py','/workspace/cache/fixture.py')
app=modal.App('sutro-rev88-cache-audit-20260914')
METRICS='dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def pack(a):return {'bytes':a.tobytes(),'shape':a.shape,'dtype':str(a.dtype)}

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,min_containers=0,max_containers=3,
              buffer_containers=0,scaledown_window=2,timeout=1200,retries=0)
def execute(payload,config,workspace,action='measure',scope='task',with_energy=False):
    import subprocess,glob
    root=Path('/tmp/cache-run');root.mkdir(exist_ok=True)
    arrays={k:np.frombuffer(v['bytes'],dtype=v['dtype']).reshape(v['shape']).copy() for k,v in payload.items()}
    assert set(arrays)=={'train_images','train_labels','test_images'}
    np.savez(root/'input.npz',**arrays);write(root/'config.json',config)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=workspace)
    command=[sys.executable,'/workspace/cache/fixture.py','--input',str(root/'input.npz'),
             '--config',str(root/'config.json'),'--output',str(root/'result.json'),
             '--mode','profile' if action=='profile' else 'measure','--scope',scope,'--repeats','16']
    if with_energy:command+=['--energy']
    ncu=next(iter(glob.glob('/opt/nvidia/nsight-compute/*/ncu')),None)
    capabilities={}
    if ncu:
        version=subprocess.run([ncu,'--version'],text=True,capture_output=True,timeout=20)
        capabilities['ncu_version']=version.stdout+version.stderr
    if action=='profile':
        assert ncu,'Nsight Compute installation missing'
        bases=','.join(m.rsplit('.',1)[0] for m in METRICS.split(','))
        query=subprocess.run([ncu,'--query-metrics','--query-metrics-mode','all','--metrics',bases],capture_output=True,text=True,timeout=60)
        capabilities['metric_query']={'returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr}
        command=[ncu,'--replay-mode','app-range','--cache-control','none','--clock-control','none',
                 '--metrics',METRICS,'--csv','--page','raw',
                 '--force-overwrite','--export',str(root/'profile')]+command
    for path in (root/'result.json',root/'result.predictions.npy',root/'profile.ncu-rep'):
        path.unlink(missing_ok=True)
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1000)
    out={'workspace':workspace,'action':action,'scope':scope,'command':command,'returncode':result.returncode,
         'stdout':result.stdout,'stderr':result.stderr,'capabilities':capabilities,'finished_at_utc':utc(),
         'input_sha256':{k:ah(v) for k,v in arrays.items()}}
    if (root/'result.json').exists():out['result']=json.loads((root/'result.json').read_text())
    artifacts={}
    for name in ('result.predictions.npy','profile.ncu-rep'):
        path=root/name
        if path.exists():artifacts[name]=path.read_bytes()
    return out,artifacts

def data_draw(seed,raw_dir):
    paths={k:data.download_source(raw_dir,*data.SOURCES[k]) for k in ('train_images','train_labels')}
    images=data.read_idx(paths['train_images'],60000,images=True)
    labels=data.read_idx(paths['train_labels'],60000,images=False)
    order=np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    def resize(ix):
        x=data.area_resize(images[ix].astype(np.float32)/np.float32(255),9);np.clip(x,0,1,out=x);return x[:,None,:,:]
    return {'train_images':resize(order[:10000]),'train_labels':labels[order[:10000]].astype(np.int64),
            'test_images':resize(order[10000:20000])}

@app.local_entrypoint()
def main(phase:str='compare',raw_dir:str='mnist/data/raw'):
    config=json.loads((BASE/'config.json').read_text())
    original=json.loads((BASE/'protocol.json').read_text())
    for name,digest in original['source_sha256'].items():assert sha(BASE/name)==digest,name
    if phase=='compare':
        plan={'created_at_utc':utc(),'base_protocol_sha256':sha(BASE/'protocol.json'),
              'fixture_sha256':sha(HERE/'fixture.py'),'runner_sha256':sha(HERE/'run.py'),
              'workspaces':[':4096:8',':16:8',':0:0'],'preferred_upgrade':':16:8',
              'policy':'Change library workspace only; no architecture/training/seed changes. Compare complete outputs and state with frozen qualification.',
              'ncu_url':NCU_URL,'ncu_sha256':NCU_SHA,'max_gpu_workers':3}
        write(HERE/'plan.json',plan)
        arrays=data_draw(original['dataset_seeds'][0],Path(raw_dir));payload={k:pack(v) for k,v in arrays.items()}
        jobs=[(payload,config,w,'measure','task',True) for w in plan['workspaces']]
        names=['baseline','small-workspace','zero-workspace']
    elif phase=='profile':
        write(HERE/'profile-plan.json',{'created_at_utc':utc(),'runner_sha256':sha(HERE/'run.py'),
              'fixture_sha256':sha(HERE/'fixture.py'),'metrics':METRICS,'replay_mode':'app-range',
              'cache_control':'none','clock_control':'none','range_selection':'CUDAProfilerStart/Stop',
              'amendment':'Removed unsupported profile-from-start flag after ncu2025.1.1 rejected initial commands; no learner or fixture changes.'})
        arrays=data_draw(original['dataset_seeds'][0],Path(raw_dir));payload={k:pack(v) for k,v in arrays.items()}
        jobs=[(payload,config,w,'profile',scope,False) for w in (':4096:8',':16:8') for scope in ('task','training')]
        names=['profile-baseline-task','profile-baseline-training','profile-small-task','profile-small-training']
    elif phase=='qualify':
        jobs=[];names=[]
        for i,seed in enumerate(original['dataset_seeds']):
            arrays=data_draw(seed,Path(raw_dir));payload={k:pack(v) for k,v in arrays.items()}
            jobs.append((payload,config,':16:8','measure','task',False));names.append(f'qualify-{i:02}')
    else:raise ValueError(phase)
    for i,(out,artifacts) in enumerate(execute.starmap(jobs,order_outputs=True)):
        stem=names[i];write(HERE/(stem+'.json'),out)
        for name,blob in artifacts.items():(HERE/(stem+'-'+name)).write_bytes(blob)
        print(stem,'returncode',out['returncode'],flush=True)
        if out['returncode']!=0:print(out['stdout'][-6000:]+out['stderr'][-6000:],flush=True)
