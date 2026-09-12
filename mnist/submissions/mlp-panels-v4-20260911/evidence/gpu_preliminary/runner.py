"""Three-way paired A100 experiment; explicitly authorized vargapowercouple only.

Preserves frozen baseline kernels, complete-task boundaries and NVML protocol.
Panel kernels are GPU SIMD implementations, not a simulation of v4 distances.
CUDA runtime capture permits independent verification of all 24,004 graph nodes.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
app = modal.App('mnist-panels-three-way-a100')
MODES = ('baseline','energy','no_slowdown')
ORDERS = (MODES,('energy','no_slowdown','baseline'),('no_slowdown','baseline','energy'))


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,startup_timeout=300,
              min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=2,retries=0)
def benchmark(source,payload,expected,provenance):
    import ctypes as ct
    import importlib.util
    import platform
    import re
    import statistics
    import time
    import numpy as np
    import torch
    import triton
    import pynvml as nv

    torch.set_num_threads(4)
    path = Path('/tmp/panel_kernels.py')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('panel_kernels',path)
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    data = np.load(io.BytesIO(payload),allow_pickle=False)
    x = torch.from_numpy(data['train_images'].reshape(600,9).copy()).cuda()
    q = torch.from_numpy(data['test_images'].reshape(600,9).copy()).cuda()
    y = torch.from_numpy(data['train_labels'].astype(np.int32)).cuda()
    nx,nq = torch.empty_like(x),torch.empty_like(q)
    rng = np.random.Generator(np.random.PCG64(101))
    initial_np = np.concatenate([rng.uniform(-1/3,1/3,(9,32)).astype(np.float32).ravel(),np.zeros(32,np.float32),
        rng.uniform(-1/np.sqrt(32),1/np.sqrt(32),(32,10)).astype(np.float32).ravel(),np.zeros(10,np.float32)])
    initial = torch.from_numpy(initial_np).cuda()
    params = torch.empty_like(initial)
    target = torch.empty((600,10),dtype=torch.float32,device='cuda')
    hidden = torch.empty((30,32),dtype=torch.float32,device='cuda')
    d1 = torch.empty_like(hidden)
    cache_and_delta = torch.empty(570,dtype=torch.float32,device='cuda')
    xc = cache_and_delta[:270]
    d2 = cache_and_delta[270:].view(30,10)
    ih = torch.empty((600,32),dtype=torch.float32,device='cuda')
    scores = torch.empty((600,10),dtype=torch.float32,device='cuda')
    output = torch.empty(600,dtype=torch.int32,device='cuda')
    kw = {'num_warps':4,'enable_fp_fusion':False}
    step = float(np.float32(.2/30))
    compiled = {}

    def invoke(mode,epochs=300):
        compiled['normalize'] = k.normalize_kernel[(triton.cdiv(5400,128),)](x,q,nx,nq,128,**kw)
        compiled['initialize'] = k.initialize_kernel[(triton.cdiv(6000,128),)](initial,params,y,target,128,**kw)
        for epoch in range(epochs):
            for first in range(0,600,30):
                if mode=='baseline':
                    compiled['baseline_hidden'] = k.hidden_kernel[(triton.cdiv(960,128),)](nx,params,hidden,first,128,**kw)
                    compiled['baseline_output'] = k.delta2_kernel[(triton.cdiv(300,128),)](hidden,params,target,d2,first,128,**kw)
                    compiled['baseline_backward'] = k.delta1_kernel[(triton.cdiv(960,128),)](hidden,params,d2,d1,128,**kw)
                    compiled['baseline_update'] = k.update_kernel[(triton.cdiv(650,128),)](nx,hidden,d1,d2,params,first,step,128,**kw)
                else:
                    compiled[mode+'_hidden'] = k.hidden_cached[(1,)](nx,xc,params,hidden,first,mode=='energy',**kw)
                    compiled['panel_output'] = k.output_panel[(4,)](hidden,params,target,d2,first,**kw)
                    compiled['panel_backward'] = k.backward_panel[(8,)](hidden,params,d2,d1,**kw)
                    if mode=='energy':
                        compiled['energy_update'] = k.update_panels[(12,)](xc,hidden,d1,d2,params,step,**kw)
                    else:
                        compiled['no_slowdown_update'] = k.update_kernel[(triton.cdiv(650,128),)](xc,hidden,d1,d2,params,0,step,128,**kw)
        if mode=='baseline':
            compiled['baseline_infer_hidden'] = k.inference_hidden_kernel[(triton.cdiv(19200,128),)](nq,params,ih,128,**kw)
            compiled['baseline_infer_output'] = k.inference_output_kernel[(triton.cdiv(600,128),)](ih,params,scores,output,128,**kw)
        elif mode=='energy':
            compiled['energy_infer_hidden'] = k.inference_hidden_panels[(20,)](nq,params,ih,**kw)
            compiled['energy_infer_output'] = k.inference_output_panels[(20,)](ih,params,scores,output,**kw)
        else:
            compiled['no_slowdown_infer_hidden'] = k.inference_hidden_cached[(150,)](nq,params,ih,**kw)
            compiled['no_slowdown_infer_output'] = k.inference_output_kernel[(triton.cdiv(600,128),)](ih,params,scores,output,128,**kw)

    arrays,checks = {},{}
    def validate(mode,case,save=False):
        current = {'params':params.cpu().numpy(),'scores':scores.cpu().numpy(),'predictions':output.cpu().numpy()}
        record = {}
        for name,value in current.items():
            assert np.isfinite(value).all(),(mode,case,name,'nonfinite')
            reference = np.frombuffer(expected[case][name],dtype=value.dtype).reshape(value.shape)
            equal = value.view(np.uint32)==reference.view(np.uint32)
            if not equal.all():
                positions = np.flatnonzero(~equal.ravel())[:8]
                raise AssertionError((mode,case,name,int(equal.sum()),value.size,positions.tolist(),
                                      value.ravel()[positions].tolist(),reference.ravel()[positions].tolist()))
            record[name] = {'count':value.size,'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
            if save:
                arrays[mode+'_'+name] = value.copy()
        return record

    libraries = [str(p) for p in (Path(torch.__file__).parent.parent/'nvidia/cuda_runtime/lib').glob('libcudart.so*')]
    libraries += ['libcudart.so.12','/usr/local/cuda/lib64/libcudart.so.12']
    runtime = None
    for library in libraries:
        try:
            runtime = ct.CDLL(library)
            break
        except OSError:
            pass
    assert runtime is not None,'CUDA runtime library unavailable'
    pointer = ct.c_void_p
    signatures = {
        'cudaStreamBeginCapture':[pointer,ct.c_int],
        'cudaStreamEndCapture':[pointer,ct.POINTER(pointer)],
        'cudaGraphGetNodes':[pointer,ct.POINTER(pointer),ct.POINTER(ct.c_size_t)],
        'cudaGraphNodeGetType':[pointer,ct.POINTER(ct.c_int)],
        'cudaGraphInstantiateWithFlags':[ct.POINTER(pointer),pointer,ct.c_ulonglong],
        'cudaGraphLaunch':[pointer,pointer], 'cudaGraphDestroy':[pointer], 'cudaGraphExecDestroy':[pointer]}
    for name,args in signatures.items():
        fn = getattr(runtime,name)
        fn.argtypes,fn.restype = args,ct.c_int
    def call(name,*args):
        code = getattr(runtime,name)(*args)
        assert code==0,(name,code)
    def stream_pointer():
        return pointer(torch.cuda.current_stream().cuda_stream)
    graphs,graph_counts = {},{}
    def replay(mode):
        call('cudaGraphLaunch',graphs[mode],stream_pointer())
    for mode in MODES:
        print('One-epoch compiler preflight',mode,flush=True)
        try:
            invoke(mode,epochs=1); torch.cuda.synchronize()
        except Exception as error:
            raise RuntimeError(f'{mode}: {type(error).__name__}: {error}') from None
    for mode in MODES:
        print('Compile, execute, validate',mode,flush=True)
        try:
            invoke(mode); torch.cuda.synchronize()
        except Exception as error:
            raise RuntimeError(f'{mode}: {type(error).__name__}: {error}') from None
        checks[mode+'_eager'] = validate(mode,'canonical',True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke(mode); torch.cuda.synchronize()
            call('cudaStreamBeginCapture',stream_pointer(),0)
            invoke(mode)
            graph = pointer()
            call('cudaStreamEndCapture',stream_pointer(),ct.byref(graph))
        n = ct.c_size_t()
        call('cudaGraphGetNodes',graph,None,ct.byref(n))
        nodes = (pointer*n.value)()
        call('cudaGraphGetNodes',graph,nodes,ct.byref(n))
        kernel_count = 0
        for node in nodes:
            kind = ct.c_int()
            call('cudaGraphNodeGetType',node,ct.byref(kind))
            kernel_count += kind.value==0
        assert n.value==kernel_count==24004,(mode,n.value,kernel_count)
        executable = pointer()
        call('cudaGraphInstantiateWithFlags',ct.byref(executable),graph,0)
        call('cudaGraphDestroy',graph)
        graphs[mode] = executable
        graph_counts[mode] = {'total':n.value,'kernels':kernel_count}
        replay(mode); torch.cuda.synchronize()
        checks[mode+'_graph'] = validate(mode,'canonical')
        print('Graph accepted',mode,graph_counts[mode],flush=True)

    def mutations(phase):
        for case in ('changed_queries','changed_labels'):
            q.copy_(torch.from_numpy(data['mutated_queries'].copy()))
            if case=='changed_labels':
                y.copy_(torch.from_numpy((data['train_labels'].astype(np.int32)+1)%10))
            for mode in MODES:
                replay(mode); torch.cuda.synchronize()
                checks[f'{phase}_{mode}_{case}'] = validate(mode,case)
        q.copy_(torch.from_numpy(data['test_images'].reshape(600,9).copy()))
        y.copy_(torch.from_numpy(data['train_labels'].astype(np.int32)))
        for mode in MODES:
            replay(mode); torch.cuda.synchronize()
            checks[f'{phase}_{mode}_restored'] = validate(mode,'canonical')
    mutations('before_timing')
    ptx = {name:kernel.asm['ptx'] for name,kernel in compiled.items()}
    assert all(not re.search(r'\b(?:fma|mad)\.[^;\n]*\.f(?:16|32|64)\b',text) for text in ptx.values())
    calibration = {}
    for mode in MODES:
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            replay(mode)
        b.record(); b.synchronize()
        ms = a.elapsed_time(b)/3
        calibration[mode] = {'cuda_ms':ms,'replays':max(1,int(np.ceil(10000/ms)))}
    print('Calibration',calibration,flush=True)

    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    def decode(value):
        return value.decode() if isinstance(value,bytes) else str(value)
    def optional(fn):
        try:
            return fn()
        except nv.NVMLError as error:
            return {'unavailable':str(error)}
    def stamp():
        a = time.perf_counter(); energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b = time.perf_counter()
        return {'energy_mj':energy,'time_s':(a+b)/2,'query_latency_s':b-a}
    def interval(a,b):
        seconds = b['time_s']-a['time_s']
        energy = (b['energy_mj']-a['energy_mj'])/1000
        return {'start':a,'end':b,'duration_s':seconds,'energy_j':energy,'average_power_w':energy/seconds}
    def idle():
        torch.cuda.synchronize(); time.sleep(3)
        a = stamp(); time.sleep(3)
        return interval(a,stamp())
    def telemetry():
        return {'temperature_c':nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                'power_w':nv.nvmlDeviceGetPowerUsage(handle)/1000,
                'graphics_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_GRAPHICS),
                'memory_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM),
                'throttle_reasons':optional(lambda:int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)))}
    properties = torch.cuda.get_device_properties(0)
    hardware = {'name':decode(nv.nvmlDeviceGetName(handle)),'uuid':decode(nv.nvmlDeviceGetUUID(handle)),
        'memory_bytes':properties.total_memory,'sm_count':properties.multi_processor_count,
        'compute_capability':f'{properties.major}.{properties.minor}',
        'power_limit_w':nv.nvmlDeviceGetPowerManagementLimit(handle)/1000,
        'mig_mode':optional(lambda:list(nv.nvmlDeviceGetMigMode(handle))),
        'processes':optional(lambda:[{'pid':p.pid,'used_memory_bytes':p.usedGpuMemory}
                                     for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)])}
    versions = {'python':platform.python_version(),'platform':platform.platform(),'torch':str(torch.__version__),
        'triton':triton.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
        'driver':decode(nv.nvmlSystemGetDriverVersion()),'nvml':decode(nv.nvmlSystemGetNVMLVersion()),'image':IMAGE_REF}
    trials = []
    for round_id,order in enumerate(ORDERS,1):
        for mode in order:
            before = idle(); tb = telemetry()
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            first = stamp(); wall_start = time.perf_counter(); a.record()
            repeats = calibration[mode]['replays']
            for _ in range(repeats):
                replay(mode)
            b.record(); b.synchronize(); wall = time.perf_counter()-wall_start
            active = interval(first,stamp()); ta = telemetry(); after = idle()
            elapsed = a.elapsed_time(b)/1000
            assert .8<elapsed/wall<1.2
            power = (before['average_power_w']+after['average_power_w'])/2
            adjusted = active['energy_j']-power*active['duration_s']
            row = {'round':round_id,'mode':mode,'replays':repeats,'idle_before':before,'idle_after':after,
                'active':active,'telemetry_before':tb,'telemetry_after':ta,'cuda_ms':elapsed*1000/repeats,
                'wall_ms':wall*1000/repeats,'adjusted_mj':adjusted*1000/repeats,'gross_mj':active['energy_j']*1000/repeats}
            for label,measure in (('before',before),('after',after)):
                row[label+'_only_mj'] = (active['energy_j']-measure['average_power_w']*active['duration_s'])*1000/repeats
            trials.append(row)
            checks[f'round{round_id}_{mode}'] = validate(mode,'canonical')
            print(json.dumps(row),flush=True)
    mutations('after_timing')
    nv.nvmlShutdown()
    for graph in graphs.values():
        call('cudaGraphExecDestroy',graph)
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for field in ('cuda_ms','wall_ms','adjusted_mj','gross_mj'):
            values = [r[field] for r in trials if r['mode']==mode]
            summary[mode][field] = {'median':statistics.median(values),'mean':statistics.mean(values),
                                  'sample_sd':statistics.stdev(values),'min':min(values),'max':max(values)}
    buf = io.BytesIO(); np.savez(buf,**arrays)
    result = {'hardware':hardware,'versions':versions,'provenance':provenance,'graphs':graph_counts,
        'calibration':calibration,'trials':trials,'summary':summary,'checks':checks,'orders':ORDERS,
        'kernel_ptx_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in ptx.items()},
        'grids':{'baseline':{'hidden':8,'output':3,'backward':8,'update':6,'infer_hidden':150,'infer_output':5},
                 'energy':{'hidden':1,'output':4,'backward':8,'update':12,'infer_hidden':20,'infer_output':20},
                 'no_slowdown':{'hidden':1,'output':4,'backward':8,'update':6,'infer_hidden':150,'infer_output':5}},
        'protocol':{'task':'Reset650 parameters; normalize600 train600 query inputs; targets;300epochs6000minibatches;600predictions6000scores',
                    'trials_per_variant':3,'target_active_seconds':10,'settle_seconds':3,'idle_seconds':3,
                    'energy':'NVML board counter minus paired idle power times actual counter interval, divided by replays',
                    'excluded':['transfers','allocation','JIT','graph capture','validation','startup','CPU energy']},
        'limitations':['GPU SIMD/register broadcasts adapt v4 panel reuse; no physical Dally distance mapping',
                       'Hidden uses one CTA to fill cross-kernel X cache and synchronize without an extra launch',
                       'GPU tile dimensions/padded lanes differ from serial v4 panels; all actual GPU work is measured',
                       'Inference parallelizes groups and retains600hidden rows as in original GPU harness, unlike serial v4 reuse',
                       'Both new variants use a real270word X cache; query caching is in registers',
                       'no_slowdown is a MODEL-derived name, not a hardware guarantee',
                       'Three cyclic-order rounds, one GPU; clock/thermal drift and idle subtraction uncertainty remain']}
    return result,ptx,buf.getvalue()


@app.local_entrypoint()
def main(output: str=''):
    import numpy as np
    import gpu_panels as g
    assert os.environ.get('MODAL_PROFILE')=='vargapowercouple','Explicit authorized profile required'
    destination = Path(output) if output else HERE/'gpu_measured'
    destination.mkdir(parents=True,exist_ok=True)
    assert not (destination/'results.json').exists(),'Preserve completed measurements'
    energy = json.loads((HERE/'selected.json').read_text())
    balanced = json.loads((HERE/'inference_refinement.json').read_text())['no_slowdown']['config']
    right = {'m':1,'n':4,'stage':'right_panel'}
    left = {'m':1,'n':10,'stage':'left'}
    assert energy=={'products':{'hidden':right,'output':left,'backward':right,'grad_w1':right,'grad_w2':left,
                               'infer_hidden':right,'infer_output':left},'inference_batch':30,'cache_x':True,'layout':'d2_first'}
    assert balanced=={'products':{'hidden':None,'output':left,'backward':right,'grad_w1':None,'grad_w2':None,
                                 'infer_hidden':None,'infer_output':None},'inference_batch':1,'cache_x':True,'layout':'d2_first'}
    source = g.generated_source()
    reference_path = HERE.parent/'mlp-best675-triton-20260911/prepared.npz'
    with np.load(reference_path,allow_pickle=False) as z:
        packed = io.BytesIO()
        arrays = {key:z[key] for key in ('train_images','train_labels','test_images','mutated_queries')}
        np.savez(packed,**arrays)
        expected = {case:{key:z[f'{case}_baseline_{key}'].tobytes() for key in ('params','scores','predictions')}
                    for case in ('canonical','changed_queries','changed_labels')}
        manifest = json.loads((g.ROOT/'mnist/doc/dataset_manifest.json').read_text())['tiers']['small']['arrays']
        for key in ('train_images','train_labels','test_images'):
            assert hashlib.sha256(arrays[key].tobytes()).hexdigest()==manifest[key]['sha256_c_order_little_endian']
    provenance = {'baseline_sha256':g.BASELINE_HASH,'generated_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256((HERE/'gpu_panels.py').read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'energy_config':energy,'no_slowdown_config':balanced,
        'input_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in arrays.items()},
        'theory_costs_sha256':hashlib.sha256((HERE/'costs.json').read_bytes()).hexdigest()}
    if (destination/'kernels.py').exists():
        previous_source = (destination/'kernels.py').read_bytes()
        archived = destination/('kernels_'+hashlib.sha256(previous_source).hexdigest()+'.py')
        if not archived.exists():
            archived.write_bytes(previous_source)
    (destination/'kernels.py').write_text(source)
    (destination/'runner.py').write_bytes(Path(__file__).read_bytes())
    result,ptx,arrays = benchmark.remote(source,packed.getvalue(),expected,provenance)
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'arrays.npz').write_bytes(arrays)
    for name,text in ptx.items():
        (destination/(name+'.ptx')).write_text(text)
    print(json.dumps(result['summary'],indent=2))
