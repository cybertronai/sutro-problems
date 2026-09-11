"""Configurable, exact-FP32 MNIST-medium A100 training/inference benchmark.

The GPU learner receives only train_images, train_labels, test_images, and
seed-dependent initial literals. Independent CPU parameters/scores/predictions
are passed separately as host-only validation oracles: they are never copied
to the GPU, used as initial weights, or consulted by any learner kernel.

Each measured task replays initialization once, one complete epoch E times,
and inference once. All input transformation, targets, parameter reset,
training, and inference are included. Allocation, transfer, compilation,
graph capture, and verification are excluded from GPU-resident timing.
"""
from pathlib import Path
import hashlib
import json
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install(
    'numpy==2.2.6', 'nvidia-ml-py==12.560.30')
app = modal.App('sutro-mnist-medium-affine')


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=16384,
              timeout=1800, startup_timeout=300, min_containers=0,
              max_containers=1, buffer_containers=0, scaledown_window=2, retries=0)
def benchmark(payload: dict, config: dict, oracle: dict, source_sha256: str,
              config_sha256: str):
    global tl
    import importlib.metadata
    import math
    import os
    import platform
    import re
    import statistics
    import time
    import numpy as np
    import pynvml as nv
    import torch
    import triton
    import triton.language as tl
    torch.set_num_threads(4)

    @triton.jit
    def normalize_kernel(X,Q,NX,NQ,N:tl.constexpr,QCOUNT:tl.constexpr,D:tl.constexpr,
                         BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        x=tl.load(X+off,off<N*D,other=0.)
        q=tl.load(Q+off,off<QCOUNT*D,other=0.)
        x=x*4.; x=x-0.5
        q=q*4.; q=q-0.5
        tl.store(NX+off,x,off<N*D)
        tl.store(NQ+off,q,off<QCOUNT*D)

    @triton.jit
    def initialize_kernel(INITIAL,P,Y,TARGET,N:tl.constexpr,PCOUNT:tl.constexpr,
                          BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        initial=tl.load(INITIAL+off,off<PCOUNT,other=0.)
        tl.store(P+off,initial,off<PCOUNT)
        label=tl.load(Y+off//10,off<N*10,other=0)
        tl.store(TARGET+off,(label==off%10).to(tl.float32),off<N*10)

    @triton.jit(do_not_specialize=['batch_start'])
    def hidden_kernel(X,P,H,batch_start,D:tl.constexpr,W:tl.constexpr,B:tl.constexpr,
                      BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=off//W; hidden=off%W; valid=off<B*W
        acc=tl.full((BLOCK,),0.,tl.float32)
        # A runtime loop keeps 81-feature compilation bounded without changing
        # the ascending multiply-then-add reduction order.
        for f in range(D):
            x=tl.load(X+(batch_start+row)*D+f,valid,other=0.)
            w=tl.load(P+f*W+hidden,valid,other=0.)
            product=x*w
            acc=acc+product
        acc=acc+tl.load(P+D*W+hidden,valid,other=0.)
        tl.store(H+off,tl.where(acc>0.,acc,0.),valid)

    @triton.jit(do_not_specialize=['batch_start'])
    def delta2_kernel(H,P,TARGET,D2,batch_start,D:tl.constexpr,W:tl.constexpr,
                      B:tl.constexpr,BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=off//10; c=off%10; valid=off<B*10
        acc=tl.full((BLOCK,),0.,tl.float32)
        for h in range(W):
            a=tl.load(H+row*W+h,valid,other=0.)
            w=tl.load(P+(D+1)*W+h*10+c,valid,other=0.)
            product=a*w
            acc=acc+product
        acc=acc+tl.load(P+(D+11)*W+c,valid,other=0.)
        target=tl.load(TARGET+(batch_start+row)*10+c,valid,other=0.)
        tl.store(D2+off,acc-target,valid)

    @triton.jit
    def delta1_kernel(H,P,D2,D1,D:tl.constexpr,W:tl.constexpr,B:tl.constexpr,
                      BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=off//W; h=off%W; valid=off<B*W
        acc=tl.full((BLOCK,),0.,tl.float32)
        for c in tl.static_range(10):
            d=tl.load(D2+row*10+c,valid,other=0.)
            w=tl.load(P+(D+1)*W+h*10+c,valid,other=0.)
            product=d*w
            acc=acc+product
        active=tl.load(H+off,valid,other=0.)>0.
        tl.store(D1+off,tl.where(active,acc,0.),valid)

    @triton.jit(do_not_specialize=['batch_start'])
    def update_kernel(X,H,D1,D2,P,batch_start,D:tl.constexpr,W:tl.constexpr,
                      B:tl.constexpr,STEP:tl.constexpr,BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        w1=off<D*W
        b1=(off>=D*W)&(off<(D+1)*W)
        w2=(off>=(D+1)*W)&(off<(D+11)*W)
        b2=(off>=(D+11)*W)&(off<(D+11)*W+10)
        hidden1=off%W
        hidden2=(off-(D+1)*W)//10
        class2=(off-(D+1)*W)%10
        grad=tl.full((BLOCK,),0.,tl.float32)
        for row in range(B):
            x=tl.load(X+(batch_start+row)*D+off//W,w1,other=0.)
            d1=tl.load(D1+row*W+hidden1,w1|b1,other=0.)
            h=tl.load(H+row*W+hidden2,w2,other=0.)
            d2=tl.load(D2+row*10+class2,w2,other=0.)
            d2bias=tl.load(D2+row*10+(off-(D+11)*W),b2,other=0.)
            term=tl.where(w1,x*d1,tl.where(b1,d1,tl.where(w2,h*d2,d2bias)))
            grad=grad+term
        change=STEP*grad
        old=tl.load(P+off,off<(D+11)*W+10,other=0.)
        tl.store(P+off,old-change,off<(D+11)*W+10)

    @triton.jit
    def inference_hidden_kernel(Q,P,H,QCOUNT:tl.constexpr,D:tl.constexpr,
                                W:tl.constexpr,BLOCK:tl.constexpr):
        off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=off//W; hidden=off%W; valid=off<QCOUNT*W
        acc=tl.full((BLOCK,),0.,tl.float32)
        for f in range(D):
            q=tl.load(Q+row*D+f,valid,other=0.)
            w=tl.load(P+f*W+hidden,valid,other=0.)
            product=q*w
            acc=acc+product
        acc=acc+tl.load(P+D*W+hidden,valid,other=0.)
        tl.store(H+off,tl.where(acc>0.,acc,0.),valid)

    @triton.jit
    def inference_output_kernel(H,P,SCORES,OUT,QCOUNT:tl.constexpr,D:tl.constexpr,
                                W:tl.constexpr,BLOCK:tl.constexpr):
        rows=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        valid=rows<QCOUNT
        best=tl.full((BLOCK,),float('-inf'),tl.float32)
        winner=tl.full((BLOCK,),0,tl.int32)
        for c in tl.static_range(10):
            acc=tl.full((BLOCK,),0.,tl.float32)
            for h in range(W):
                x=tl.load(H+rows*W+h,valid,other=0.)
                w=tl.load(P+(D+1)*W+h*10+c)
                product=x*w
                acc=acc+product
            acc=acc+tl.load(P+(D+11)*W+c)
            tl.store(SCORES+rows*10+c,acc,valid)
            change=acc>best
            winner=tl.where(change,c,winner)
            best=tl.where(change,acc,best)
        tl.store(OUT+rows,winner,valid)

    def unpack(value):
        return np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
    train=unpack(payload['train_images']).reshape(6000,81)
    labels=unpack(payload['train_labels']).astype(np.int32)
    queries=unpack(payload['test_images']).reshape(6000,81)
    D,W,B,E,SEED=81,int(config['width']),int(config['batch_size']),int(config['epochs']),int(config['seed'])
    assert B==30 and min(W,E)>0
    assert train.dtype==queries.dtype==np.float32
    assert np.isfinite(train).all() and np.isfinite(queries).all()
    rng=np.random.Generator(np.random.PCG64(SEED))
    initial_params=[rng.uniform(-1/math.sqrt(D),1/math.sqrt(D),(D,W)).astype(np.float32),
                    np.zeros(W,dtype=np.float32),
                    rng.uniform(-1/math.sqrt(W),1/math.sqrt(W),(W,10)).astype(np.float32),
                    np.zeros(10,dtype=np.float32)]
    packed_initial=np.concatenate([a.reshape(-1) for a in initial_params])
    PCOUNT=len(packed_initial)
    step=np.float32(float(config['learning_rate'])/B)
    kw={'num_warps':4,'enable_fp_fusion':False}

    def make_state(tx,ty,tq):
        # The only host arrays copied to GPU are learner inputs and seed-only
        # initial literals; the separate CPU oracle is never used here.
        n,qcount=len(tx),len(tq)
        assert n%B==0
        state={'n':n,'qcount':qcount,'compiled':{}}
        for name,array in [('x',tx),('y',ty),('q',tq),('initial',packed_initial)]:
            state[name]=torch.from_numpy(np.ascontiguousarray(array)).cuda()
        shapes={'nx':tx.shape,'nq':tq.shape,'params':(PCOUNT,),
                'target':(n,10),'hidden':(B,W),'d1':(B,W),'d2':(B,10),
                'inference_hidden':(qcount,W),'scores':(qcount,10)}
        for name,shape in shapes.items():
            state[name]=torch.empty(shape,dtype=torch.float32,device='cuda')
        state['output']=torch.empty((qcount,),dtype=torch.int32,device='cuda')
        return state

    def normalize(state):
        s=state
        s['compiled']['normalize']=normalize_kernel[(triton.cdiv(max(s['n'],s['qcount'])*D,128),)](
            s['x'],s['q'],s['nx'],s['nq'],s['n'],s['qcount'],D,128,**kw)

    def initialize(state):
        s=state
        normalize(s)
        s['compiled']['initialize']=initialize_kernel[(triton.cdiv(max(s['n']*10,PCOUNT),128),)](
            s['initial'],s['params'],s['y'],s['target'],s['n'],PCOUNT,128,**kw)

    def one_epoch(state):
        s=state
        for first in range(0,s['n'],B):
            s['compiled']['hidden']=hidden_kernel[(triton.cdiv(B*W,128),)](
                s['nx'],s['params'],s['hidden'],first,D,W,B,128,**kw)
            s['compiled']['delta2']=delta2_kernel[(triton.cdiv(B*10,128),)](
                s['hidden'],s['params'],s['target'],s['d2'],first,D,W,B,128,**kw)
            s['compiled']['delta1']=delta1_kernel[(triton.cdiv(B*W,128),)](
                s['hidden'],s['params'],s['d2'],s['d1'],D,W,B,128,**kw)
            s['compiled']['update']=update_kernel[(triton.cdiv(PCOUNT,128),)](
                s['nx'],s['hidden'],s['d1'],s['d2'],s['params'],first,D,W,B,float(step),128,**kw)

    def infer(state):
        s=state
        s['compiled']['inference_hidden']=inference_hidden_kernel[(triton.cdiv(s['qcount']*W,128),)](
            s['nq'],s['params'],s['inference_hidden'],s['qcount'],D,W,128,**kw)
        s['compiled']['inference_output']=inference_output_kernel[(triton.cdiv(s['qcount'],128),)](
            s['inference_hidden'],s['params'],s['scores'],s['output'],s['qcount'],D,W,128,**kw)

    def direct_task(state,epochs):
        initialize(state)
        for epoch in range(epochs):
            one_epoch(state)
            if epochs>10 and (epoch+1)%50==0:
                print(f'GPU direct training epoch {epoch+1}/{epochs}',flush=True)
        infer(state)

    def mm(a,b):
        out=np.zeros((a.shape[0],b.shape[1]),dtype=np.float32)
        for k in range(a.shape[1]): out=out+a[:,k,None]*b[None,k,:]
        return out
    def rowsum(a):
        out=np.zeros(a.shape[1],dtype=np.float32)
        for row in a: out=out+row
        return out
    def unpack_params(packed):
        return [packed[:D*W].reshape(D,W),packed[D*W:(D+1)*W],
                packed[(D+1)*W:(D+11)*W].reshape(W,10),packed[(D+11)*W:]]
    def cpu_predict(query_array,params):
        w1,b1,w2,b2=params
        nq=query_array*np.float32(4)-np.float32(.5)
        z=mm(nq,w1)+b1
        h=np.where(z>0,z,np.float32(0))
        score=mm(h,w2)+b2
        return np.argmax(score,axis=1).astype(np.int32),score
    def bounded_reference(tx,ty,tq,epochs=1):
        x=tx*np.float32(4)-np.float32(.5)
        target=(ty[:,None]==np.arange(10)).astype(np.float32)
        w1,b1,w2,b2=[a.copy() for a in initial_params]
        for _ in range(epochs):
            for first in range(0,len(tx),B):
                xb,t=x[first:first+B],target[first:first+B]
                z=mm(xb,w1)+b1; h=np.where(z>0,z,np.float32(0))
                d2=mm(h,w2)+b2-t
                d1=np.where(z>0,mm(d2,w2.T),np.float32(0))
                gw1,gb1=mm(xb.T,d1),rowsum(d1)
                gw2,gb2=mm(h.T,d2),rowsum(d2)
                w1=w1-step*gw1; b1=b1-step*gb1
                w2=w2-step*gw2; b2=b2-step*gb2
        params=[w1,b1,w2,b2]
        pred,score=cpu_predict(tq,params)
        return pred,score,np.concatenate([a.reshape(-1) for a in params])

    def validate(state,expected_prediction,expected_score,expected_parameter):
        actual=state['output'].cpu().numpy()
        actual_scores=state['scores'].cpu().numpy()
        actual_params=state['params'].cpu().numpy()
        for a,b,name in [(actual_params,expected_parameter,'parameters'),
                         (actual_scores,expected_score,'scores')]:
            eq=a.view(np.uint32)==b.view(np.uint32)
            if not np.all(eq):
                positions=np.flatnonzero(~eq.reshape(-1))[:10]
                raise AssertionError(f'{name}: {int(np.sum(eq))}/{a.size} exact; '
                    f'positions {positions.tolist()}, actual {a.reshape(-1)[positions].tolist()}, '
                    f'expected {b.reshape(-1)[positions].tolist()}')
        assert np.array_equal(actual,expected_prediction)
        return actual.copy(),{'prediction_matches':len(actual),'total_predictions':len(actual),
            'parameter_bitwise_matches':actual_params.size,'total_parameters':actual_params.size,
            'score_bitwise_matches':actual_scores.size,'total_scores':actual_scores.size,
            'parameter_sha256_float32_le':hashlib.sha256(actual_params.astype('<f4').tobytes()).hexdigest(),
            'scores_sha256_float32_le':hashlib.sha256(actual_scores.astype('<f4').tobytes()).hexdigest()}

    full=make_state(train,labels,queries)
    print(f'Compiling/running canonical D{D} H{W} E{E} seed{SEED} from scratch',flush=True)
    direct_task(full,E); torch.cuda.synchronize()
    ptx={name:kernel.asm['ptx'] for name,kernel in full['compiled'].items()}
    assert not any(re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',text) for text in ptx.values())
    assert not any('.ftz.' in text for text in ptx.values())

    # CPU-only validation oracle is first consulted after full GPU learning.
    expected=unpack(oracle['predictions']).astype(np.int32)
    expected_scores=unpack(oracle['scores'])
    expected_params=unpack(oracle['parameters'])
    predictions,canonical_validation=validate(full,expected,expected_scores,expected_params)
    print('Canonical parameters, all output scores, and predictions match CPU oracle exactly',flush=True)

    new_queries=np.random.default_rng(20260911).uniform(0,1,queries.shape).astype(np.float32)
    full['q'].copy_(torch.from_numpy(new_queries)); normalize(full); infer(full); torch.cuda.synchronize()
    ep,es=cpu_predict(new_queries,unpack_params(expected_params))
    new_predictions,mutation_validation=validate(full,ep,es,expected_params)
    mutation_validation['predictions_changed_from_canonical']=int(np.sum(new_predictions!=predictions))
    assert mutation_validation['predictions_changed_from_canonical']>0
    full['q'].copy_(torch.from_numpy(queries))

    changed_labels=((labels[:2*B]+1)%10).astype(np.int32)
    small=make_state(train[:2*B],changed_labels,new_queries[:17])
    direct_task(small,1); torch.cuda.synchronize()
    ep,es,ew=bounded_reference(train[:2*B],changed_labels,new_queries[:17])
    _,label_mutation_validation=validate(small,ep,es,ew)
    label_mutation_validation['scope']='Two complete minibatches (60 training examples), one epoch, 17 queries, same width/seed/rate; changed labels modulo10'
    unmutated,unmutated_scores,unmutated_params=bounded_reference(train[:2*B],labels[:2*B],new_queries[:17])
    label_mutation_validation['predictions_changed_from_original_labels']=int(np.sum(ep!=unmutated))
    label_mutation_validation['parameter_words_changed_from_original_labels']=int(np.sum(
        ew.view(np.uint32)!=unmutated_params.view(np.uint32)))
    label_mutation_validation['score_words_changed_from_original_labels']=int(np.sum(
        es.view(np.uint32)!=unmutated_scores.view(np.uint32)))
    # Two updates can change every score without changing the winning class.
    assert label_mutation_validation['parameter_words_changed_from_original_labels']>0
    assert label_mutation_validation['score_words_changed_from_original_labels']>0
    print('Query mutation and bounded two-minibatch training-label mutation passed',flush=True)

    capture_stream=torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        initialize(full); one_epoch(full); infer(full)
    torch.cuda.current_stream().wait_stream(capture_stream); torch.cuda.synchronize()
    graphs=[]
    for operation in (initialize,one_epoch,infer):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=capture_stream): operation(full)
        graphs.append(graph)
    torch.cuda.synchronize()

    def invocation():
        graphs[0].replay()
        for _ in range(E): graphs[1].replay()
        graphs[2].replay()
    invocation(); torch.cuda.synchronize()
    _,captured_validation=validate(full,expected,expected_scores,expected_params)
    start_event,stop_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start_event.record(); invocation(); stop_event.record(); stop_event.synchronize()
    calibration_task_ms=start_event.elapsed_time(stop_event)
    task_repeats=max(1,int(np.ceil(10000/calibration_task_ms)))
    print(f'Captured full task {calibration_task_ms:.2f} ms; {task_repeats} tasks per energy trial',flush=True)

    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)

    def decode(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def optional(call):
        try:
            return call()
        except nv.NVMLError as error:
            return {"unavailable": str(error)}

    def energy_stamp():
        before = time.perf_counter()
        energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
        after = time.perf_counter()
        return {"energy_mj": energy, "time_s": (before + after) / 2,
                "query_latency_s": after - before}

    def interval(first, last):
        duration = last["time_s"] - first["time_s"]
        energy = (last["energy_mj"] - first["energy_mj"]) / 1000
        return {"start": first, "end": last, "duration_s": duration,
                "energy_j": energy, "average_power_w": energy / duration}

    def idle_measurement(seconds=3.0, settle_seconds=3.0):
        torch.cuda.synchronize()
        # Allow clock/temperature and the integrated NVML counter to settle
        # before measuring idle power; this gap is outside the active interval.
        time.sleep(settle_seconds)
        first = energy_stamp()
        time.sleep(seconds)
        result = interval(first, energy_stamp())
        result['settle_seconds_before_measurement'] = settle_seconds
        return result

    def telemetry():
        return {
            "temperature_c": nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
            "power_w": nv.nvmlDeviceGetPowerUsage(handle) / 1000,
            "graphics_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS),
            "memory_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
            "throttle_reasons_mask": optional(lambda: int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle))),
        }

    props = torch.cuda.get_device_properties(0)
    hardware = {
        "gpu_name": decode(nv.nvmlDeviceGetName(handle)),
        "gpu_uuid": decode(nv.nvmlDeviceGetUUID(handle)),
        "gpu_pci_bus_id": decode(nv.nvmlDeviceGetPciInfo(handle).busId),
        "gpu_total_memory_bytes": int(props.total_memory),
        "multiprocessors": int(props.multi_processor_count),
        "compute_capability": f"{props.major}.{props.minor}",
        "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        "mig_mode": optional(lambda: list(nv.nvmlDeviceGetMigMode(handle))),
        "compute_processes": optional(lambda: [
            {"pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory}
            for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
    }
    versions = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "triton": triton.__version__,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "nvidia_driver": decode(nv.nvmlSystemGetDriverVersion()),
        "nvml": decode(nv.nvmlSystemGetNVMLVersion()),
        "nvidia_ml_py": importlib.metadata.version("nvidia-ml-py"),
        "modal": getattr(modal, "__version__", "runtime-injected; package metadata unavailable"),
        "container_image": IMAGE_REF,
    }
    trials=[]
    for trial_id in range(3):
        idle_before=idle_measurement()
        telemetry_before=telemetry()
        start=energy_stamp()
        wall_start=time.perf_counter()
        start_event.record()
        for _ in range(task_repeats): invocation()
        stop_event.record(); stop_event.synchronize()
        wall_end=time.perf_counter()
        finish=energy_stamp()
        telemetry_after=telemetry()
        active=interval(start,finish)
        idle_after=idle_measurement()
        idle_power=(idle_before['average_power_w']+idle_after['average_power_w'])/2
        adjusted=active['energy_j']-idle_power*active['duration_s']
        gpu_duration_s=start_event.elapsed_time(stop_event)/1000
        assert 0.8<gpu_duration_s/(wall_end-wall_start)<1.2,'CUDA/wall timing units disagree'
        trials.append({'trial':trial_id+1,'invocations':task_repeats,
            'cuda_graph_replays':task_repeats*(E+2),'cuda_graph_replays_per_invocation':E+2,
            'idle_before':idle_before,'idle_after':idle_after,'active':active,
            'telemetry_before':telemetry_before,'telemetry_after':telemetry_after,
            'paired_idle_power_w':idle_power,'idle_adjusted_energy_j':adjusted,
            'idle_adjusted_j_per_invocation':adjusted/task_repeats,
            'unadjusted_j_per_invocation':active['energy_j']/task_repeats,
            'before_only_adjusted_j_per_invocation':(
                active['energy_j']-idle_before['average_power_w']*active['duration_s'])/task_repeats,
            'after_only_adjusted_j_per_invocation':(
                active['energy_j']-idle_after['average_power_w']*active['duration_s'])/task_repeats,
            'cuda_event_duration_s':gpu_duration_s,
            'cuda_event_us_per_invocation':gpu_duration_s*1e6/task_repeats,
            'wall_duration_s':wall_end-wall_start,
            'wall_us_per_invocation':(wall_end-wall_start)*1e6/task_repeats})
        print(json.dumps(trials[-1]),flush=True)
    torch.cuda.synchronize()
    _,post_timing_validation=validate(full,expected,expected_scores,expected_params)
    final_parameter_bits=full['params'].cpu().numpy().view(np.uint32).astype(int).tolist()
    nv.nvmlShutdown()

    def summary(field):
        values=[trial[field] for trial in trials]
        return {'mean':statistics.mean(values),'median':statistics.median(values),
                'min':min(values),'max':max(values),'sample_stddev':statistics.stdev(values)}
    result={'schema_version':2,
        'algorithm':f'exact ordered-FP32 D{D} H{W} E{E} squared-error minibatch SGD, seed{SEED}',
        'source_sha256':source_sha256,'config_sha256':config_sha256,'model_config':config,
        'input_sha256':{name:hashlib.sha256(value['bytes']).hexdigest() for name,value in payload.items()},
        'hardware':hardware,'versions':versions,
        'protocol':{
            'dataset':'MNIST-medium competition-v2, 6000 train / 6000 test, 81 FP32 pixels',
            'training':f'Every task normalizes train/test pixels, constructs targets, resets {PCOUNT} seed-only parameter words, and runs all {E} epochs ({E*6000//B} minibatches)',
            'prediction':f'6000 queries through the freshly trained H{W} network; ordered FP32 reductions, fusion disabled, first-class ties',
            'timing':'CUDA-event and wall throughput of complete task schedules: initialization graph once, single-epoch graph E times, inference graph once',
            'cuda_graphs_stored':3,'cuda_graph_replays_per_task':E+2,
            'kernel_launches_per_task':4+4*E*6000//B,
            'captured_kernel_nodes':4+4*6000//B,
            'energy':'NVML cumulative counter delta minus average paired 3-second idle power times active interval; each idle sample follows 3-second settling gap',
            'trials':3,'target_active_seconds_per_trial':10,
            'idle_seconds_before_and_after_each_trial':3,'settling_seconds_before_each_idle_sample':3,
            'included':['all train/test normalization','all one-hot target construction','full initial parameter reset',
                        'all SGD minibatches','all 6000 predictions','all 60000 output scores','all task graph replay overhead'],
            'excluded':['host/device transfers','allocations','JIT','graph capture','verification','cold start'],
            'energy_scope':'NVML whole GPU board; host CPU energy excluded',
            'limitations':['GPU-resident repetition can reuse caches','Paired idle baseline may drift',
                           'NVML is integrated board telemetry, not per-kernel metering',
                           'Direct scalar-ordered kernels are not claimed optimal for A100']},
        'validation':{'canonical':canonical_validation,'captured_task':captured_validation,
            'after_all_timed_graph_replays':post_timing_validation,
            'new_input_random_queries':mutation_validation,'changed_training_labels_bounded':label_mutation_validation,
            'oracle_scope':'Independent CPU final parameters/scores/predictions are host-only comparison inputs; never transferred to GPU or used for learning',
            'cpu_oracle_artifact_sha256':oracle['artifact_sha256'],
            'ptx_has_fp32_fma':False,'ptx_has_ftz':False,
            'ptx_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in ptx.items()}},
        'calibration':{'complete_task_ms':calibration_task_ms,'tasks_per_trial':task_repeats},
        'initial_parameter_bits_u32':packed_initial.view(np.uint32).astype(int).tolist(),
        'final_parameter_bits_u32':final_parameter_bits,
        'trials':trials,'summary':{field:summary(field) for field in (
            'cuda_event_us_per_invocation','wall_us_per_invocation',
            'idle_adjusted_j_per_invocation','unadjusted_j_per_invocation')},
        'predictions':predictions.astype(int).tolist(),
        'prediction_sha256_int64_le':hashlib.sha256(predictions.astype('<i8').tobytes()).hexdigest()}
    return result,ptx


@app.local_entrypoint()
def main(data: str='mnist/data/medium.npz', config: str='', reference: str='', output: str=''):
    import numpy as np
    destination=Path(output) if output else HERE
    reference_dir=Path(reference) if reference else HERE
    config_path=Path(config) if config else HERE/'config.json'
    settings=json.loads(config_path.read_text())
    for key in ('width','epochs','learning_rate','seed','batch_size'):
        if key not in settings: raise ValueError(f'Missing frozen configuration field: {key}')
    destination.mkdir(parents=True,exist_ok=True)
    def pack(array):
        array=np.ascontiguousarray(array)
        return {'shape':array.shape,'dtype':str(array.dtype),'bytes':array.tobytes()}
    payload={}
    with np.load(data,allow_pickle=False) as dataset:
        for name in ('train_images','train_labels','test_images'):
            payload[name]=pack(dataset[name])
    assert payload['train_images']['shape']==(6000,1,9,9)
    assert payload['test_images']['shape']==(6000,1,9,9)
    assert payload['train_labels']['shape']==(6000,)
    canonical=json.loads((HERE.parent.parent/'doc'/'dataset_manifest.json').read_text())['tiers']['medium']['arrays']
    for name,value in payload.items():
        assert hashlib.sha256(value['bytes']).hexdigest()==canonical[name]['sha256_c_order_little_endian'],name
    # These independent CPU artifacts stay in host memory inside benchmark;
    # make_state and all GPU kernels accept only learner inputs and initial literals.
    with np.load(reference_dir/'parameters.npz',allow_pickle=False) as parameters:
        packed_parameters=np.concatenate([parameters[key].reshape(-1) for key in ('w1','b1','w2','b2')])
    oracle={'parameters':pack(packed_parameters),
            'scores':pack(np.load(reference_dir/'output-scores.npy',allow_pickle=False)),
            'predictions':pack(np.load(reference_dir/'predictions.npy',allow_pickle=False)),
            'artifact_sha256':{name:hashlib.sha256((reference_dir/name).read_bytes()).hexdigest()
                               for name in ('parameters.npz','output-scores.npy','predictions.npy')}}
    result,ptx=benchmark.remote(payload,settings,oracle,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hashlib.sha256(config_path.read_bytes()).hexdigest())
    (destination/'gpu_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,text in ptx.items(): (destination/('gpu_'+name+'.ptx')).write_text(text)
    np.save(destination/'gpu_predictions.npy',np.asarray(result['predictions'],dtype=np.int64))
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    raise SystemExit('Run with: uvx --with numpy==2.2.6 modal==1.5.5 run '+__file__)
