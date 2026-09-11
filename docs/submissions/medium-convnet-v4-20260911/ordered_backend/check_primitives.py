"""CPU/GPU bit tests and feasibility timings; no MNIST data or labels are used."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import modal
HERE=Path(__file__).resolve().parent
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
SOURCES=['ops.py','cpu_ref.py','check_primitives.py']
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6')
for name in SOURCES:image=image.add_local_file(str(HERE/name),remote_path='/root/ordered_backend/'+name)
app=modal.App('sutro-ordered-convolution-primitives')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1200,
              startup_timeout=300,max_containers=1,retries=0)
def check(source_hashes):
    import sys
    import re
    import time
    import numpy as np
    import torch
    import triton
    sys.path.insert(0,'/root/ordered_backend')
    import ops
    import cpu_ref as cpu
    for name,digest in source_hashes.items():assert sha(Path('/root/ordered_backend')/name)==digest
    torch.set_num_threads(4)
    backend=ops.Backend()
    rng=np.random.Generator(np.random.PCG64(20261201))
    def random(shape):return rng.uniform(-.25,.25,shape).astype(np.float32)
    def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).cuda()
    def empty(shape):return torch.empty(shape,dtype=torch.float32,device='cuda')
    records=[];ptx={}
    def retain(tag):
        for name,kernel in backend.compiled.items():ptx[tag+'__'+name]=kernel.asm['ptx']
    def exact(actual,expected,name):
        a=actual.cpu().numpy()
        assert a.dtype==expected.dtype==np.float32 and a.shape==expected.shape
        equal=a.view(np.uint32)==expected.view(np.uint32)
        if not equal.all():
            indices=np.flatnonzero(~equal.reshape(-1))[:8]
            raise AssertionError({'name':name,'matched':int(equal.sum()),'words':a.size,
                'indices':indices.tolist(),'actual':a.reshape(-1)[indices].tolist(),'expected':expected.reshape(-1)[indices].tolist()})
        return {'name':name,'words':a.size,'bitwise_match':True,
                'sha256_float32_le':hashlib.sha256(a.astype('<f4').tobytes()).hexdigest()}
    def timing(call):
        # Five warmed standalone primitive invocations. This is feasibility
        # evidence, not a complete-task throughput or energy measurement.
        for _ in range(2):call()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):call()
        end.record();end.synchronize()
        return float(start.elapsed_time(end))/5
    for label,n,ci,co,h,w in [('tiny_rectangular',2,2,3,4,5),('tiny_swapped',3,3,2,3,4),('representative_c32',128,32,32,9,9)]:
        x=random((n,ci,h,w));k=random((co,ci,3,3));dy=random((n,co,h,w))
        tx,tk,td=tensor(x),tensor(k),tensor(dy)
        xp=empty((n,ci,h+2,w+2));dp=empty((n,co,h+2,w+2))
        out=empty(dy.shape);dx=empty(x.shape);dk=empty(k.shape)
        backend.pad(tx,xp);backend.pad(td,dp)
        backend.forward(xp,tk,out);backend.dinput(dp,tk,dx);backend.dweight(xp,td,dk)
        torch.cuda.synchronize()
        start=time.perf_counter()
        expected_o=cpu.forward(x,k);expected_dx=cpu.dinput(dy,k);expected_dk=cpu.dweight(x,dy)
        checks=[exact(xp,cpu.pad(x),label+'/padding'),exact(out,expected_o,label+'/forward'),
                exact(dx,expected_dx,label+'/input_gradient'),exact(dk,expected_dk,label+'/weight_gradient')]
        reference_seconds=time.perf_counter()-start
        native_geometry_check=None
        if n<10:
            # Independent autograd convolution in FP64 checks gradient geometry,
            # including edges, rather than just copying the declared loops.
            xx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            kk=torch.tensor(k,dtype=torch.float64,requires_grad=True)
            yy=torch.nn.functional.conv2d(xx,kk,padding=1)
            yy.backward(torch.tensor(dy,dtype=torch.float64))
            assert np.allclose(expected_o,yy.detach().numpy(),rtol=2e-5,atol=1e-6)
            assert np.allclose(expected_dx,xx.grad.numpy(),rtol=2e-5,atol=1e-6)
            assert np.allclose(expected_dk,kk.grad.numpy(),rtol=2e-5,atol=1e-6)
            native_geometry_check='passed: independent FP64 torch conv2d/autograd, rtol2e-5 atol1e-6'
        ms={name:timing(call) for name,call in [('pad',lambda:backend.pad(tx,xp)),
            ('forward',lambda:backend.forward(xp,tk,out)),('input_gradient',lambda:backend.dinput(dp,tk,dx)),
            ('weight_gradient',lambda:backend.dweight(xp,td,dk))]}
        retain(label)
        record={'case':label,'shape_n_ci_co_h_w':[n,ci,co,h,w],'checks':checks,
                'independent_native_geometry_check':native_geometry_check,
                'cpu_reference_seconds':reference_seconds,'feasibility_ms_per_primitive':ms}
        records.append(record);print(json.dumps(record),flush=True)
    # Dense reductions, both transpose cases, row sums, branch edges and update.
    a=random((7,13));b=random((13,11));ta,tb=tensor(a),tensor(b);out=empty((7,11))
    backend.mm(ta,tb,out);checks=[exact(out,cpu.mm(a,b),'mm')]
    t_at=tensor(a.T.copy());backend.mm(t_at,tb,out,at=True)
    checks.append(exact(out,cpu.mm(a,b),'mm/transposed_left'))
    t_bt=tensor(b.T.copy());backend.mm(ta,t_bt,out,bt=True)
    checks.append(exact(out,cpu.mm(a,b),'mm/transposed_right'))
    row=empty((13,));backend.rows(ta,row);checks.append(exact(row,cpu.rows(a),'row_sum'))
    edge=np.array([-1.,-0.,0.,1.,1e-30,-1e-30],np.float32);edge_t=tensor(edge)
    dout=tensor(np.array([2.,3.,4.,5.,6.,7.],np.float32));r=empty(edge.shape)
    backend.relu(edge_t,r);checks.append(exact(r,np.where(edge>0,edge,np.float32(0)),'relu/zero_edges'))
    backend.drelu(edge_t,dout,r);checks.append(exact(r,np.where(edge>0,dout.cpu().numpy(),np.float32(0)),'relu_gradient/zero_edges'))
    weights=random((23,));grad=random((23,));velocity=random((23,))
    tw,tg,tv=tensor(weights),tensor(grad),tensor(velocity)
    for step in range(3):
        backend.momentum(tw,tg,tv,float(np.float32(.03)),float(np.float32(.9)),float(np.float32(.0001)))
        weights,velocity=cpu.momentum(weights,grad,velocity,.03)
        checks.append(exact(tw,weights,f'momentum/weights/step{step}'))
        checks.append(exact(tv,velocity,f'momentum/velocity/step{step}'))
    normalized=empty(a.shape);backend.normalize(ta,normalized)
    checks.append(exact(normalized,a*np.float32(4)-np.float32(.5),'normalization'))
    bias=random((13,));backend.bias(ta,tensor(bias),normalized)
    checks.append(exact(normalized,a+bias,'dense_bias'))
    scores=random((7,10));labels=np.arange(7,dtype=np.int64);delta=empty(scores.shape)
    inv=np.float32(1/7);backend.delta(tensor(scores),tensor(labels),delta,float(inv))
    expected_delta=(scores-(labels[:,None]==np.arange(10)).astype(np.float32))*inv
    checks.append(exact(delta,expected_delta,'scaled_output_delta'))
    records.append({'case':'dense_activation_optimizer','checks':checks});retain('dense_activation_optimizer')
    assert not any(re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',text) for text in ptx.values())
    assert not any('.ftz.' in text for text in ptx.values())
    assert not any(re.search(r'\b(?:mma|wmma|wgmma)\.',text) for text in ptx.values())
    return {'status':'passed','source_sha256':source_hashes,'cases':records,
        'hardware':{'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))},
        'software':{'numpy':np.__version__,'torch':str(torch.__version__),'triton':triton.__version__,'cuda':torch.version.cuda},
        'ptx_checks':{'no_fp32_fma':True,'no_ftz':True,'no_tensorcore_instructions':True,
                     'sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in ptx.items()}},
        'scope':'Primitive feasibility and correctness only; no MNIST data, no trained model, no submission benchmark or accuracy claim.',
        'completed_at_utc':datetime.now(timezone.utc).isoformat()},ptx

@app.local_entrypoint()
def main(output:str=''):
    destination=Path(output) if output else HERE/'primitive-results'
    destination.mkdir(parents=True,exist_ok=True)
    assert not any(destination.iterdir()),'Output must be empty; no overwritten results'
    hashes={name:sha(HERE/name) for name in SOURCES}
    (destination/'source-freeze.json').write_text(json.dumps(hashes,indent=2)+'\n')
    result,ptx=check.remote(hashes)
    assert {name:sha(HERE/name) for name in SOURCES}==hashes,'Sources changed during check'
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,text in ptx.items():(destination/(name+'.ptx')).write_text(text)
    print(json.dumps({'status':result['status'],'hardware':result['hardware'],
        'timings':[{k:c[k] for k in ('case','feasibility_ms_per_primitive') if k in c} for c in result['cases']]},indent=2))
