"""Actual A100 bit checks and bounded timing for explicit BN primitives."""
from pathlib import Path
import hashlib
import json
import modal

HERE=Path(__file__).resolve().parent
SOURCES=['bn_ref.py','bn_ops.py','ops.py','check_bn.py']
IMAGE_REF=('ghcr.io/ab-10/wikitext-bench@'
 'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6')
for name in SOURCES:image=image.add_local_file(str(HERE/name),remote_path='/root/bn/'+name)
app=modal.App('sutro-ordered-bn-primitive-check')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1200,max_containers=1,scaledown_window=2)
def run(hashes):
    import sys,time,re
    sys.path.insert(0,'/root/bn')
    import numpy as np
    import torch
    import bn_ref
    from bn_ops import BatchNorm
    from ops import Backend
    for name,digest in hashes.items():assert hashlib.sha256((Path('/root/bn')/name).read_bytes()).hexdigest()==digest
    b=Backend();bn=BatchNorm(b);rng=np.random.default_rng(7124);cases=[]
    def gpu(x):return torch.from_numpy(np.ascontiguousarray(x)).cuda()
    def exact(actual,expected,name):
        result=actual.cpu().numpy()
        assert result.shape==expected.shape
        bad=np.flatnonzero(result.view(np.uint32).reshape(-1)!=expected.view(np.uint32).reshape(-1))
        if len(bad):
            i=int(bad[0]);raise AssertionError((name,len(bad),float(result.flat[i]),float(expected.flat[i])))
    for shape,scale in [((3,3,3,3),1.),((2,2,2,2),0.),((2,3,3,3),1e-20),((128,32,9,9),1.)]:
        x=(rng.normal(size=shape)*scale).astype(np.float32);c=shape[1]
        gamma=rng.uniform(.5,1.5,c).astype(np.float32);beta=rng.uniform(-.1,.1,c).astype(np.float32)
        rm=rng.uniform(-.2,.2,c).astype(np.float32);rv=rng.uniform(.2,1.5,c).astype(np.float32)
        dy=rng.normal(size=shape).astype(np.float32)
        expected=bn_ref.train(x,gamma,beta,rm,rv)
        tx,tg,tb,trm,trv,tdy=map(gpu,(x,gamma,beta,rm,rv,dy))
        mean=torch.empty_like(trm);inverse=torch.empty_like(trm);norm=torch.empty_like(tx);out=torch.empty_like(tx)
        dx=torch.empty_like(tx);dg=torch.empty_like(tg);db=torch.empty_like(tb)
        bn.train(tx,tg,tb,trm,trv,mean,inverse,norm,out)
        for name,a,e in zip(['out','normalized','mean','inverse','runningmean','runningvar'],
                           [out,norm,mean,inverse,trm,trv],expected):exact(a,e,name)
        gradient=bn_ref.backward(dy,expected[1],gamma,expected[3])
        bn.backward(tdy,norm,tg,inverse,dx,dg,db)
        for name,a,e in zip(['dx','dgamma','dbeta'],[dx,dg,db],gradient):exact(a,e,name)
        inference=bn_ref.infer(x,gamma,beta,expected[4],expected[5])
        bn.infer(tx,tg,tb,trm,trv,inverse,norm,out)
        for name,a,e in zip(['inference','inference_norm','inference_inverse'],[out,norm,inverse],inference):exact(a,e,name)
        timings={}
        if shape[0]==128:
            for name,call in [('forward',lambda:bn.train(tx,tg,tb,trm,trv,mean,inverse,norm,out)),
                              ('backward',lambda:bn.backward(tdy,norm,tg,inverse,dx,dg,db))]:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                for _ in range(3):call()
                torch.cuda.synchronize();begin.record()
                for _ in range(10):call()
                end.record();end.synchronize();timings[name+'_ms']=begin.elapsed_time(end)/10
        cases.append(dict(shape=shape,input_scale=scale,bitwise_match=True,timings=timings))
        print(json.dumps(cases[-1]),flush=True)
    ptx={name:kernel.asm['ptx'] for name,kernel in b.compiled.items()}
    for name,text in ptx.items():
        assert not re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',text),name
        assert '.ftz.' not in text,name
        assert not re.search(r'\b(?:mma|wmma)\.',text),name
    return dict(status='passed',cases=cases,source_sha256=hashes,
        hardware=dict(gpu=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(0).uuid)),
        scope='Synthetic tensors only, not submission accuracy or whole-task timing',
        ptx_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in ptx.items()}),ptx


@app.local_entrypoint()
def main(output:str=''):
    destination=Path(output) if output else HERE/'bn-results'
    destination.mkdir(exist_ok=False)
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    (destination/'source-freeze.json').write_text(json.dumps(hashes,indent=2)+'\n')
    result,ptx=run.remote(hashes)
    assert hashes=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,text in ptx.items():(destination/(name+'.ptx')).write_text(text)
    print(json.dumps(result,indent=2))
