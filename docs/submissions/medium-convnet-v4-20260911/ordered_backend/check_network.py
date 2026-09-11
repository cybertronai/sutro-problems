"""Independent complete tiny-network CPU/GPU tests and graph-reset checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import modal
HERE=Path(__file__).resolve().parent
SOURCES={name:HERE/name for name in ['ops.py','cpu_ref.py','schedule.py','network.py','bn_ops.py','bn_ref.py','check_network.py']}
SOURCES['native_initialization_reference.py']=HERE/'network-results-bn-dropout/sources/native_initialization_reference.py'
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6')
for name,path in SOURCES.items():image=image.add_local_file(str(path),remote_path='/root/ordered_backend/'+name)
app=modal.App('sutro-ordered-complete-network-test')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,
              startup_timeout=300,max_containers=1,retries=0)
def check(hashes):
    import sys
    import re
    import numpy as np
    import torch
    import triton
    sys.path.insert(0,'/root/ordered_backend')
    import schedule
    import cpu_ref as cpu
    from network import Trainer
    for name,digest in hashes.items():assert sha(Path('/root/ordered_backend')/name)==digest
    torch.set_num_threads(4)
    rng=np.random.Generator(np.random.PCG64(20261203))
    images=rng.uniform(0,1,(7,1,4,4)).astype(np.float32)
    labels=np.array([0,1,2,3,4,5,6],np.int64)
    queries=rng.uniform(0,1,(131,1,4,4)).astype(np.float32)
    records=[];ptx={}
    def exact(a,b,name):
        assert a.dtype==b.dtype and a.shape==b.shape,(name,a.dtype,b.dtype,a.shape,b.shape)
        if a.dtype==np.float32:equal=a.view(np.uint32)==b.view(np.uint32)
        else:equal=a==b
        if not equal.all():
            positions=np.flatnonzero(~equal.reshape(-1))[:8]
            raise AssertionError({'name':name,'matched':int(equal.sum()),'total':a.size,
                'positions':positions.tolist(),'actual':a.reshape(-1)[positions].tolist(),'expected':b.reshape(-1)[positions].tolist()})
        return {'name':name,'words':a.size,'bitwise_match':True,'sha256':schedule.array_hash(a)}
    # Test the newly integrated feature combination; previous non-BN losses
    # already have complete retained checks in network-results-03.
    for loss in ('approx_softmax_gradient',):
        config={'width':2,'depth':2,'head_width':5,'image_size':4,'batch_size':3,'epochs':2,
            'learning_rate':.01,'momentum':.9,'weight_decay':.0001,'augmentation':'mild_affine','loss':loss,
            'batch_norm':True,'dropout':.2}
        trainer=Trainer(images,labels,config,101);trainer.prepare(queries)
        p={name:value.copy() for name,value in trainer.initial_arrays.items()}
        v={name:np.zeros_like(value) for name,value in p.items()}
        buffers=cpu.initial_buffers(p,config['depth'])
        masks=list(schedule.head_masks(101,7,2,5,.2))
        checks=[];schedule_checks=[]
        for epoch,(order,indices,coefficients,manifest) in enumerate(schedule.iter_epochs(101,7,2,4),1):
            manifest['head_mask_sha256']=schedule.array_hash(masks[epoch-1])
            assert manifest==trainer.schedule_manifests[epoch-1]
            transformed=schedule.apply_maps(images,indices,coefficients)
            # Direct scheduled augmentation is independently checked against
            # NumPy before the complete graph run resets its position.
            trainer.offset.fill_((epoch-1)*7)
            ws=trainer.workspaces[3]
            trainer.network.backend.augment(trainer.raw,trainer.labels,trainer.indices,trainer.coefficients,
                trainer.order,trainer.offset,ws['x'],ws['y'])
            checks.append(exact(ws['x'].cpu().numpy(),transformed[:3],f'{loss}/epoch{epoch}/augmentation'))
            checks.append(exact(ws['y'].cpu().numpy(),labels[order[:3]],f'{loss}/epoch{epoch}/label_gather'))
            trainer.network.backend.head_mask(trainer.head_masks,trainer.offset,ws['head_mask'])
            checks.append(exact(ws['head_mask'].cpu().numpy(),masks[epoch-1][:3],f'{loss}/epoch{epoch}/head_mask'))
            for first in range(0,7,3):
                p,v,gradient,scores=cpu.network_step(transformed[first:first+3],labels[order[first:first+3]],
                    p,v,config,schedule.learning_rate(config,epoch),buffers=buffers,head_mask=masks[epoch-1][first:first+3])
            schedule_checks.append(manifest)
        trainer.invoke();torch.cuda.synchronize();result=trainer.outputs()
        expected_scores=cpu.network_forward(queries*np.float32(4)-np.float32(.5),p,2,buffers=buffers)['scores']
        for name,value in p.items():
            checks.append(exact(result['parameters'][name],value,f'{loss}/parameters/{name}'))
            checks.append(exact(result['velocities'][name],v[name],f'{loss}/velocities/{name}'))
        for name,value in buffers.items():checks.append(exact(result['buffers'][name],value,f'{loss}/buffers/{name}'))
        assert set(trainer.network.state())==set(p)|set(buffers)
        checks.append(exact(result['scores'],expected_scores,f'{loss}/scores'))
        checks.append(exact(result['predictions'],expected_scores.argmax(1).astype(np.int64),f'{loss}/predictions'))
        # A second full invocation must reset every weight, velocity, and schedule
        # position, and reproduce the entire task after an intervening mutation.
        for value in trainer.network.params.values():value.fill_(17.)
        for value in trainer.network.velocity.values():value.fill_(-4.)
        for value in trainer.network.buffers.values():value.fill_(9.)
        trainer.offset.fill_(2)
        trainer.invoke();torch.cuda.synchronize();again=trainer.outputs()
        for name in p:checks.append(exact(again['parameters'][name],p[name],f'{loss}/reset/{name}'))
        for name in buffers:checks.append(exact(again['buffers'][name],buffers[name],f'{loss}/reset/{name}'))
        checks.append(exact(again['scores'],expected_scores,f'{loss}/reset/scores'))
        for name,kernel in trainer.network.backend.compiled.items():ptx[loss+'__'+name]=kernel.asm['ptx']
        records.append({'loss':loss,'config':config,'checks':checks,'schedule_manifests':schedule_checks,
            'initial_parameter_sha256':{name:schedule.array_hash(a) for name,a in trainer.initial_arrays.items()},
            'training_steps':6,'query_predictions':result['predictions'].tolist()})
        print(json.dumps({'loss':loss,'status':'passed','checked_arrays':len(checks)}),flush=True)
    import native_initialization_reference as native
    cfg={'width':32,'depth':3,'head_width':128,'image_size':9,'batch_norm':True,'dropout':.2,
         'bn_epsilon':1e-5,'bn_momentum':.1}
    torch.manual_seed(11)
    expected_initial=[p.detach().numpy().copy() for p in native.build_model(cfg).parameters()]
    actual_initial=schedule.initial_parameters(cfg,11)
    initialization_checks=[exact(a,b,'native_seed11_initialization/'+name)
        for (name,a),b in zip(actual_initial.items(),expected_initial)]
    assert len(actual_initial)==len(expected_initial)
    assert not any(re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',value) for value in ptx.values())
    ftz_lines={name:[line.strip() for line in value.splitlines() if '.ftz.' in line]
               for name,value in ptx.items() if '.ftz.' in value}
    assert not any(re.search(r'\b(?:mma|wmma|wgmma)\.',value) for value in ptx.values())
    return {'status':'needs_ptx_review' if ftz_lines else 'passed','source_sha256':hashes,'cases':records,
        'initialization_checks':initialization_checks,
        'hardware':{'gpu':torch.cuda.get_device_name(0),
          'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))},'software':{'torch':str(torch.__version__),
        'numpy':np.__version__,'triton':triton.__version__,'cuda':torch.version.cuda},
        'ptx_checks':{'no_fp32_fma':True,'no_ftz':not bool(ftz_lines),'ftz_lines':ftz_lines,'no_tensorcore_instructions':True,
            'sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in ptx.items()}},
        'scope':'Two complete tiny-network epochs with BN and dropout, final partial minibatch, all parameters/velocities/running buffers/scores/predictions, augmentation and reset; C32D3 seed11 initial literals match native CPU constructors exactly; no MNIST data or submission timing.',
        'completed_at_utc':datetime.now(timezone.utc).isoformat()},ptx

@app.local_entrypoint()
def main(output:str=''):
    destination=Path(output) if output else HERE/'network-results'
    destination.mkdir(parents=True,exist_ok=True)
    assert not any(destination.iterdir()),'Output must be empty'
    hashes={name:sha(path) for name,path in SOURCES.items()}
    (destination/'sources').mkdir()
    for name,path in SOURCES.items():(destination/'sources'/name).write_bytes(path.read_bytes())
    (destination/'source-freeze.json').write_text(json.dumps(hashes,indent=2)+'\n')
    result,ptx=check.remote(hashes)
    assert {name:sha(path) for name,path in SOURCES.items()}==hashes,'Source changed during test'
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,value in ptx.items():(destination/(name+'.ptx')).write_text(value)
    print(json.dumps({'status':result['status'],'hardware':result['hardware'],'cases':len(result['cases'])}))
