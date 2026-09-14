"""Run the frozen ordered learner on isolated draws, then benchmark a full task."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import sys
import modal

HERE=Path(__file__).resolve().parent
IMAGE_REF=('ghcr.io/ab-10/wikitext-bench@'
 'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
SOURCES=['modal_run.py','evaluation.py','measure_gpu.py','ensemble.py',
         'ordered_backend/network.py','ordered_backend/ops.py','ordered_backend/schedule.py','ordered_backend/cpu_ref.py',
         'ordered_backend/bn_ops.py','ordered_backend/bn_ref.py']
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
for filename in SOURCES:
    image=image.add_local_file(str(HERE/filename),remote_path='/root/submission/'+filename)
app=modal.App('sutro-mnist-medium-fully-scored')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,timeout=3600,
              startup_timeout=300,max_containers=4,scaledown_window=2,retries=0)
def execute(payload,draw,protocol,protocol_hash,mode,oracle):
    import gc
    import re
    import time
    import numpy as np
    import torch
    import triton
    sys.path.insert(0,'/root/submission/ordered_backend')
    sys.path.insert(0,'/root/submission')
    from network import Trainer
    from ensemble import Ensemble
    from measure_gpu import measure
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert set(payload)=={'train_images','train_labels','test_images'}
    assert set(protocol['source_sha256'])==set(SOURCES)
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((Path('/root/submission')/name).read_bytes()).hexdigest()==digest,name
    arrays={name:np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
            for name,value in payload.items()}
    for name,array in arrays.items():
        assert hashlib.sha256(array.tobytes()).hexdigest()==draw['arrays'][name]['sha256']
        assert list(array.shape)==draw['arrays'][name]['shape'] and str(array.dtype)==draw['arrays'][name]['dtype']
    assert arrays['train_images'].shape==arrays['test_images'].shape==(10000,1,9,9)
    assert arrays['train_labels'].shape==(10000,)
    config=protocol['config'];seeds=config['member_seeds'];members=[];ptx={};trainers=[]
    logits={};initials={};checkpoints={};started=time.perf_counter()
    for seed in seeds:
        member_config=dict(config)
        member_config['epochs']=config.get('member_epochs',{}).get(str(seed),config['epochs'])
        trainer=Trainer(arrays['train_images'],arrays['train_labels'],member_config,seed)
        trainer.prepare(arrays['test_images'])
        torch.cuda.synchronize();start=time.perf_counter();trainer.invoke();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        output=trainer.outputs()
        def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        logits[f'logits_seed{seed}']=output['scores']
        assert output['scores'].shape==(10000,10) and np.isfinite(output['scores']).all()
        for name,array in trainer.initial_arrays.items():initials[f'seed{seed}/{name}']=array
        if draw['draw_index']==0:
            for name,array in output['parameters'].items():checkpoints[f'seed{seed}/{name}']=array
            for name,array in output.get('buffers',{}).items():checkpoints[f'seed{seed}/{name}']=array
        members.append(dict(seed=seed,epochs=member_config['epochs'],config=member_config,
            minibatches_per_epoch=(10000+config['batch_size']-1)//config['batch_size'],
            parameter_count=sum(a.size for a in output['parameters'].values()),
            final_parameter_sha256={k:array_hash(a) for k,a in output['parameters'].items()},
            final_buffer_sha256={k:array_hash(a) for k,a in output.get('buffers',{}).items()},
            final_velocity_sha256={k:array_hash(a) for k,a in output['velocities'].items()},
            initial_parameter_sha256={k:array_hash(a) for k,a in trainer.initial_arrays.items()},
            schedule_manifests=trainer.schedule_manifests,logits_sha256=array_hash(output['scores']),
            diagnostic_invocation_seconds=elapsed,diagnostic_is_submission_measurement=False,
            fresh_state=True))
        if mode=='benchmark':
            # Hash-only learned-state references stay on the CPU and never enter
            # Trainer or any GPU tensor. The actual learner receives the allowlist above.
            expected=next(m for m in oracle['result']['members'] if m['seed']==seed)
            for field in ('config','epochs','final_parameter_sha256','final_buffer_sha256','final_velocity_sha256',
                          'initial_parameter_sha256','logits_sha256','schedule_manifests'):
                assert members[-1][field]==expected[field],f'Frozen accuracy reference mismatch: {seed}/{field}'
        for name,kernel in trainer.network.backend.compiled.items():ptx[name]=kernel.asm['ptx']
        print(f'Draw {draw["draw_index"]:02d}, seed {seed}: completed {member_config["epochs"]} epochs',flush=True)
        if mode=='benchmark':trainers.append(trainer)
        else:
            del trainer,output;gc.collect();torch.cuda.empty_cache()
    total=np.zeros((10000,10),dtype=np.float32)
    for seed in seeds:total=np.add(total,logits[f'logits_seed{seed}'],dtype=np.float32)
    predictions=total.argmax(1).astype(np.int64)
    measured=None
    if mode=='benchmark':
        assert oracle['result']['protocol_sha256']==protocol_hash
        with np.load(io.BytesIO(oracle['predictions']),allow_pickle=False) as original:
            assert np.array_equal(predictions,original['predictions'])
            for seed in seeds:
                assert np.array_equal(logits[f'logits_seed{seed}'].view(np.uint32),
                                      original[f'logits_seed{seed}'].view(np.uint32))
        ensemble=Ensemble(trainers)
        ensemble.invoke();torch.cuda.synchronize()
        assert np.array_equal(ensemble.total.cpu().numpy().view(np.uint32),total.view(np.uint32))
        assert np.array_equal(ensemble.predictions.cpu().numpy(),predictions)
        measured=measure(ensemble.invoke,ensemble.fingerprint)
        for name,kernel in ensemble.compiled.items():ptx[name]=kernel.asm['ptx']
    for name,text in ptx.items():
        assert not re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',text),name
        assert '.ftz.' not in text,name
        assert not re.search(r'\b(?:mma|wmma)\.',text),name
    result=dict(draw_index=draw['draw_index'],dataset_seed=draw['dataset_seed'],config=config,
        protocol_sha256=protocol_hash,source_sha256=protocol['source_sha256'],
        input_sha256={name:entry['sha256'] for name,entry in draw['arrays'].items()},
        member_seeds=seeds,members=members,test_labels_opened=False,fresh_state_per_member=True,
        hardware=dict(name=torch.cuda.get_device_name(0),uuid=str(torch.cuda.get_device_properties(0).uuid)),
        software=dict(torch=str(torch.__version__),triton=str(triton.__version__),numpy=np.__version__,
            cuda=torch.version.cuda,image=IMAGE_REF),completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_including_preparation_seconds=time.perf_counter()-started,
        ptx_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in ptx.items()},
        ptx_has_fp32_fma=False,ptx_has_ftz=False,ptx_has_tensorcore=False)
    def pack(**values):
        stream=io.BytesIO();np.savez_compressed(stream,**values);return stream.getvalue()
    return result,pack(predictions=predictions,**logits),pack(**initials),pack(**checkpoints),ptx,measured


@app.local_entrypoint()
def main(mode:str='accuracy'):
    import numpy as np
    sys.path.insert(0,str(HERE))
    import evaluation as ev
    assert mode in ('accuracy','benchmark')
    protocol,master,draws=ev.verify_protocol()
    assert set(protocol['source_sha256'])==set(SOURCES)
    assert protocol['config']['target_percent']==96
    if mode=='benchmark':
        # Benchmark the exact finalized predictor after the independent audit.
        accuracy=ev.read(HERE/'accuracy.json')
        assert accuracy['meets_target'] and accuracy['total_correct']>=105600 and accuracy['total_predictions']==110000
        assert accuracy['prediction_manifest_sha256']==ev.sha(HERE/'prediction_manifest.json')
        ev.validate_outputs(protocol,draws)
        draws=draws[:1]
    jobs=[]
    for draw in draws:
        output=(HERE/'results'/f'draw-{draw["draw_index"]:02d}.json') if mode=='accuracy' else HERE/'benchmark/results.json'
        if output.exists():
            assert mode=='accuracy','A completed benchmark must not be overwritten'
            continue
        archive=ev.safe_path(draw['archive']);assert ev.sha(archive)==draw['archive_sha256']
        with np.load(archive,allow_pickle=False) as data:
            assert set(data.files)=={'train_images','train_labels','test_images'}
            payload={name:dict(bytes=np.ascontiguousarray(data[name]).tobytes(),shape=data[name].shape,
                              dtype=str(data[name].dtype)) for name in data.files}
        oracle=None
        if mode=='benchmark':
            oracle=dict(result=ev.read(HERE/'results'/'draw-00.json'),
                        predictions=(HERE/'predictions'/'draw-00.npz').read_bytes())
        jobs.append((payload,draw,protocol,ev.sha(HERE/'protocol.json'),mode,oracle))
    if not jobs:
        print('No unfinished accuracy draws remain');return
    ev.write_new(HERE/('accuracy_run_plan.json' if mode=='accuracy' else 'benchmark/run_plan.json'),
        dict(created_at_utc=datetime.now(timezone.utc).isoformat(),mode=mode,
             protocol_sha256=ev.sha(HERE/'protocol.json'),draw_indices=[job[1]['draw_index'] for job in jobs],
             sources=protocol['source_sha256'],test_labels_supplied=False))
    for result,predictions,initials,checkpoint,ptx,measurement in execute.starmap(jobs,order_outputs=False):
        assert result['source_sha256']==protocol['source_sha256']
        for name,digest in protocol['source_sha256'].items():assert ev.sha(HERE/name)==digest
        index=result['draw_index'];folder=HERE if mode=='accuracy' else HERE/'benchmark'
        for sub in ('predictions','results','ptx','initial','checkpoints'):(folder/sub).mkdir(parents=True,exist_ok=True)
        path=folder/'predictions'/f'draw-{index:02d}.npz';path.write_bytes(predictions)
        result['prediction_archive_sha256']=ev.sha(path)
        ev.write_new(folder/'results'/f'draw-{index:02d}.json',result)
        if index==0:
            (folder/'initial'/'parameters.npz').write_bytes(initials)
            (folder/'checkpoints'/'draw-00.npz').write_bytes(checkpoint)
            for name,text in ptx.items():(folder/'ptx'/(name+'.ptx')).write_text(text)
        if measurement is not None:
            measurement['learner_result_sha256']=ev.sha(folder/'results'/f'draw-{index:02d}.json')
            ev.write_new(folder/'results.json',measurement)
        print(f'Saved draw {index:02d}: predictions frozen locally; no labels opened',flush=True)
