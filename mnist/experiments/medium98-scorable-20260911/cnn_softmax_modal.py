"""Freeze and run a bounded training-only screen; there is no test phase."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import modal

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cnn_softmax_runner as runner
image=(modal.Image.from_registry(runner.IMAGE_REF).pip_install('numpy==2.2.6')
       .add_local_file(str(HERE/'cnn_softmax_runner.py'),remote_path='/root/cnn_softmax_runner.py'))
app=modal.App('sutro-medium98-softmax-cnn-screen')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1800,
 min_containers=0,max_containers=2,buffer_containers=0,scaledown_window=2,retries=0)
def run(payload,split,config,seed,provenance):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import importlib.util
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(Path('/root/cnn_softmax_runner.py'))==provenance['source_sha256']['cnn_softmax_runner.py']
    spec=importlib.util.spec_from_file_location('screen','/root/cnn_softmax_runner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(payload)=={'train_images','train_labels'}
    arrays={k:np.frombuffer(v['bytes'],dtype=v['dtype']).reshape(v['shape']).copy() for k,v in payload.items()}
    for name,value in arrays.items():assert module.array_hash(value)==provenance['input_sha256'][name]
    result,checkpoint,logits=module.train_run(arrays['train_images'],arrays['train_labels'],split,config,seed,provenance)
    return result,checkpoint,None if logits is None else logits.tobytes()

@app.local_entrypoint()
def main(phase: str='prepare',data: str='',output: str=''):
    import numpy as np
    root=Path(output) if output else HERE;root.mkdir(parents=True,exist_ok=True)
    canonical=json.loads((HERE.parents[1]/'doc/dataset_manifest.json').read_text())
    images,labels=runner.load_training(data,canonical)
    sources={name:sha(HERE/name) for name in ('cnn_softmax_runner.py','cnn_softmax_modal.py')}
    if phase=='prepare':
        assert not (root/'cnn_softmax_protocol.json').exists(),'Preserve existing protocol'
        fit,val=runner.split_indices(labels)
        split={'seed':runner.SPLIT_SEED,'fit_positions':fit.tolist(),'validation_positions':val.tolist(),
            'fit_label_counts':np.bincount(labels[fit],minlength=10).tolist(),
            'validation_label_counts':np.bincount(labels[val],minlength=10).tolist()}
        write(root/'cnn_softmax_split.json',split)
        protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Training-only preliminary native-FP32 screen of scorable ReLU CNNs',
            'source_sha256':sources,'input_sha256':{'train_images':runner.array_hash(images),'train_labels':runner.array_hash(labels)},
            'input_allowlist':['train_images','train_labels'],'split_seed':runner.SPLIT_SEED,'fit_examples':4800,'validation_examples':1200,
            'split_sha256':sha(root/'cnn_softmax_split.json'),'candidates':runner.candidates(),'initial_seed':11,
            'checkpoint_rule':'Highest validation correct, then lowest half-Brier probability loss, then earliest epoch',
            'candidate_rule':'Highest best-checkpoint validation correct, then lowest half-Brier loss, parameter count, id; exclude nonfinite runs',
            'replication_rule':'If initial best reaches1170/1200, replicate top2 at seeds22/33; rank mean best count then mean half-Brier loss, parameter count,id',
            'ensemble_rule':'Ordered FP32 sum of raw logits for seeds11,22,33; argmax without division; preliminary only, no final seed/test policy yet',
            'lr_schedule':'Initial rate epochs1–90,0.1× epochs91–127,0.01× epochs128–150; FP32 constants',
            'optimizer':'Explicit separate FP32 kernels: grad+=wd*weight; velocity*=momentum; velocity+=grad; weight-=lr*velocity; velocity starts zero',
            'loss_gradient':'Subtract row max; clamp[-16,0]; base=1+shift*FP32(1/1024); square10times; divide by sum_class; delta=(probability-onehot)*FP32(1/B). No log. Validation tie-break uses half-Brier probability loss.',
            'batch_order':'Seeded CUDA permutation each epoch; B128, final64 retained',
            'augmentation':'Training-only, raw images: independent ±8degree rotation,0.94–1.06 inverse scale,±0.35pixel translation, probability0.5; bilinear zero padding, align_cornersFalse; before fixed4x−0.5',
            'compiler_note':'Random initialization/augmentation/order/schedule input-independent; final ordered backend must materialize/charge schedules and be revalidated',
            'test_gate':'No test arrays accepted; no new11draw evaluation until final ordered learner is validated',
            'container_image':runner.IMAGE_REF,'test_arrays_accessed':False}
        write(root/'cnn_softmax_protocol.json',protocol);print('Frozen8configuration training-only protocol.',flush=True);return
    assert phase in ('search','replicate')
    protocol=json.loads((root/'cnn_softmax_protocol.json').read_text());split=json.loads((root/'cnn_softmax_split.json').read_text())
    assert protocol['source_sha256']==sources and protocol['split_sha256']==sha(root/'cnn_softmax_split.json')
    for name,value in [('train_images',images),('train_labels',labels)]:assert runner.array_hash(value)==protocol['input_sha256'][name]
    payload={name:{'shape':v.shape,'dtype':str(v.dtype),'bytes':v.tobytes()} for name,v in [('train_images',images),('train_labels',labels)]}
    provenance={'protocol_sha256':sha(root/'cnn_softmax_protocol.json'),'source_sha256':sources,'input_sha256':protocol['input_sha256'],'split_sha256':protocol['split_sha256']}
    for name in ('cnn_softmax_results','cnn_softmax_logits','cnn_softmax_checkpoints'):(root/name).mkdir(exist_ok=True)
    (root/'cnn_softmax_checkpoints/.gitignore').write_text('*\n!.gitignore\n')
    def rank(row):return (-row['best']['validation']['correct'],row['best']['validation']['loss'],row['parameter_count'],row['config']['id'])
    existing=[]
    for path in (root/'cnn_softmax_results').glob('*.json'):
        row=json.loads(path.read_text());assert row['provenance']==provenance
        assert row['config'] in protocol['candidates'] and row['seed'] in (11,22,33)
        if row.get('validation_logits_file'):
            values=np.load(root/row['validation_logits_file'],allow_pickle=False)
            assert runner.array_hash(values)==row['validation_logits_sha256']
        existing.append(row)
    if phase=='search':configs=protocol['candidates'];seeds=[11]
    else:
        first=[r for r in existing if r['seed']==11];assert len(first)==8
        finite=sorted([r for r in first if r['status']=='complete'],key=rank)
        assert finite and finite[0]['best']['validation']['correct']>=1170
        configs=[r['config'] for r in finite[:2]];seeds=[22,33]
        write(root/'cnn_softmax_replication_plan.json',{'configs':configs,'seeds':seeds,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    jobs=[(payload,split,c,seed,provenance) for c in configs for seed in seeds if not (root/'cnn_softmax_results'/f'{c["id"]}-s{seed}.json').exists()]
    for row,checkpoint,raw in run.starmap(jobs,order_outputs=False):
        assert row['provenance']==provenance
        if raw is not None:
            logits=np.frombuffer(raw,dtype=np.float32).reshape(1200,10)
            assert runner.array_hash(logits)==row['validation_logits_sha256']
            assert hashlib.sha256(checkpoint).hexdigest()==row['checkpoint_sha256']
            cfile='cnn_softmax_checkpoints/'+row['id']+'.pt';(root/cfile).write_bytes(checkpoint)
            lfile='cnn_softmax_logits/'+row['id']+'.npy';np.save(root/lfile,logits)
            row['checkpoint_file']=cfile;row['validation_logits_file']=lfile
        write(root/'cnn_softmax_results'/(row['id']+'.json'),row)
        print(json.dumps({'id':row['id'],'status':row['status'],'best':row['best']}),flush=True)
    all_rows=[json.loads(p.read_text()) for p in (root/'cnn_softmax_results').glob('*.json')]
    complete=[r for r in all_rows if r['status']=='complete']
    summary={'phase':phase,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'provenance':provenance,
        'ranking':[{'id':r['id'],'correct':r['best']['validation']['correct'],'loss':r['best']['validation']['loss'],'epoch':r['best']['epoch']} for r in sorted(complete,key=rank)],
        'failed_ids':[r['id'] for r in all_rows if r['status']!='complete'],'test_arrays_accessed':False}
    if phase=='replicate':
        summary['ensemble_validation']=[]
        for config in configs:
            rows=sorted([r for r in complete if r['config']==config],key=lambda r:r['seed'])
            if [r['seed'] for r in rows]!=[11,22,33]:continue
            logits=[np.load(root/r['validation_logits_file'],allow_pickle=False) for r in rows]
            ensemble=(logits[0]+logits[1])+logits[2]
            correct=int((ensemble.argmax(1)==labels[np.asarray(split['validation_positions'])]).sum())
            summary['ensemble_validation'].append({'config':config,'individual_correct':[r['best']['validation']['correct'] for r in rows],'ensemble_correct':correct,'total':1200})
    write(root/'cnn_softmax_summary.json',summary)
    print('STOP: preliminary validation screen only; no test arrays accessed.',flush=True)
