"""Frozen, training-validation-only aggregation and translation probes."""
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import modal
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cnn_dropout_runner as learner

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ah(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

image=(modal.Image.from_registry(learner.IMAGE_REF).pip_install('numpy==2.2.6')
       .add_local_file(str(HERE/'cnn_dropout_runner.py'),remote_path='/root/cnn_dropout_runner.py'))
app=modal.App('sutro-medium98-validation-inference-probe')

def probability(scores,temperature=1):
    values=scores/np.float32(temperature)
    maximum=values[:,0:1].copy()
    for digit in range(1,10):maximum=np.maximum(maximum,values[:,digit:digit+1])
    p=np.clip(values-maximum,np.float32(-16),np.float32(0))
    p=p*np.float32(1/1024)+np.float32(1)
    for _ in range(10):p=p*p
    denominator=np.zeros((len(values),1),np.float32)
    for digit in range(10):denominator=denominator+p[:,digit:digit+1]
    return p/denominator

def add(values):
    total=np.zeros_like(values[0])
    for value in values:total=total+value
    return total

def policy_predictions(logits,policy):
    if policy.startswith('probability_T'):
        scores=add([probability(value,int(policy[-1])) for value in logits])
        return scores.argmax(1).astype(np.int64),scores
    assert policy=='majority_tie_probability_T1'
    scores=add([probability(value) for value in logits])
    votes=np.zeros_like(scores,dtype=np.int32)
    for value in logits:votes[np.arange(len(value)),value.argmax(1)]+=1
    best=np.zeros(len(scores),np.int64)
    for digit in range(1,10):
        rows=np.arange(len(scores));old=votes[rows,best]
        change=(votes[:,digit]>old)|((votes[:,digit]==old)&(scores[:,digit]>scores[rows,best]))
        best=np.where(change,digit,best)
    return best,scores

def shift(raw,dx,dy):
    assert raw.dtype==np.float32 and raw.shape[1:]==(1,9,9)
    yy,xx=np.indices((9,9),dtype=np.float32)
    x=xx+np.float32(dx);y=yy+np.float32(dy)
    ix=np.floor(x).astype(np.int32);iy=np.floor(y).astype(np.int32)
    fx=x-ix.astype(np.float32);fy=y-iy.astype(np.float32)
    coefficients=[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
    result=np.zeros_like(raw)
    for (ox,oy),coefficient in zip([(0,0),(1,0),(0,1),(1,1)],coefficients):
        qx=ix+ox;qy=iy+oy;valid=(qx>=0)&(qx<9)&(qy>=0)&(qy<9)
        samples=raw[:,:,np.clip(qy,0,8),np.clip(qx,0,8)]
        samples=np.where(valid[None,None],samples,np.float32(0))
        result=result+samples*coefficient[None,None]
    return result

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1200,
              min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=2,retries=0)
def infer(views,checkpoint,config,seed,epoch,provenance):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    import importlib.util
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(Path('/root/cnn_dropout_runner.py'))==provenance['learner_source_sha256']
    assert hashlib.sha256(checkpoint).hexdigest()==provenance['checkpoint_sha256']
    spec=importlib.util.spec_from_file_location('frozen_learner','/root/cnn_dropout_runner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True)
    assert state['config']==config and state['seed']==seed and state['epoch']==epoch
    model=module.build_model(config).cuda();model.load_state_dict(state['state_dict']);model.eval()
    outputs=[]
    with torch.inference_mode():
        for raw in views:
            x=torch.from_numpy(raw.copy()).cuda()*4-.5
            outputs.append(np.concatenate([model(x[first:first+512]).cpu().numpy() for first in range(0,len(x),512)]))
    values=np.stack(outputs)
    return {'logits':values.tobytes(),'shape':values.shape,'sha256':ah(values),
            'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'numpy':np.__version__,
            'provenance':provenance}

@app.local_entrypoint()
def main(phase: str='prepare',data: str='',output: str=''):
    root=Path(output) if output else HERE
    canonical=json.loads((HERE.parents[1]/'doc/dataset_manifest.json').read_text())
    images,labels=learner.load_training(data,canonical)
    split=json.loads((root/'cnn_dropout_split.json').read_text())
    validation=np.asarray(split['validation_positions']);truth=labels[validation]
    if phase=='prepare':
        assert not (root/'cnn_probe_protocol.json').exists()
        families=[]
        for screen in ('softmax','bn','dropout'):
            summary=json.loads((root/f'cnn_{screen}_summary.json').read_text())
            assert summary['phase']=='replicate'
            for family in summary['ensemble_validation']:
                config=family['config'];rows=[]
                for seed in [11,22,33]:
                    path=root/f'cnn_{screen}_results/{config["id"]}-s{seed}.json'
                    row=json.loads(path.read_text())
                    assert row['config']==config and row['seed']==seed and row['status']=='complete'
                    rows.append({'result_file':str(path.relative_to(root)),'result_sha256':sha(path),
                                 'logits_file':row['validation_logits_file'],'logits_sha256':row['validation_logits_sha256'],
                                 'checkpoint_file':row['checkpoint_file'],'checkpoint_sha256':row['checkpoint_sha256'],
                                 'epoch':row['best']['epoch'],'seed':seed})
                families.append({'screen':screen,'config':config,'members':rows})
        protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                  'source_sha256':{'cnn_probe.py':sha(HERE/'cnn_probe.py'),'cnn_dropout_runner.py':sha(HERE/'cnn_dropout_runner.py')},
                  'families':families,'policies':['probability_T1','probability_T2','probability_T4','majority_tie_probability_T1'],
                  'temperature_rule':'Divide each member raw logits byFP32(T), then ordered-FP32 approximate softmax, then +0-start member sum in seed11/22/33 order.',
                  'majority_rule':'Most per-member argmax votes; ties largest probability_T1 sum; exact ties smallest digit.',
                  'tta_family':'dropout-c32-d3-lr0.03','tta_views':[[0,0],[.25,0],[-.25,0],[0,.25],[0,-.25]],
                  'tta_rule':'Explicit four-neighbor bilinear raw-image sampling with zero padding and quarter-pixel inverse translations. Sum probability_T1 in seed-major then view-major order; no view or seed selection.',
                  'validation_positions_sha256':ah(validation),'input_sha256':{'train_images':ah(images),'train_labels':ah(labels)},
                  'test_arrays_accessed':False,'accuracy_scope':'Preliminary native checkpoint validation only; final exact backend still requires new training-only validation before formal new draws.'}
        write(root/'cnn_probe_protocol.json',protocol);print('Frozen four aggregation policies and one five-view TTA ensemble.');return
    assert phase=='run'
    protocol=json.loads((root/'cnn_probe_protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest
    assert ah(validation)==protocol['validation_positions_sha256']
    for name,value in [('train_images',images),('train_labels',labels)]:assert ah(value)==protocol['input_sha256'][name]
    results={'protocol_sha256':sha(root/'cnn_probe_protocol.json'),'aggregations':[],'test_arrays_accessed':False}
    for family in protocol['families']:
        logits=[]
        for member in family['members']:
            assert sha(root/member['result_file'])==member['result_sha256']
            values=np.load(root/member['logits_file'],allow_pickle=False)
            assert ah(values)==member['logits_sha256'];logits.append(values)
        for policy in protocol['policies']:
            prediction,scores=policy_predictions(logits,policy)
            results['aggregations'].append({'family':family['config']['id'],'policy':policy,
                                           'correct':int((prediction==truth).sum()),'total':1200,
                                           'predictions_sha256':ah(prediction),'scores_sha256':ah(scores)})
    family=next(f for f in protocol['families'] if f['config']['id']==protocol['tta_family'])
    raw=np.ascontiguousarray(images[validation])
    views=[shift(raw,*offset) for offset in protocol['tta_views']]
    assert np.array_equal(views[0].view(np.uint32),raw.view(np.uint32))
    (root/'cnn_probe_outputs').mkdir(exist_ok=True)
    all_logits=[];members=[]
    for member in family['members']:
        checkpoint=(root/member['checkpoint_file']).read_bytes()
        assert hashlib.sha256(checkpoint).hexdigest()==member['checkpoint_sha256']
        provenance={'learner_source_sha256':protocol['source_sha256']['cnn_dropout_runner.py'],
                    'checkpoint_sha256':member['checkpoint_sha256'],'protocol_sha256':results['protocol_sha256']}
        output=infer.remote(views,checkpoint,family['config'],member['seed'],member['epoch'],provenance)
        assert output['provenance']==provenance
        values=np.frombuffer(output.pop('logits'),dtype=np.float32).reshape(output['shape']).copy()
        assert ah(values)==output['sha256'] and ah(values[0])==member['logits_sha256']
        path=root/f'cnn_probe_outputs/tta-seed{member["seed"]}.npz'
        np.savez_compressed(path,logits=values)
        output.update({'seed':member['seed'],'file':str(path.relative_to(root)),
                       'original_logits_exact':True,'individual_five_view_correct':int((add([probability(v) for v in values]).argmax(1)==truth).sum())})
        members.append(output);all_logits.extend(values)
    ensemble=add([probability(v) for v in all_logits]);predictions=ensemble.argmax(1).astype(np.int64)
    np.savez_compressed(root/'cnn_probe_outputs/tta-ensemble.npz',scores=ensemble,predictions=predictions)
    results['tta']={'family':family['config']['id'],'correct':int((predictions==truth).sum()),'total':1200,
                    'members':members,'scores_sha256':ah(ensemble),'predictions_sha256':ah(predictions)}
    results['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    write(root/'cnn_probe_results.json',results);print(json.dumps(results),flush=True)
