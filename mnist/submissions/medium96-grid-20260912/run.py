"""Prepare, validate, fit/freeze, and separately evaluate the current MNIST draws."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
from mnist.code import data
import learner
app=learner.app
BASE={'width':512,'epochs':200,'learning_rate':0.1,'batch_size':25,'features':81,'n_train':10000,'n_test':10000}
SEEDS=list(range(2026091200,2026091211))
RAW=Path('/tmp/mnist-medium96-current/raw')

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ah(a):return data.array_hash(a)
def utc():return datetime.now(timezone.utc).isoformat()
def pack(a):return {'bytes':a.tobytes(),'shape':a.shape,'dtype':str(a.dtype)}

def source():
    paths={name:data.download_source(RAW,*data.SOURCES[name]) for name in ('train_images','train_labels')}
    pixels=data.read_idx(paths['train_images'],60000,images=True)
    labels=data.read_idx(paths['train_labels'],60000,images=False)
    return pixels,labels,{name:{'sha256':sha(path),'md5':data.file_hash(path,'md5'),'url':data.SOURCE_BASE+path.name} for name,path in paths.items()}

def draw(pixels,labels,seed,n=10000,q=10000):
    order=np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train,test=order[:n].astype('<i8'),order[n:n+q].astype('<i8')
    assert len(np.unique(np.r_[train,test]))==n+q
    def resized(ix):
        a=data.area_resize(pixels[ix].astype(np.float32)/np.float32(255),9)
        np.clip(a,0,1,out=a)
        return a[:,None,:,:]
    arrays={'train_images':resized(train),'train_labels':labels[train].astype('<i8'),'test_images':resized(test)}
    return arrays,train,test

@app.local_entrypoint()
def main(phase: str='validate',output: str=''):
    out=Path(output) if output else HERE
    if phase=='validate':
        if (out/'validation.json').exists():raise FileExistsError('Validation already recorded')
        plan={'created_at_utc':utc(),'base_configuration':BASE,'validation_dataset_seed':2026091299,
              'validation_learner_seed':991,'validation_split':'Independent PCG64 permutation; first8000 original-training rows fit, next2000 validate',
              'candidates_epochs':[200],'selection':'Fixed200epochs regardless of diagnostic validation result; no tuning or selection on these rows.',
              'test_dataset_seeds':SEEDS,'final_learner_seeds':list(range(101,112)),
              'learner_sha256':sha(HERE/'learner.py'),'no_test_results_inspected':True}
        write(out/'validation_plan.json',plan)
        pixels,labels,raw=source(); arrays,train,valid=draw(pixels,labels,2026091299,8000,2000)
        records=[]; selected=None
        for epochs in plan['candidates_epochs']:
            cfg={**BASE,'seed':991,'epochs':epochs,'n_train':8000,'n_test':2000}
            result,ptx=learner.benchmark.remote({k:pack(v) for k,v in arrays.items()},cfg,sha(HERE/'learner.py'),hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),False)
            prediction=np.array(result.pop('predictions'),dtype=np.uint8)
            correct=int(np.sum(prediction==labels[valid]))
            records.append({'epochs':epochs,'correct':correct,'total':2000,'result':result})
            print(f'TRAINING-ONLY validation epochs{epochs}: {correct}/2000',flush=True)
            selected=200
        write(out/'validation.json',{'completed_at_utc':utc(),'plan_sha256':sha(out/'validation_plan.json'),'raw_sources':raw,
              'train_indices_sha256':ah(train),'validation_indices_sha256':ah(valid),'records':records,'selected_epochs':selected})
        if selected is None:raise RuntimeError('Training-only validation did not reach target')
        config={**BASE,'epochs':selected}
        write(out/'config.json',config)
        write(out/'protocol.json',{'frozen_at_utc':utc(),'configuration':config,'dataset_seeds':SEEDS,'learner_seeds':list(range(101,112)),
          'dataset':'10000 train and10000 test sampled without replacement from original60000 training rows; independent PCG64(seed) permutation for each draw; first10000 train,next10000 test',
          'preprocessing':'uint8/255 float32; exact box-area28→9 using mnist/code/data.py; clipped[0,1]; input transform4*x-0.5 in learner',
          'training':'Fixed cyclic contiguous minibatches25; squared-error SGD; 81→512→10 ReLU; ascending separate FP32 multiplies/adds; seed-only PCG64 initialization; fresh complete training every draw',
          'selection':'Historical200epochs retained regardless of diagnostic validation; no search or checkpoint selection. Configuration and all seeds frozen before final-draw test evaluation; diagnostic validation is not accuracy evidence.',
          'target_accuracy_percent':96,'required_correct_of_110000':105600,'accuracy_statistic':'arithmetic mean across11draws; sample SD in pp; exact sum(correct)/110000',
          'learner_inputs':['train_images','train_labels','test_images'],'raw_sources':raw,
          'source_sha256':{name:sha(HERE/name) for name in ('learner.py','run.py')},'generator_sha256':sha(ROOT/'mnist/code/data.py')})
        return
    protocol=json.loads((out/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest,name
    assert protocol['generator_sha256']==sha(ROOT/'mnist/code/data.py')
    if phase=='fit':
        if (out/'prediction_manifest.json').exists():raise FileExistsError('Predictions already frozen')
        pixels,labels,raw=source(); assert raw==protocol['raw_sources']
        jobs=[]; manifests=[]
        for i,(ds,ls) in enumerate(zip(protocol['dataset_seeds'],protocol['learner_seeds'])):
            arrays,train,test=draw(pixels,labels,ds)
            cfg={**protocol['configuration'],'seed':ls}
            record={'draw':i,'dataset_seed':ds,'learner_seed':ls,'train_indices_sha256':ah(train),'test_indices_sha256':ah(test),
                    'train_test_disjoint':True,'input_sha256':{k:ah(v) for k,v in arrays.items()}}
            manifests.append(record)
            jobs.append(({k:pack(v) for k,v in arrays.items()},cfg,sha(HERE/'learner.py'),sha(out/'config.json'),True))
        write(out/'draw_manifest.json',{'created_at_utc':utc(),'protocol_sha256':sha(out/'protocol.json'),'draws':manifests,'test_labels_opened':False})
        entries=[]
        for i,(result,ptx) in enumerate(learner.benchmark.starmap(jobs,order_outputs=True)):
            pred=np.asarray(result.pop('predictions'),dtype=np.uint8)
            assert pred.shape==(10000,) and (pred<=9).all()
            assert result['input_sha256']==manifests[i]['input_sha256']
            path=out/'predictions'/f'draw-{i:02d}.npy';path.parent.mkdir(exist_ok=True);np.save(path,pred)
            write(out/'results'/f'draw-{i:02d}.json',result)
            entries.append({'draw':i,'dataset_seed':SEEDS[i],'file':str(path.relative_to(out)),'file_sha256':sha(path),'array_sha256_u8':ah(pred),
                            'result_file':f'results/draw-{i:02d}.json','result_sha256':sha(out/'results'/f'draw-{i:02d}.json')})
            print(f'Saved draw{i:02d}; no evaluation labels opened',flush=True)
        assert len(entries)==11
        write(out/'prediction_manifest.json',{'frozen_at_utc':utc(),'protocol_sha256':sha(out/'protocol.json'),'draw_manifest_sha256':sha(out/'draw_manifest.json'),'test_labels_opened':False,'predictions':entries})
    elif phase=='evaluate':
        manifest=json.loads((out/'prediction_manifest.json').read_text())
        assert manifest['protocol_sha256']==sha(out/'protocol.json')
        _,labels,raw=source(); assert raw==protocol['raw_sources']
        records=[]
        for e in manifest['predictions']:
            path=out/e['file']; assert sha(path)==e['file_sha256']
            result_path=out/e['result_file'];assert sha(result_path)==e['result_sha256']
            prediction=np.load(path,allow_pickle=False); assert ah(prediction)==e['array_sha256_u8']
            order=np.random.Generator(np.random.PCG64(e['dataset_seed'])).permutation(60000)
            truth=labels[order[10000:20000]]
            correct=int(np.sum(prediction==truth))
            result=json.loads(result_path.read_text()); m=result['summary']
            records.append({'draw':e['draw'],'dataset_seed':e['dataset_seed'],'learner_seed':result['model_config']['seed'],'correct':correct,'total':10000,
                            'accuracy_percent':correct/100,'test_labels_sha256_int64':ah(truth),'energy_a100_mj':m['idle_adjusted_j_per_invocation']['mean']*1000,
                            'time_a100_ms':m['cuda_event_us_per_invocation']['mean']/1000})
        count=sum(r['correct'] for r in records); percentages=np.array([r['accuracy_percent'] for r in records])
        accuracy={'evaluated_at_utc':utc(),'prediction_manifest_sha256':sha(out/'prediction_manifest.json'),'draws':records,'correct':count,'total':110000,
                  'accuracy_percent_mean':float(percentages.mean()),'accuracy_percent_sample_sd_pp':float(percentages.std(ddof=1)),
                  'target_accuracy_percent':96,'meets_target':count>=105600,
                  'energy_a100_mj_mean':float(np.mean([r['energy_a100_mj'] for r in records])),
                  'time_a100_ms_mean':float(np.mean([r['time_a100_ms'] for r in records]))}
        write(out/'accuracy.json',accuracy);print(json.dumps(accuracy,indent=2))
    else:raise ValueError(phase)
