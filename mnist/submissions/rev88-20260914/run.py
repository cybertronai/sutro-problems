"""Training-only selection, frozen eleven-draw qualification, and NVML energy."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time

import modal
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data_reference as data

IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image=(modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
       .env({'PYTHONPATH':'/workspace/rev88','CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
if modal.is_local():
    for name in ('learner.py','energy.py','data_reference.py'): image=image.add_local_file(HERE/name,'/workspace/rev88/'+name)
app=modal.App('sutro-rev88-20260914')
SEEDS=list(range(2026091400,2026091411))
BASE={'alpha':0.5,'batch_size':128,'momentum':0.9,'weight_decay':0.0,'seed':11}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def pack(a): return {'bytes':a.tobytes(),'shape':a.shape,'dtype':str(a.dtype)}

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,max_containers=4,
              min_containers=0,buffer_containers=0,scaledown_window=2,timeout=900,retries=0)
def execute(payload,config,measure=False):
    import gc
    import torch
    import learner
    torch.set_num_threads(4)
    arrays={k:np.frombuffer(v['bytes'],dtype=v['dtype']).reshape(v['shape']).copy() for k,v in payload.items()}
    assert set(arrays)=={'train_images','train_labels','test_images'}
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    task=learner.prepare_task(**arrays,config=config)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();task.run();b.record();b.synchronize()
    memory={'allocated_before_prepare_bytes':baseline,'prepare_and_invoke_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'allocated_after_invoke_bytes':torch.cuda.memory_allocated(),'reserved_after_invoke_bytes':torch.cuda.memory_reserved(),
            'scope':'Tensorallocator including graphpools/libraryworkspace; before diagnostic CUDAvalidation/serialization. Excludesdriver/context.'}
    try:
        pred,metadata=task.outputs()
    except RuntimeError as error:
        if not measure and str(error)=='Nonfinite scores or model state':
            return {'failed':True,'error':str(error),'config':config}
        raise
    out={'predictions':pred.astype(int).tolist(),'metadata':metadata,'config':config,
         'input_sha256':{k:ah(v) for k,v in arrays.items()},'cuda_ms':a.elapsed_time(b),
         'learner_sha256':sha('/workspace/rev88/learner.py'),'completed_at_utc':utc(),'memory':memory}
    if measure:
        from energy import measure as measure_task
        if hasattr(learner,'validate_cuda_replay'):
            out['cuda_validation']=learner.validate_cuda_replay(config)
        out['energy']=measure_task(task)
    return out

def source(raw_dir):
    paths={k:data.download_source(raw_dir,*data.SOURCES[k]) for k in ('train_images','train_labels')}
    x=data.read_idx(paths['train_images'],60000,images=True)
    y=data.read_idx(paths['train_labels'],60000,images=False).astype(np.int64)
    resized=data.area_resize(x.astype(np.float32)/np.float32(255),9)
    np.clip(resized,0,1,out=resized)
    return resized[:,None,:,:],y,{k:{'file':p.name,'sha256':sha(p),'md5':data.file_hash(p,'md5')} for k,p in paths.items()}

@app.local_entrypoint()
def main(phase:str='validate',raw_dir:str='mnist/data/raw'):
    x,y,raw=source(Path(raw_dir))
    if phase=='validate':
        assert not (HERE/'protocol.json').exists(),'Protocol already frozen'
        order=np.random.Generator(np.random.PCG64(SEEDS[0])).permutation(60000)
        ti,vi=order[:8000],order[8000:10000]
        arrays={'train_images':x[ti],'train_labels':y[ti],'test_images':x[vi]}
        configs=[{**BASE,'depth':d,'epochs':e,'learning_rate':lr} for d in (1,2) for e in (1,2,4,8) for lr in (.1,.3)]
        write(HERE/'validation-plan.json',{'created_at_utc':utc(),'configs':configs,'dataset_seed':SEEDS[0],
             'split':'First8000train, next2000validation of draw0 training portion; no qualification-test labels',
             'selection':'Among candidates with validationaccuracy>=89%, choose minimumepochs, then minimumdepth, then smallest accuracy surplus over89%; no test-result selection',
             'train_indices_sha256':ah(ti),'validation_indices_sha256':ah(vi),'learner_sha256':sha(HERE/'learner.py')})
        records=[]
        jobs=[({k:pack(v) for k,v in arrays.items()},c,False) for c in configs]
        for ci,result in enumerate(execute.starmap(jobs,order_outputs=True)):
            cfg=configs[ci]
            if result.get('failed'):
                records.append({'config':cfg,'correct':0,'total':2000,'failed':True,'result':result})
                print('VALIDATION FAILED',cfg,result['error'],flush=True)
                write(HERE/'validation.json',{'records':records,'complete':len(records)==len(configs)})
                continue
            pred=np.array(result.pop('predictions'),np.int64);correct=int(np.sum(pred==y[vi]))
            vp=HERE/'validation-predictions'/f'candidate-{ci:02}.npy';vp.parent.mkdir(exist_ok=True);np.save(vp,pred)
            record={'config':cfg,'correct':correct,'total':2000,'accuracy_percent':correct/20,'result':result,
                    'prediction_path':str(vp.relative_to(HERE)),'prediction_sha256':sha(vp)}
            records.append(record);print('VALIDATION',cfg,correct/20,flush=True)
            write(HERE/'validation.json',{'records':records,'complete':len(records)==len(configs)})
        eligible=[r for r in records if r['correct']>=1780]
        if not eligible: raise RuntimeError('No candidate reached training-only validation89%; no final test scored')
        selected=min(eligible,key=lambda r:(r['config']['epochs'],r['config']['depth'],r['correct']))
        cfg=selected['config'];write(HERE/'config.json',cfg)
        write(HERE/'protocol.json',{'frozen_at_utc':utc(),'configuration':cfg,'dataset_seeds':SEEDS,
            'learner_seed':cfg['seed'],'draw_count':11,'train_count':10000,'test_count':10000,'error_target_percent':12,
            'required_correct':96800,'total':110000,'raw_sources':raw,
            'dataset':'Independent PCG64(seed) permutation of official60000training rows; first10000train,next10000test, disjointwithin eachdraw; draws mayoverlap',
            'preprocessing':'float32uint8/255 then exact fractional box-area28to9 via vendored canonical data.py; clip[0,1]; learner4*x-.5',
            'selection':'training-only8000/2000 split of draw0training; protocol frozen before qualification-test scoring; selection rule in validation-plan.json',
            'source_sha256':{n:sha(HERE/n) for n in ('learner.py','energy.py','run.py','data_reference.py')},
            'validation_sha256':sha(HERE/'validation.json'),'gpu_workers_max':4,'new_state_every_run':True})
        print('SELECTED',cfg,flush=True);return
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest,name
    assert raw==protocol['raw_sources']
    if phase=='fit':
        assert not (HERE/'prediction-manifest.json').exists(),'Predictions already frozen'
        jobs=[]; draws=[]
        for i,seed in enumerate(SEEDS):
            order=np.random.Generator(np.random.PCG64(seed)).permutation(60000);ti,qi=order[:10000],order[10000:20000]
            assert len(np.unique(np.r_[ti,qi]))==20000
            arrays={'train_images':x[ti],'train_labels':y[ti],'test_images':x[qi]}
            draws.append({'draw':i,'dataset_seed':seed,'train_indices_sha256':ah(ti),'test_indices_sha256':ah(qi),
                          'train_test_disjoint':True,'input_sha256':{k:ah(v) for k,v in arrays.items()}})
            jobs.append(({k:pack(v) for k,v in arrays.items()},protocol['configuration'],i==0))
        write(HERE/'draw-manifest.json',{'created_at_utc':utc(),'draws':draws,'protocol_sha256':sha(HERE/'protocol.json')})
        entries=[]
        for i,result in enumerate(execute.starmap(jobs,order_outputs=True)):
            pred=np.array(result.pop('predictions'),np.int64);assert pred.shape==(10000,) and (pred>=0).all() and (pred<=9).all()
            assert result['input_sha256']==draws[i]['input_sha256']
            path=HERE/'predictions'/f'draw-{i:02}.npy';path.parent.mkdir(exist_ok=True);np.save(path,pred)
            rp=HERE/'results'/f'draw-{i:02}.json';write(rp,result)
            entries.append({'draw':i,'dataset_seed':SEEDS[i],'prediction_path':str(path.relative_to(HERE)),
                            'prediction_sha256':sha(path),'result_path':str(rp.relative_to(HERE)),'result_sha256':sha(rp)})
            print('FROZEN PREDICTIONS',i,flush=True)
        assert len(entries)==11
        write(HERE/'prediction-manifest.json',{'frozen_at_utc':utc(),'protocol_sha256':sha(HERE/'protocol.json'),
              'draw_manifest_sha256':sha(HERE/'draw-manifest.json'),'test_labels_scored':False,'entries':entries})
    elif phase=='evaluate':
        manifest=json.loads((HERE/'prediction-manifest.json').read_text());assert manifest['protocol_sha256']==sha(HERE/'protocol.json')
        rows=[]
        for entry in manifest['entries']:
            path=HERE/entry['prediction_path'];assert sha(path)==entry['prediction_sha256']
            assert sha(HERE/entry['result_path'])==entry['result_sha256']
            order=np.random.Generator(np.random.PCG64(entry['dataset_seed'])).permutation(60000);truth=y[order[10000:20000]]
            pred=np.load(path,allow_pickle=False);correct=int(np.sum(pred==truth))
            rows.append({'draw':entry['draw'],'dataset_seed':entry['dataset_seed'],'correct':correct,'total':10000,
                         'accuracy_percent':correct/100,'test_labels_sha256':ah(truth)})
        count=sum(r['correct'] for r in rows);acc=[r['accuracy_percent'] for r in rows]
        result={'evaluated_at_utc':utc(),'prediction_manifest_sha256':sha(HERE/'prediction-manifest.json'),
                'rows':rows,'correct':count,'total':110000,'accuracy_percent':count/1100,
                'sample_sd_pp':float(np.std(acc,ddof=1)),'error_percent':100-count/1100,'meets_12_percent_error_target':count>=96800}
        write(HERE/'accuracy.json',result);print(json.dumps(result,indent=2))
    else:raise ValueError(phase)
