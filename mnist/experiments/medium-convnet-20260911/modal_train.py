"""Prepare, search, and replicate ConvNets using physically training-only input.

This command has no refit or test phase. After validation selection it stops so
the final policy can be frozen separately before any test data is supplied.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import sys
import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runner

IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).pip_install('numpy==2.2.6').add_local_file(
    str(HERE/'runner.py'),remote_path='/root/runner.py')
app = modal.App('sutro-mnist-medium-convnet-accuracy')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1200,
              startup_timeout=300,min_containers=0,max_containers=2,
              buffer_containers=0,scaledown_window=2,retries=0)
def train_remote(payload,split,config,seed,phase,provenance):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    import numpy as np
    import importlib.util
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('accuracy_runner','/root/runner.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert hashlib.sha256(Path('/root/runner.py').read_bytes()).hexdigest()==provenance['source_sha256']['runner.py']
    assert set(payload)=={'train_images','train_labels'}
    arrays={name:np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
            for name,value in payload.items()}
    for name,value in arrays.items():assert module.array_hash(value)==provenance['input_sha256'][name]
    result,checkpoint,logits=module.train_run(arrays['train_images'],arrays['train_labels'],
        split,config,seed,phase,provenance)
    result['container_image']=IMAGE_REF
    return result,checkpoint,logits


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_hashes():
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('runner.py','modal_train.py')}


def initial_rank(row):
    return (-row['best']['validation']['correct'],row['best']['validation']['loss'],
            row['parameter_count'],row['config']['id'])


@app.local_entrypoint()
def main(phase: str='prepare',data: str='mnist/data/medium-train-only.npz',output: str=''):
    import numpy as np
    destination=Path(output) if output else HERE
    destination.mkdir(parents=True,exist_ok=True)
    if phase not in ('prepare','search','replicate'):
        raise ValueError('Only prepare/search/replicate phases exist; no test phase is permitted')
    canonical=json.loads((HERE.parent.parent/'doc'/'dataset_manifest.json').read_text())
    images,labels=runner.load_training(data,canonical)
    if phase=='prepare':
        if (destination/'protocol.json').exists():raise RuntimeError('Protocol already frozen; do not overwrite it')
        fit,val=runner.split_indices(labels)
        split={'seed':runner.SPLIT_SEED,'fit_positions':fit.tolist(),'validation_positions':val.tolist(),
            'fit_label_counts':np.bincount(labels[fit],minlength=10).tolist(),
            'validation_label_counts':np.bincount(labels[val],minlength=10).tolist(),
            'input_sha256':{'train_images':runner.array_hash(images),'train_labels':runner.array_hash(labels)},
            'created_at_utc':datetime.now(timezone.utc).isoformat()}
        write_json(destination/'validation_split.json',split)
        protocol=runner.protocol_document(images,labels,source_hashes())
        protocol['split_file_sha256']=hashlib.sha256((destination/'validation_split.json').read_bytes()).hexdigest()
        protocol['container_image']=IMAGE_REF
        write_json(destination/'protocol.json',protocol)
        print('Prepared immutable ten-candidate protocol and stratified 4800/1200 training-only split.',flush=True)
        return
    protocol=json.loads((destination/'protocol.json').read_text())
    split=json.loads((destination/'validation_split.json').read_text())
    assert protocol['source_sha256']==source_hashes(),'Source changed after protocol freeze'
    assert protocol['split_file_sha256']==hashlib.sha256((destination/'validation_split.json').read_bytes()).hexdigest()
    for name,array in [('train_images',images),('train_labels',labels)]:
        assert protocol['input_sha256'][name]==runner.array_hash(array)
    payload={name:{'shape':array.shape,'dtype':str(array.dtype),'bytes':np.ascontiguousarray(array).tobytes()}
             for name,array in [('train_images',images),('train_labels',labels)]}
    provenance={'protocol_sha256':hashlib.sha256((destination/'protocol.json').read_bytes()).hexdigest(),
        'source_sha256':source_hashes(),'input_sha256':protocol['input_sha256'],
        'split_file_sha256':protocol['split_file_sha256'],'allowed_input_arrays':list(payload)}
    result_dir=destination/'results'; result_dir.mkdir(exist_ok=True)
    checkpoints=destination/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    logits_dir=destination/'validation_logits'; logits_dir.mkdir(exist_ok=True)
    if phase=='search':
        configs=protocol['candidates']; seeds=[protocol['initial_seed']]
    else:
        search=[json.loads(path.read_text()) for path in result_dir.glob('search-*.json')]
        assert len(search)==len(protocol['candidates']),'Complete all initial candidates before replication'
        configs=[r['config'] for r in sorted(search,key=initial_rank)[:protocol['replication']['top_initial_candidates']]]
        seeds=protocol['replication']['additional_seeds']
        write_json(destination/'replication_plan.json',{'selected_before_replication':configs,'seeds':seeds,
            'initial_ranking':[r['config']['id'] for r in sorted(search,key=initial_rank)],
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'provenance':provenance})
    jobs=[]
    for config in configs:
        for seed in seeds:
            name=f'{phase}-{config["id"]}-s{seed}'
            if (result_dir/(name+'.json')).exists():continue
            jobs.append((payload,split,config,seed,phase,provenance))
    print(f'{phase}: dispatching {len(jobs)} independent runs, at most two A100 containers',flush=True)
    for result,checkpoint,logit_bytes in train_remote.starmap(jobs,order_outputs=False):
        name=result['id']
        assert result['provenance']==provenance
        assert hashlib.sha256(checkpoint).hexdigest()==result['best_checkpoint_sha256']
        logits=np.frombuffer(logit_bytes,dtype=np.float32).reshape(1200,10)
        assert runner.array_hash(logits)==result['best_validation_logits_sha256']
        (checkpoints/(name+'.pt')).write_bytes(checkpoint)
        np.save(logits_dir/(name+'.npy'),logits)
        result['checkpoint_file']='checkpoints/'+name+'.pt'
        result['validation_logits_file']='validation_logits/'+name+'.npy'
        write_json(result_dir/(name+'.json'),result)
        print(json.dumps({'run':name,'best_epoch':result['best']['epoch'],
            'validation_correct':result['best']['validation']['correct'],
            'validation_total':1200,'validation_loss':result['best']['validation']['loss'],
            'elapsed_seconds':result['training_elapsed_seconds']}),flush=True)
    results=[json.loads(path.read_text()) for path in result_dir.glob('*.json')]
    search=[r for r in results if r['phase']=='search']
    write_json(destination/'search_summary.json',{'provenance':provenance,
        'initial_ranking':[{'id':r['config']['id'],'best_epoch':r['best']['epoch'],
            'validation_correct':r['best']['validation']['correct'],'validation_loss':r['best']['validation']['loss'],
            'parameter_count':r['parameter_count']} for r in sorted(search,key=initial_rank)],
        'complete_initial_runs':len(search),'complete_replication_runs':sum(r['phase']=='replicate' for r in results),
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    if phase=='replicate':
        aggregation=[]
        for config in configs:
            reps=[r for r in results if r['config']['id']==config['id']]
            assert sorted(r['seed'] for r in reps)==[11,22,33]
            logits=[np.load(destination/r['validation_logits_file'],allow_pickle=False) for r in sorted(reps,key=lambda r:r['seed'])]
            ensemble=np.mean(np.stack(logits),axis=0,dtype=np.float64)
            truth=labels[np.asarray(split['validation_positions'])]
            aggregation.append({'config':config,'parameter_count':reps[0]['parameter_count'],
                'seeds':[11,22,33],'mean_best_validation_correct':statistics.mean(r['best']['validation']['correct'] for r in reps),
                'mean_best_validation_loss':statistics.mean(r['best']['validation']['loss'] for r in reps),
                'best_epochs_by_seed':{str(r['seed']):r['best']['epoch'] for r in reps},
                'refit_epochs':int(statistics.median(r['best']['epoch'] for r in reps)),
                'individual_correct_by_seed':{str(r['seed']):r['best']['validation']['correct'] for r in reps},
                'validation_logits_average_ensemble_correct':int((ensemble.argmax(1)==truth).sum()),
                'ensemble_accumulation':'float64 arithmetic mean of three float32 raw-logit arrays',
                'validation_total':len(truth)})
        aggregation.sort(key=lambda r:(-r['mean_best_validation_correct'],r['mean_best_validation_loss'],r['parameter_count'],r['config']['id']))
        preferred=aggregation[0]
        preferred['selected_inference']='ensemble' if preferred['validation_logits_average_ensemble_correct']>preferred['individual_correct_by_seed']['11'] else 'single'
        preferred['ensemble_selection_rule']='Strictly higher validation correct count than fixed primary seed11 checkpoint; ties favor single'
        write_json(destination/'validation_selection.json',{'ranked_candidates':aggregation,
            'preferred_candidate':preferred,'provenance':provenance,
            'test_arrays_accessed':False,'phase_gate':'STOP: final configuration, epochs, seeds, and ensemble policy must be frozen before separate refit/test',
            'created_at_utc':datetime.now(timezone.utc).isoformat()})
    print('Accuracy-only phase finished. No test array was opened; stop before refit/test.',flush=True)


if __name__=='__main__':
    raise SystemExit('Run with: uvx --with numpy==2.2.6 modal==1.5.5 run '+__file__)
