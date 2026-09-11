"""Run fixed learner members on isolated draws; freeze all predictions before evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import modal

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import learner

image=(modal.Image.from_registry(learner.IMAGE_REF).pip_install('numpy==2.2.6')
       .add_local_file(str(HERE/'learner.py'),remote_path='/root/learner.py'))
app=modal.App('sutro-medium-convnet-eleven-draw-accuracy')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=8192,timeout=1200,
              startup_timeout=300,min_containers=0,max_containers=2,
              buffer_containers=0,scaledown_window=2,retries=0)
def fit_member(payload,config,seed,provenance,keep_checkpoint):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import importlib.util
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('isolated_learner','/root/learner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(Path('/root/learner.py'))==provenance['source_sha256']['learner.py']
    assert set(payload)=={'train_images','train_labels','test_images'}
    arrays={name:np.frombuffer(value['bytes'],dtype=value['dtype']).reshape(value['shape']).copy()
            for name,value in payload.items()}
    for name,array in arrays.items():assert module.array_hash(array)==provenance['input_sha256'][name]
    result,checkpoint,logits=module.train_member(arrays,config,seed,provenance,keep_checkpoint)
    del arrays
    torch.cuda.empty_cache()
    return result,checkpoint,logits.tobytes()


@app.local_entrypoint()
def main(study: str=''):
    import numpy as np
    root=Path(study) if study else HERE
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['config']==learner.CONFIG and protocol['epochs']==learner.EPOCHS
    assert protocol['member_seeds']==learner.MEMBER_SEEDS==[101,102,103]
    assert protocol['dataset_seeds']==list(range(20261001,20261012))
    assert protocol['selected_inference']=='ensemble'
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest,name
    assert sha(HERE.parents[1]/'code'/'data.py')==protocol['generator_sha256']
    master=json.loads((root/'draw_manifest.json').read_text())
    assert master['protocol_sha256']==sha(root/'protocol.json')
    assert len(master['draws'])==11
    for name in ('results','predictions','logits','checkpoints'):
        path=root/name
        if path.exists() and any(path.iterdir()):raise FileExistsError(f'Refusing to resume/replace {path}')
        path.mkdir(exist_ok=True)
    if (root/'prediction_manifest.json').exists():raise FileExistsError('Predictions already frozen')
    provenance_common={'protocol_sha256':sha(root/'protocol.json'),'source_sha256':protocol['source_sha256'],
        'generator_sha256':protocol['generator_sha256'],'draw_manifest_sha256':sha(root/'draw_manifest.json'),
        'allowed_input_arrays':['train_images','train_labels','test_images'],
        'checkpoint_inputs':[],'state_policy':'Fresh model, optimizer, scheduler and RNG for every member/draw'}
    jobs=[]; seen=[]; manifests={}
    for entry in master['draws']:
        assert sha(root/entry['path'])==entry['sha256']
        draw=json.loads((root/entry['path']).read_text()); index=draw['draw_index']
        assert entry['draw_index']==index and entry['dataset_seed']==draw['dataset_seed']==protocol['dataset_seeds'][index]
        assert draw['protocol_sha256']==provenance_common['protocol_sha256']
        assert draw['generator_sha256']==protocol['generator_sha256']
        assert draw['train_test_disjoint'] is True and draw['test_labels_created'] is False
        fit=np.asarray(draw['train_indices'],dtype=np.int64);query=np.asarray(draw['test_indices'],dtype=np.int64)
        assert fit.shape==query.shape==(6000,) and len(np.unique(np.r_[fit,query]))==12000
        assert learner.array_hash(fit)==draw['train_indices_spec']['sha256']
        assert learner.array_hash(query)==draw['test_indices_spec']['sha256']
        assert sha(root/draw['allowed_archive'])==draw['allowed_archive_sha256']
        with np.load(root/draw['allowed_archive'],allow_pickle=False) as archive:
            assert set(archive.files)=={'train_images','train_labels','test_images'}
            arrays={name:archive[name] for name in archive.files}
        for name,array in arrays.items():
            spec=draw['arrays'][name]
            assert list(array.shape)==spec['shape'] and str(array.dtype)==spec['dtype']
            assert learner.array_hash(array)==spec['sha256']
        payload={name:{'shape':array.shape,'dtype':str(array.dtype),'bytes':np.ascontiguousarray(array).tobytes()}
                 for name,array in arrays.items()}
        provenance={**provenance_common,'draw_index':index,'dataset_seed':draw['dataset_seed'],
            'manifest_sha256':entry['sha256'],'input_sha256':{name:spec['sha256'] for name,spec in draw['arrays'].items()}}
        for seed in learner.MEMBER_SEEDS:jobs.append((payload,protocol['config'],seed,provenance,index==0))
        seen.append(index);manifests[index]=draw
    assert sorted(seen)==list(range(11))
    write(root/'accuracy_run_plan.json',{'provenance':provenance_common,'dataset_seeds':protocol['dataset_seeds'],
        'member_seeds':learner.MEMBER_SEEDS,'jobs':33,'max_containers':2,
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'per_draw_test_labels_created':False})
    pending={index:{} for index in range(11)}; records={}
    print('Running33fresh members across11isolated draws; no evaluator is present.',flush=True)
    for result,checkpoint,raw_logits in fit_member.starmap(jobs,order_outputs=False):
        index=result['provenance']['draw_index'];seed=result['seed'];draw=manifests[index]
        assert seed in learner.MEMBER_SEEDS and seed not in pending[index]
        assert result['provenance']['protocol_sha256']==provenance_common['protocol_sha256']
        assert result['provenance']['input_sha256']=={name:spec['sha256'] for name,spec in draw['arrays'].items()}
        assert result['config']==protocol['config'] and result['epochs']==learner.EPOCHS
        logits=np.frombuffer(raw_logits,dtype=np.float32).reshape(6000,10)
        assert np.isfinite(logits).all() and learner.array_hash(logits)==result['logits_sha256']
        assert learner.array_hash(logits.argmax(1).astype(np.int64))==result['predictions_sha256']
        if index==0:
            assert hashlib.sha256(checkpoint).hexdigest()==result['checkpoint_sha256']
            cpath=root/'checkpoints'/f'draw-{index:02d}-seed{seed}.pt';cpath.write_bytes(checkpoint)
            result['checkpoint_file']=str(cpath.relative_to(root))
        else:assert checkpoint==b''
        pending[index][seed]=(result,logits)
        print(f'Saved draw{index:02d},dataset{draw["dataset_seed"]},member{seed}; '
              f'{sum(len(v) for v in pending.values())}/33 complete',flush=True)
        if len(pending[index])==3:
            members=[pending[index][s][0] for s in learner.MEMBER_SEEDS]
            member_logits={f'seed{s}':pending[index][s][1] for s in learner.MEMBER_SEEDS}
            ensemble=np.mean(np.stack(list(member_logits.values())),axis=0,dtype=np.float64)
            predictions=ensemble.argmax(1).astype(np.int64)
            ppath=root/'predictions'/f'draw-{index:02d}.npy';np.save(ppath,predictions)
            lpath=root/'logits'/f'draw-{index:02d}.npz'
            np.savez_compressed(lpath,**member_logits,ensemble=ensemble,
                **{f'predictions_seed{s}':pending[index][s][1].argmax(1).astype(np.int64) for s in learner.MEMBER_SEEDS})
            record={'draw_index':index,'dataset_seed':draw['dataset_seed'],'protocol_sha256':provenance_common['protocol_sha256'],
                'manifest_sha256':members[0]['provenance']['manifest_sha256'],'members':members,
                'predictions_file':str(ppath.relative_to(root)),'predictions_file_sha256':sha(ppath),
                'predictions_sha256':learner.array_hash(predictions),
                'logits_file':str(lpath.relative_to(root)),'logits_file_sha256':sha(lpath),
                'ensemble_logits_sha256':learner.array_hash(ensemble),'test_labels_opened':False,
                'completed_at_utc':datetime.now(timezone.utc).isoformat()}
            rpath=root/'results'/f'draw-{index:02d}.json';write(rpath,record);records[index]=record
    assert len(records)==11
    entries=[]
    for index in range(11):
        record=records[index]
        # Verify every final array and file again before the global freeze.
        ppath=root/record['predictions_file'];lpath=root/record['logits_file']
        assert sha(ppath)==record['predictions_file_sha256'] and sha(lpath)==record['logits_file_sha256']
        pred=np.load(ppath,allow_pickle=False)
        assert pred.shape==(6000,) and pred.dtype==np.int64 and np.all((pred>=0)&(pred<=9))
        entries.append({'draw_index':index,'dataset_seed':record['dataset_seed'],
            'path':record['predictions_file'],'sha256':record['predictions_file_sha256'],
            'array_sha256':record['predictions_sha256'],'logits_path':record['logits_file'],
            'logits_sha256':record['logits_file_sha256'],'result_path':f'results/draw-{index:02d}.json',
            'result_sha256':sha(root/'results'/f'draw-{index:02d}.json')})
    write(root/'prediction_manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':provenance_common['protocol_sha256'],'draw_manifest_sha256':sha(root/'draw_manifest.json'),
        'selected_inference':'ensemble','member_seeds':learner.MEMBER_SEEDS,'predictions':entries,
        'test_labels_opened':False,'total_draws':11,'total_predictions':66000})
    print('ALL11ensemble and diagnostic member predictions frozen. STOP before separate local evaluation.',flush=True)


if __name__=='__main__':raise SystemExit('Run using modal run '+__file__)
