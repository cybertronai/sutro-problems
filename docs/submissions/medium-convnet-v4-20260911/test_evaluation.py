"""Synthetic freeze/tamper tests; no MNIST image or label file is read."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import evaluation as ev


def rejects(call, text):
    try:
        call()
    except ValueError as error:
        assert text in str(error), (text,str(error))
    else:
        raise AssertionError('Invalid artifact was accepted: '+text)


def main():
    for counts,passes in [([9600]*11,True),([9600]*10+[9599],False),([10000]*11,True)]:
        row=ev.summarize(counts)
        assert row['meets_target'] is passes
        if len(set(counts))==1:
            assert row['sample_standard_deviation_pp']==0
    row=ev.summarize([9600+i for i in range(-5,6)])
    assert row['mean_accuracy_percent']==96
    assert abs(row['sample_standard_deviation_pp']-np.sqrt(11)/100)<1e-14
    original_here=ev.HERE
    original_read_idx=ev.generator.read_idx
    def forbid_labels(*args,**kwargs):
        raise AssertionError('Freeze phase attempted to read a raw data file')
    with tempfile.TemporaryDirectory(prefix='sutro-evaluator-synthetic-') as directory:
        ev.HERE=Path(directory)
        ev.generator.read_idx=forbid_labels
        try:
            shutil.copy(original_here/'evaluation.py',ev.HERE/'evaluation.py')
            source={'evaluation.py':ev.sha(ev.HERE/'evaluation.py')}
            config={'member_seeds':[11,22,33],'ensemble_arithmetic':'ordered_fp32_sum','target_percent':96,
                    'epochs':2,'batch_size':128}
            protocol=dict(dataset_seeds=ev.DATASET_SEEDS,target_percent=96,n_train=10000,n_test=10000,
                source_pool=60000,config=config,source_sha256=source,
                generator_sha256=ev.sha(Path(ev.generator.__file__)))
            ev.write_new(ev.HERE/'protocol.json',protocol)
            metadata=[]
            for index,seed in enumerate(ev.DATASET_SEEDS):
                order,_=ev.generator.source_permutations(seed=seed)
                arrays={name:dict(shape=[10000] if name=='train_labels' else [10000,1,9,9],
                    dtype='int64' if name=='train_labels' else 'float32',sha256='synthetic-'+name)
                    for name in ev.INPUT_NAMES}
                draw=dict(draw_index=index,dataset_seed=seed,protocol_sha256=ev.sha(ev.HERE/'protocol.json'),
                    arrays=arrays,train_indices=order[:10000].tolist(),test_indices=order[10000:20000].tolist())
                path=f'draws/draw-{index:02d}.json';ev.write_new(ev.safe_path(path),draw)
                metadata.append(dict(draw_index=index,dataset_seed=seed,path=path,sha256=ev.sha(ev.safe_path(path))))
                logits=[np.zeros((10000,10),dtype=np.float32) for _ in range(3)]
                logits[0][:,0]=np.float32(1e20)
                logits[1][:,0]=np.float32(-1e20)
                logits[2][:,0]=np.float32(1)
                logits[0][:,1]=np.float32(.5)
                pred=np.zeros(10000,dtype=np.int64)
                path=ev.HERE/'predictions'/f'draw-{index:02d}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,predictions=pred,**{f'logits_seed{s}':a for s,a in zip([11,22,33],logits)})
                result=dict(draw_index=index,dataset_seed=seed,protocol_sha256=ev.sha(ev.HERE/'protocol.json'),
                    source_sha256=source,config=config,input_sha256={k:v['sha256'] for k,v in arrays.items()},
                    member_seeds=[11,22,33],test_labels_opened=False,fresh_state_per_member=True,
                    prediction_archive_sha256=ev.sha(path))
                result['members']=[dict(seed=s,config=config,epochs=2,minibatches_per_epoch=79,fresh_state=True,
                    logits_sha256=ev.generator.array_hash(a)) for s,a in zip([11,22,33],logits)]
                ev.write_new(ev.HERE/'results'/f'draw-{index:02d}.json',result)
            ev.write_new(ev.HERE/'draw_manifest.json',dict(protocol_sha256=ev.sha(ev.HERE/'protocol.json'),draws=metadata))
            protocol,_,draws=ev.verify_protocol()
            rows,predictions=ev.validate_outputs(protocol,draws)
            assert len(rows)==len(predictions)==11 and all(np.all(p==0) for p in predictions)
            ev.freeze(None)
            frozen=ev.read(ev.HERE/'prediction_manifest.json')
            assert frozen['total_predictions']==110000 and frozen['test_labels_opened'] is False
            path=ev.HERE/'predictions'/'draw-00.npz'
            contents=path.read_bytes()
            path.write_bytes(contents+b'tamper')
            rejects(lambda:ev.validate_outputs(protocol,draws),'Changed output archive')
            path.write_bytes(contents)
            # Preserve declared file hash but use the result of a reassociated sum.
            np.savez_compressed(path,predictions=np.ones(10000,dtype=np.int64),
                **{f'logits_seed{s}':a for s,a in zip([11,22,33],logits)})
            result_path=ev.HERE/'results'/'draw-00.json'
            result=ev.read(result_path);result['prediction_archive_sha256']=ev.sha(path)
            result_path.write_text(json.dumps(result))
            rejects(lambda:ev.validate_outputs(protocol,draws),'ordered FP32 ensemble')
            # A changed source fails before any result or data array is used.
            (ev.HERE/'evaluation.py').write_text('changed source')
            rejects(ev.verify_protocol,'Source changed after freeze')
        finally:
            ev.HERE=original_here
            ev.generator.read_idx=original_read_idx
    result=dict(status='passed',scope='Synthetic fixtures only; no MNIST raw files read',
        checks=['inclusive 4 percent error boundary','unrounded below-threshold failure','sample SD ddof1',
            'all eleven predictions required','raw IDX reads forbidden during freeze',
            'output byte mutation rejected','reassociated FP32 ensemble rejected','source mutation rejected'],
        evaluator_sha256=ev.sha(original_here/'evaluation.py'),test_sha256=ev.sha(Path(__file__)))
    (original_here/'evaluator-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
