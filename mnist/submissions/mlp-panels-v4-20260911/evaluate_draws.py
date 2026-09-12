"""Fresh submission evaluation: predeclare, prepare/predict, then score.

Dataset preparation is trusted; the learner only sees train_images,
train_labels and test_images. All predictions are frozen before label scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import learner
import panels as p

sys.path.insert(0,str(p.ROOT))
from mnist.code import data as dataset

HERE = Path(__file__).resolve().parent
SEEDS = list(range(2026091301,2026091312))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def predeclare(output):
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'plan.json').exists(),'Use a fresh evaluation directory'
    plan = {'dataset_seeds':SEEDS,'learner_seed':101,'architecture':[9,32,10],'epochs':300,'batch':30,'lr':.2,
        'training_order':'Fixed supplied-row order, no shuffle, no stopping/architecture/hyperparameter selection',
        'optimization':'Frozen cost-only panel/layout selection. No accuracy-based selection.',
        'sampling':'Repository source_permutations:600 train from offset0,600 test from offset6000, original60000 pool',
        'preprocessing':'Repository float32 divide255,area_resize3,clip0..1; per-draw manifests capture actual bytes',
        'source_hashes':{path.name:sha(path) for path in (HERE/'learner.py',HERE/'reference.py',HERE/'panels.py',HERE/'selected.json')},
        'dataset_generator_sha256':sha(p.ROOT/'mnist/code/data.py'),
        'prediction_policy':'Write and hash all11 predictions before opening test-label slices in score phase',
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'created_unix_seconds':time.time()}
    json_write(output/'plan.json',plan)
    print(json.dumps(plan,indent=2))


def sources():
    paths = {}
    for key in ('train_images','train_labels'):
        filename,md5 = dataset.SOURCES[key]
        paths[key] = dataset.download_source(p.ROOT/'mnist/data/raw',filename,md5)
    return paths


def predict(output):
    plan = json.loads((output/'plan.json').read_text())
    assert plan['dataset_seeds']==SEEDS
    for name,digest in plan['source_hashes'].items():
        assert sha(HERE/name)==digest,name
    assert not (output/'predictions_frozen.json').exists(),'Do not overwrite frozen predictions'
    paths = sources()
    pixels = dataset.read_idx(paths['train_images'],60000,True)
    labels = dataset.read_idx(paths['train_labels'],60000,False)
    records = []
    for seed in SEEDS:
        directory = output/str(seed)
        directory.mkdir(exist_ok=True)
        order,_ = dataset.source_permutations(seed=seed)
        train_indices,test_indices = order[:600],order[6000:6600]
        assert not set(train_indices)&set(test_indices)
        arrays = {}
        for split,indices in (('train',train_indices),('test',test_indices)):
            images = dataset.area_resize(pixels[indices].astype(np.float32)/np.float32(255),3)
            arrays[split+'_images'] = np.clip(images,0,1)[:,None,:,:]
        arrays['train_labels'] = labels[train_indices].copy()
        metadata = {key:{'shape':list(value.shape),'dtype':str(value.dtype),
                    'sha256_c_order_little_endian':learner.array_hash(value)} for key,value in arrays.items()}
        manifest = {'profile':'competition-v2','seed':seed,'tiers':{'small':{'arrays':metadata,
                    'train_indices':train_indices.tolist(),'test_indices':test_indices.tolist()}},
                    'source_sha256':{key:sha(path) for key,path in paths.items()}}
        np.savez(directory/'inputs.npz',**arrays)
        json_write(directory/'manifest.json',manifest)
        allowed = learner.load_inputs(directory/'inputs.npz',directory/'manifest.json')
        result = learner.learn(allowed,'energy')
        baseline = learner.learn(allowed,'baseline')
        for key in ('params','scores','predictions'):
            assert result[key].tobytes()==baseline[key].tobytes(),(seed,key)
        prediction = result['predictions'].astype(np.int64)
        np.save(directory/'predictions.npy',prediction)
        records.append({'seed':seed,'learner_seed':101,'manifest_file':f'{seed}/manifest.json',
            'manifest_sha256':sha(directory/'manifest.json'),'prediction_file':f'{seed}/predictions.npy',
            'prediction_file_sha256':sha(directory/'predictions.npy'),'prediction_array_sha256':learner.array_hash(prediction),
            'parameter_sha256':learner.array_hash(result['params']),'score_sha256':learner.array_hash(result['scores']),
            'matches_original_baseline_bits':True})
        print('Predictions frozen',seed,flush=True)
    json_write(output/'predictions_frozen.json',{'plan_sha256':sha(output/'plan.json'),'draws':records,
                                             'frozen_unix_seconds':time.time(),'test_labels_scored':False})


def score(output):
    frozen = json.loads((output/'predictions_frozen.json').read_text())
    assert frozen['plan_sha256']==sha(output/'plan.json')
    assert [r['seed'] for r in frozen['draws']]==SEEDS
    predictions = []
    for record in frozen['draws']:
        assert sha(output/record['manifest_file'])==record['manifest_sha256']
        assert sha(output/record['prediction_file'])==record['prediction_file_sha256']
        array = np.load(output/record['prediction_file'],allow_pickle=False)
        assert array.shape==(600,) and array.dtype==np.int64
        assert learner.array_hash(array)==record['prediction_array_sha256']
        predictions.append(array)
    labels = dataset.read_idx(sources()['train_labels'],60000,False)
    rows = []
    for record,prediction in zip(frozen['draws'],predictions):
        manifest = json.loads((output/record['manifest_file']).read_text())
        correct = int(np.sum(prediction==labels[manifest['tiers']['small']['test_indices']]))
        rows.append({**record,'correct':correct,'total':600,'accuracy_percent':correct/6})
    values = np.array([r['accuracy_percent'] for r in rows])
    total = sum(r['correct'] for r in rows)
    result = {'draws':rows,'correct':total,'total':6600,'mean_percent':float(values.mean()),
        'sample_sd_pp':float(values.std(ddof=1)),'required_correct':3960,'meets_accuracy_target':total>=3960,
        'predictions_frozen_sha256':sha(output/'predictions_frozen.json'),'scored_unix_seconds':time.time(),
        'scope':'Fresh predeclared11draws after frozen final panel policy; no tuning after scoring'}
    json_write(output/'accuracy.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predeclare','predict','score'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    {'predeclare':predeclare,'predict':predict,'score':score}[args.phase](args.output)
