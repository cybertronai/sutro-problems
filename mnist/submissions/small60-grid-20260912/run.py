"""Prepare allowed inputs, freeze fresh-training predictions, then evaluate.

Run commands in README.md. Test labels never enter the learner input archives.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import reference

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from mnist.code import data as generator

SEEDS = list(range(20261201, 20261212))
CONFIG = dict(hidden_width=32, epochs=300, learning_rate=0.2, batch_size=25, learner_seed=101)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def check_protocol():
    p = json.loads((HERE/'protocol.json').read_text())
    assert p['config'] == CONFIG and p['dataset_seeds'] == SEEDS
    assert reference.BATCH == CONFIG['batch_size']
    for name, digest in p['source_sha256'].items():
        assert sha(HERE/name) == digest, name
    assert sha(Path(generator.__file__)) == p['generator_sha256']
    return p


def source(raw, key):
    filename, md5 = generator.SOURCES[key]
    path = generator.download_source(raw, filename, md5)
    return path, generator.read_idx(path, 60000, images=key.endswith('images'))


def prepare(raw):
    if (HERE/'protocol.json').exists():
        raise FileExistsError('Protocol already exists; never overwrite an evaluated study.')
    protocol = dict(created_at_utc=now(), config=CONFIG, dataset_seeds=SEEDS,
        dataset='1000 train and 1000 disjoint test examples from official 60000 training images',
        draw='numpy.random.Generator(PCG64(seed)).permutation(60000): first1000 train, next1000 test',
        preprocessing='float32 image/255 then generator.area_resize to3x3; learner transforms4*x-0.5',
        target_accuracy=0.60, target_total_correct=6600, planned_draws=11,
        selection='Architecture, hyperparameters and seed fixed from historical H32 learner; batch25 chosen to divide1000. No new-test tuning.',
        source_sha256={name: sha(HERE/name) for name in ('run.py','reference.py')},
        generator_sha256=sha(Path(generator.__file__)),
        training='Fresh seed101 initialization for each draw; ascending fixed sample order each epoch; all gradients from pre-update weights; no learned state transfer',
        evaluator='Only evaluate after prediction_manifest.json freezes all11 predictions')
    write(HERE/'protocol.json', protocol)
    image_path, images = source(raw, 'train_images')
    label_path, labels = source(raw, 'train_labels')
    private = HERE/'private'; private.mkdir(exist_ok=True)
    manifests=[]
    for index, seed in enumerate(SEEDS):
        order=np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train=order[:1000].astype(np.int64); test=order[1000:2000].astype(np.int64)
        assert len(np.unique(np.r_[train,test])) == 2000
        arrays={'train_images': generator.area_resize(images[train].astype(np.float32)/np.float32(255),3)[:,None],
                'train_labels': labels[train],
                'test_images': generator.area_resize(images[test].astype(np.float32)/np.float32(255),3)[:,None]}
        path=private/f'draw-{index:02d}.npz'
        np.savez_compressed(path, **arrays)
        manifests.append(dict(draw=index,dataset_seed=seed, train_indices=train.tolist(), test_indices=test.tolist(),
            train_test_disjoint=True, archive=str(path.relative_to(HERE)), archive_sha256=sha(path),
            arrays={name:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=generator.array_hash(a)) for name,a in arrays.items()}))
    write(HERE/'draw_manifest.json', dict(protocol_sha256=sha(HERE/'protocol.json'),
          source_files={p.name:sha(p) for p in (image_path,label_path)}, draws=manifests))
    print('Prepared11 allowed-input archives after freezing protocol; no test-label archives created.',flush=True)


def fit():
    check_protocol()
    if (HERE/'prediction_manifest.json').exists(): raise FileExistsError('Already frozen')
    manifest=json.loads((HERE/'draw_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(HERE/'protocol.json')
    predictions=HERE/'predictions'; predictions.mkdir(exist_ok=True)
    records=[]
    for d in manifest['draws']:
        path=HERE/d['archive']; assert sha(path)==d['archive_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)=={'train_images','train_labels','test_images'}
            arrays={name:z[name] for name in z.files}
        for name,a in arrays.items(): assert generator.array_hash(a)==d['arrays'][name]['sha256']
        start=time.perf_counter()
        x=reference.transform(arrays['train_images']); q=reference.transform(arrays['test_images'])
        target=(arrays['train_labels'][:,None]==np.arange(10)).astype(np.float32)
        params=reference.parameters(32,101)
        for _ in range(300): params=reference.epoch(x,target,params,0.2)
        scores=reference.forward(q,params)[2]
        assert np.isfinite(scores).all()
        pred=scores.argmax(1).astype(np.int64)
        seconds=time.perf_counter()-start
        ppath=predictions/f'draw-{d["draw"]:02d}.npy'; np.save(ppath,pred)
        records.append(dict(draw=d['draw'],dataset_seed=d['dataset_seed'],path=str(ppath.relative_to(HERE)),
            prediction_sha256=sha(ppath),array_sha256=generator.array_hash(pred),
            parameter_sha256=[generator.array_hash(a) for a in params],
            scores_sha256=generator.array_hash(scores),cpu_training_and_prediction_seconds=seconds))
        print(f'Frozen predictions draw {d["draw"]:02d}, CPU {seconds:.3f}s; no test labels read.',flush=True)
    write(HERE/'prediction_manifest.json',dict(frozen_at_utc=now(),test_labels_opened=False,
        protocol_sha256=sha(HERE/'protocol.json'),draw_manifest_sha256=sha(HERE/'draw_manifest.json'),draws=records,
        software=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform())))


def evaluate(raw, verify=False):
    check_protocol()
    freeze=json.loads((HERE/'prediction_manifest.json').read_text())
    manifest=json.loads((HERE/'draw_manifest.json').read_text())
    assert freeze['protocol_sha256']==sha(HERE/'protocol.json')
    assert freeze['draw_manifest_sha256']==sha(HERE/'draw_manifest.json')
    label_path, labels=source(raw,'train_labels')
    assert sha(label_path)==manifest['source_files'][label_path.name]
    counts=[]
    for d,p in zip(manifest['draws'],freeze['draws'],strict=True):
        assert d['draw']==p['draw'] and d['dataset_seed']==p['dataset_seed']
        order=np.random.Generator(np.random.PCG64(d['dataset_seed'])).permutation(60000)
        assert d['train_indices']==order[:1000].tolist() and d['test_indices']==order[1000:2000].tolist()
        path=HERE/p['path']; assert sha(path)==p['prediction_sha256']
        pred=np.load(path,allow_pickle=False)
        assert pred.shape==(1000,) and pred.dtype==np.int64 and ((pred>=0)&(pred<10)).all()
        assert generator.array_hash(pred)==p['array_sha256']
        correct=int(np.sum(pred==labels[np.asarray(d['test_indices'])]))
        counts.append(dict(draw=d['draw'],dataset_seed=d['dataset_seed'],correct=correct,total=1000,accuracy=correct/1000))
    acc=np.array([d['accuracy'] for d in counts])
    result=dict(draws=counts,correct=sum(d['correct'] for d in counts),total=11000,
        mean_accuracy=float(acc.mean()),sample_stddev_pp=float(acc.std(ddof=1)*100),
        target_accuracy=0.60,target_met=sum(d['correct'] for d in counts)>=6600,
        prediction_manifest_sha256=sha(HERE/'prediction_manifest.json'),
        protocol_sha256=sha(HERE/'protocol.json'))
    if verify:
        assert result==json.loads((HERE/'accuracy.json').read_text())
        print('Verified11 draw seeds, disjoint indices, prediction hashes, exact counts and sample SD.')
    else:
        if (HERE/'accuracy.json').exists():raise FileExistsError('Already evaluated')
        write(HERE/'evaluation_freeze.json',dict(evaluated_at_utc=now(),prediction_manifest_sha256=sha(HERE/'prediction_manifest.json')))
        write(HERE/'accuracy.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','fit','evaluate','verify'])
    parser.add_argument('--raw',type=Path,default=HERE/'private'/'raw')
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.raw)
    elif args.phase=='fit':fit()
    else:evaluate(args.raw,verify=args.phase=='verify')
