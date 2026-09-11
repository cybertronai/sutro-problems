"""Fixed-shape ordered FP32 medium learner; only three allowed input arrays.

The configuration is selected on training-only validation before test evaluation.
Saved parameters and predictions are evidence, never inputs to this learner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE = ROOT/'mnist/experiments/accuracy-il-20260911/accuracy_study.py'
sys.path.insert(0, str(REFERENCE.parent))
from accuracy_study import fast_mm, fast_rows, ordered_mm, ordered_rows, forward, update, epoch


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inputs(path, manifest_path, queries=True):
    names = ['train_images','train_labels'] + (['test_images'] if queries else [])
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in names}
    canonical = json.loads(manifest_path.read_text())['tiers']['medium']['arrays']
    for name,array in data.items():
        assert list(array.shape) == canonical[name]['shape'], name
        assert str(array.dtype) == canonical[name]['dtype'], name
        assert digest(array) == canonical[name]['sha256_c_order_little_endian'], name
    return data


def parameters(width, seed, features=81):
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1/math.sqrt(features),1/math.sqrt(features),(features,width)).astype(np.float32),
            np.zeros(width,np.float32),
            rng.uniform(-1/math.sqrt(width),1/math.sqrt(width),(width,10)).astype(np.float32),
            np.zeros(10,np.float32)]


def transform(images):
    return images.reshape(len(images),-1)*np.float32(4)-np.float32(.5)


def fit(images, labels, config, mm=fast_mm, rows=fast_rows):
    x = transform(images)
    target = (labels[:,None] == np.arange(10)).astype(np.float32)
    assert config['batch_size'] == 30 and len(x)%30 == 0
    params = parameters(config['width'],config['seed'],x.shape[1])
    for _ in range(config['epochs']):
        params = epoch(x,target,params,config['learning_rate'],mm,rows)
    return params


def learn(data, config):
    params = fit(data['train_images'],data['train_labels'],config)
    scores = forward(transform(data['test_images']),params)[2]
    predictions = scores.argmax(axis=1).astype(np.int64)
    return predictions,params,scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('mnist/data/medium.npz'))
    parser.add_argument('--manifest',type=Path,default=Path('mnist/doc/dataset_manifest.json'))
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    data=inputs(args.data,args.manifest)
    start=time.perf_counter()
    predictions,params,scores=learn(data,config)
    elapsed=time.perf_counter()-start
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/'predictions.npy',predictions)
    np.save(args.output/'output-scores.npy',scores)
    np.savez(args.output/'parameters.npz',**dict(zip(['w1','b1','w2','b2'],params)))
    record={'configuration':config,'input_sha256':{k:digest(v) for k,v in data.items()},
            'decoded_input_members':list(data),'prediction_sha256_int64_le':digest(predictions),
            'parameter_sha256':{k:digest(v) for k,v in zip(['w1','b1','w2','b2'],params)},
            'output_scores_sha256':digest(scores),'cpu_reference_seconds':elapsed,
            'cpu_reference_scope':'normalization, one-hot targets, seed-only initial constants, complete training and prediction; excludes loading, hashing, output files',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),REFERENCE)},
            'config_file_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'software':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
    save_json(args.output/'cpu_results.json',record)
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
