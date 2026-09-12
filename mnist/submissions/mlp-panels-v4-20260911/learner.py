"""Train the frozen MNIST-small panel MLP and produce one label per query.

Only the three permitted NPZ arrays are read. Each invocation initializes a new
model and learns from the supplied training data; no saved parameters are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import panels
from reference import cpu

HERE = Path(__file__).resolve().parent
KEYS = ('train_images','train_labels','test_images')


def array_hash(array):
    value = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'),copy=False))
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_inputs(path,manifest=None):
    with np.load(path,allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in KEYS}
    for key in ('train_images','test_images'):
        value = arrays[key]
        assert value.shape==(600,1,3,3) and value.dtype==np.float32,key
        assert np.isfinite(value).all() and (value>=0).all() and (value<=1).all(),key
    labels = arrays['train_labels']
    assert labels.shape==(600,) and labels.dtype==np.int64
    assert ((labels>=0)&(labels<=9)).all()
    if manifest:
        metadata = json.loads(Path(manifest).read_text())['tiers']['small']['arrays']
        for key,value in arrays.items():
            assert list(value.shape)==metadata[key]['shape'],key
            assert str(value.dtype)==metadata[key]['dtype'],key
            assert array_hash(value)==metadata[key]['sha256_c_order_little_endian'],key
    return arrays


def config(variant='energy'):
    if variant=='baseline':
        return panels.default_config()
    if variant=='no_slowdown':
        return json.loads((HERE/'inference_refinement.json').read_text())['no_slowdown']['config']
    return json.loads((HERE/'selected.json').read_text())


def learn(data,variant='energy'):
    return cpu(data,config(variant),300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('energy','baseline','no_slowdown'),default='energy')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'predictions.npy').exists(),'Use a fresh output directory'
    arrays = load_inputs(args.data,args.manifest)
    start = time.perf_counter()
    result = learn(arrays,args.variant)
    elapsed = time.perf_counter()-start
    np.save(args.output/'predictions.npy',result['predictions'].astype(np.int64))
    np.save(args.output/'scores.npy',result['scores'])
    np.save(args.output/'parameters.npy',result['params'])
    metadata = {'variant':args.variant,'train':600,'test':600,'epochs':300,'learner_seed':101,
        'input_hashes':{key:array_hash(value) for key,value in arrays.items()},
        'output_hashes':{key:array_hash(result[key]) for key in ('params','scores','predictions')},
        'cpu_reference_wall_seconds':elapsed,'timing_scope':'CPU numerical implementation, not A100 or Dally timing',
        'test_labels_accessed':False}
    (args.output/'learner.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
