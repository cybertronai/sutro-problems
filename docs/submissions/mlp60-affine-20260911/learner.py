"""Fixed H32 / 300-epoch / seed-101 MNIST-small learner.

Only train_images, train_labels, and test_images are decoded. The implementation
imports the frozen ordered-FP32 reference; no test labels or saved predictions
are used to compute the output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]/'experiments'/'accuracy-il-20260911'
sys.path.insert(0, str(EXPERIMENT))
import accuracy_study as reference


def learn(data):
    x = reference.transform(data['train_images'])
    q = reference.transform(data['test_images'])
    target = (data['train_labels'][:, None] == np.arange(10)).astype(np.float32)
    params = reference.parameters(32, 101)
    for _ in range(300):
        params = reference.epoch(x, target, params, 0.2)
    scores = reference.forward(q, params)[2]
    return scores.argmax(axis=1).astype(np.int64), params, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('mnist/data/small.npz'))
    parser.add_argument('--manifest', type=Path, default=Path('mnist/doc/dataset_manifest.json'))
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    data = reference.inputs(args.data, args.manifest)
    begin = time.perf_counter()
    predictions, params, scores = learn(data)
    elapsed = time.perf_counter() - begin
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output/'predictions.npy', predictions)
    np.save(args.output/'output-scores.npy', scores)
    np.savez(args.output/'parameters.npz', **dict(zip(['w1','b1','w2','b2'],params)))
    record = {
        'configuration': {'hidden_width':32,'epochs':300,'learning_rate':0.2,'batch_size':30,'seed':101},
        'decoded_input_members':['train_images','train_labels','test_images'],
        'input_sha256':{key:reference.digest(value) for key,value in data.items()},
        'prediction_sha256_int64_le':reference.digest(predictions),
        'parameter_sha256':{key:reference.digest(value) for key,value in zip(['w1','b1','w2','b2'],params)},
        'output_scores_sha256':reference.digest(scores),
        'cpu_reference_seconds':elapsed,
        'cpu_reference_scope':'input normalization, one-hot targets, seed-only initialization, complete training and inference; excludes loading, hashing, and output files',
        'software':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
        'source_sha256':{str(path.relative_to(HERE.parents[2])):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [Path(__file__),EXPERIMENT/'accuracy_study.py']},
    }
    (args.output/'cpu_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
