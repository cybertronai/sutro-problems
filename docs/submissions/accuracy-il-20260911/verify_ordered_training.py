"""Replay the validation-best final model with explicit FP32 primitive reductions.

This check reads only allowlisted learning inputs and an already frozen
train-only shortlist. It never reads test labels. Run before the test evaluator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import accuracy_study as study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('mnist/data/small.npz'))
    p.add_argument('--manifest', type=Path, default=Path('mnist/doc/dataset_manifest.json'))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    selected = json.loads((args.output/'shortlist.json').read_text())
    config = min(selected['candidates'], key=lambda r: (-r['validation_correct'],
                        r['epochs']*r['width'], r['width'], r['learning_rate']))
    seed = selected['final_seeds'][0]
    data = study.inputs(args.data,args.manifest)
    x, q = study.transform(data['train_images']), study.transform(data['test_images'])
    target = (data['train_labels'][:,None]==np.arange(10)).astype(np.float32)
    fast, slow = study.parameters(config['width'],seed), study.parameters(config['width'],seed)
    start = time.perf_counter()
    for ep in range(1,config['epochs']+1):
        slow = study.epoch(x,target,slow,config['learning_rate'],study.ordered_mm,study.ordered_rows)
        fast = study.epoch(x,target,fast,config['learning_rate'])
        if ep == 1 or ep % 1000 == 0 or ep == config['epochs']:
            assert all(np.array_equal(a,b) for a,b in zip(fast,slow)), f'epoch {ep}'
            print(f"{config['id']} seed{seed}: epoch{ep}, all parameters bitwise equal",flush=True)
    scores_fast = study.forward(q,fast)[2]
    scores_ordered = study.forward(q,slow,study.ordered_mm)[2]
    assert np.array_equal(scores_fast,scores_ordered)
    predictions = study.predict(q,slow,study.ordered_mm)
    result = {'config_id':config['id'],'seed':seed,'epochs':config['epochs'],
              'width':config['width'],'learning_rate':config['learning_rate'],
              'selection':'best train-only validation; first predeclared final seed',
              'all_final_parameters_bitwise_equal':True,
              'all_600_final_score_vectors_bitwise_equal':True,
              'parameter_sha256':[study.digest(a) for a in slow],
              'predictions_sha256_int64_le':study.digest(predictions),
              'replay_and_reference_comparison_wall_seconds':time.perf_counter()-start,
              'shortlist_sha256':hashlib.sha256((args.output/'shortlist.json').read_bytes()).hexdigest(),
              'learner_source_sha256':hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
              'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'test_labels_accessed':False}
    study.write_json(args.output/'ordered_training_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
