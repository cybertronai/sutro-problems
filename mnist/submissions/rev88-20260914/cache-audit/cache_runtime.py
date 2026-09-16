"""Run the frozen Rev88 learner with the qualified small cuBLAS workspace.

Import this module before initializing CUDA. The mathematical procedure is
unchanged. CLI output is a JSON metadata file plus an adjacent predictions.npy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = ':16:8'
HERE = Path(__file__).resolve().parent
BASE = HERE.parent

_torch = sys.modules.get('torch')
if _torch is not None and _torch.cuda.is_initialized():
    raise RuntimeError('Import cache_runtime before CUDA initialization; existing cuBLAS handles cannot be reconfigured reliably.')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
if str(BASE) not in sys.path:
    sys.path.insert(0,str(BASE))
_loaded = sys.modules.get('learner')
if _loaded is not None and Path(_loaded.__file__).resolve() != BASE/'learner.py':
    raise RuntimeError('A different learner module is already imported')
import learner as _learner
import numpy as np
import torch


def _guard():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != WORKSPACE:
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG changed after cache_runtime import')


def _annotate(metadata):
    return {**metadata,'cache_runtime':{
        'cublas_workspace_config':WORKSPACE,
        'configured_before_cuda_initialization':True,
        'frozen_learner_sha256':hashlib.sha256((BASE/'learner.py').read_bytes()).hexdigest(),
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Library workspace configuration only; original architecture, seeds, normalization, permutations, optimizer and training procedure.'}}


class PreparedTask:
    def __init__(self,task):
        self._task = task

    def __getattr__(self,name):
        return getattr(self._task,name)

    def reset(self):
        _guard()
        return self._task.reset()

    def run(self):
        _guard()
        return self._task.run()

    def outputs(self):
        predictions,metadata = self._task.outputs()
        return predictions,_annotate(metadata)


def prepare_task(train_images,train_labels,test_images,config):
    _guard()
    return PreparedTask(_learner.prepare_task(train_images,train_labels,test_images,config))


def train_predict(train_images,train_labels,test_images,config):
    task = prepare_task(train_images,train_labels,test_images,config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    task.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    predictions,metadata = task.outputs()
    metadata['training_prediction_seconds'] = elapsed
    return predictions,metadata


def validate_cuda_replay(config=None):
    _guard()
    return _annotate(_learner.validate_cuda_replay(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='NPZ containing exactly train_images, train_labels, test_images')
    parser.add_argument('--config',required=True,help='Frozen training configuration JSON')
    parser.add_argument('--output',required=True,help='Output JSON; predictions go to the adjacent .predictions.npy file')
    args = parser.parse_args()
    with np.load(args.input,allow_pickle=False) as archive:
        expected = {'train_images','train_labels','test_images'}
        if set(archive.files) != expected:
            raise ValueError('Input NPZ must contain exactly the three learner arrays; evaluation labels are forbidden')
        arrays = {key:archive[key].copy() for key in expected}
    config = json.loads(Path(args.config).read_text())
    predictions,metadata = train_predict(**arrays,config=config)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    prediction_path = output.with_suffix('.predictions.npy')
    np.save(prediction_path,predictions,allow_pickle=False)
    result = {'passed':True,'predictions_path':prediction_path.name,'metadata':metadata}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
