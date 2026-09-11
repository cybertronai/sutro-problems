"""Predeclare a bounded train-only search and freeze one final configuration.

Never loads test images or test labels. Completed records are not overwritten.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import learner

HERE=Path(__file__).resolve().parent
WIDTHS=(128,256,512)
RATES=(0.01,0.03,0.1,0.2)
CHECKPOINTS=(25,50,100,200,300)
SPLIT_SEED=20260913


def worker(task):
    width,data_path,manifest_path,output_path=task
    output=Path(output_path)
    data=learner.inputs(Path(data_path),Path(manifest_path),queries=False)
    indices=np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(6000)
    train,val=indices[:4800],indices[4800:]
    x=learner.transform(data['train_images'])
    y=data['train_labels']
    train_x=x[train]
    targets=(y[train,None]==np.arange(10)).astype(np.float32)
    rows=[]
    for rate in RATES:
        params=learner.parameters(width,11)
        begin=time.perf_counter()
        for ep in range(1,max(CHECKPOINTS)+1):
            with np.errstate(over='ignore',invalid='ignore'):
                params=learner.epoch(train_x,targets,params,rate)
            if not all(np.isfinite(p).all() for p in params):
                rows.append({'width':width,'learning_rate':rate,'epochs':ep,'status':'nonfinite_training'})
                print(f'H{width} rate{rate}: nonfinite epoch{ep}',flush=True)
                break
            if ep in CHECKPOINTS:
                prediction=learner.forward(x[val],params)[2].argmax(axis=1)
                correct=int((prediction==y[val]).sum())
                row={'width':width,'learning_rate':rate,'epochs':ep,'status':'finite',
                     'validation_correct':correct,'validation_total':1200,
                     'training_correct':int((learner.forward(train_x,params)[2].argmax(axis=1)==y[train]).sum()),
                     'training_total':4800,'wall_seconds_to_checkpoint':time.perf_counter()-begin}
                rows.append(row)
                learner.save_json(output/f'validation-h{width}.json',rows)
                print(f'H{width} rate{rate} epoch{ep}: validation {correct}/1200',flush=True)
    learner.save_json(output/f'validation-h{width}.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('mnist/data/medium.npz'))
    parser.add_argument('--manifest',type=Path,default=Path('mnist/doc/dataset_manifest.json'))
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/name).exists() for name in ('predeclared_plan.json','config.json')):
        raise RuntimeError('Search evidence already exists; refuse to overwrite it.')
    data=learner.inputs(args.data,args.manifest,queries=False)
    order=np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(6000)
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'widths':WIDTHS,
          'learning_rates':RATES,'checkpoint_epochs':CHECKPOINTS,'search_seed':11,'final_seed':101,
          'validation_split_seed':SPLIT_SEED,'validation_fit_rows':4800,'validation_holdout_rows':1200,
          'batch_size':30,'algorithm':'81-H-10 ReLU, squared-error SGD, fixed contiguous minibatches, separate FP32 operations, ascending reductions, norm4x-.5',
          'target_percent':98,'repository_target_percent':'98.14',
          'selection':'If any checkpoint reaches 1176/1200 (98%), minimize epochs*width then width then learning_rate among those. Otherwise maximize validation_correct; break ties by epochs*width, width, learning_rate. Freeze one configuration and seed101 before opening test labels.',
          'test_policy':'No test images or labels are opened by this script. Final learner saves predictions before a separate evaluator opens labels. No further search after test evaluation.',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/'learner.py')},
          'training_input_sha256':{k:learner.digest(v) for k,v in data.items()}}
    learner.save_json(args.output/'predeclared_plan.json',plan)
    learner.save_json(args.output/'validation_split.json',{'training_rows':order[:4800].tolist(),'validation_rows':order[4800:].tolist()})
    tasks=[(width,str(args.data.resolve()),str(args.manifest.resolve()),str(args.output.resolve())) for width in WIDTHS]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[row for group in pool.map(worker,tasks) for row in group]
    finite=[row for row in rows if row['status']=='finite']
    if not finite:raise RuntimeError('No finite candidate')
    def cost(row):return (row['epochs']*row['width'],row['width'],row['learning_rate'])
    eligible=[row for row in finite if row['validation_correct']>=1176]
    chosen=min(eligible,key=cost) if eligible else min(finite,key=lambda row:(-row['validation_correct'],*cost(row)))
    config={k:chosen[k] for k in ('width','epochs','learning_rate')}
    config.update({'seed':101,'batch_size':30,'features':81,'n_train':6000,'n_test':6000})
    learner.save_json(args.output/'validation_results.json',{'rows':rows,'selected':chosen,
             'search_wall_seconds':time.perf_counter()-start,'target_reached_on_validation':bool(eligible),
             'config':config,'frozen_at_utc':datetime.now(timezone.utc).isoformat()})
    learner.save_json(args.output/'config.json',config)
    print('FROZEN CONFIG '+json.dumps(config),flush=True)


if __name__ == '__main__':
    main()
