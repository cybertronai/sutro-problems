"""Export seed-only compiler literals and epoch hashes without loading a dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import schedule

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--n-train',type=int,default=6000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-initial',action='store_true',help='Requires CPU PyTorch matching the pinned learner environment')
    args=parser.parse_args()
    config=json.loads(args.config.read_text());config.setdefault('image_size',9)
    assert args.n_train>0 and len(set(args.seeds))==len(args.seeds)
    args.output.mkdir(parents=True,exist_ok=True)
    assert not any(args.output.iterdir()),'Output directory must be empty'
    sources={name:sha(HERE/name) for name in ('schedule.py','export_constants.py')}
    records=[]
    for seed in args.seeds:
        row={'seed':seed,'epochs':[]}
        if args.include_initial:
            initial=schedule.initial_parameters(config,seed)
            path=args.output/f'initial-seed-{seed}.npz'
            np.savez(path,**initial)
            row['initial']={'path':path.name,'file_sha256':sha(path),
                'arrays':{name:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':schedule.array_hash(a)} for name,a in initial.items()}}
        for order,indices,coefficients,manifest in schedule.iter_epochs(seed,args.n_train,
                config['epochs'],config['image_size'],config.get('augmentation','mild_affine')):
            row['epochs'].append(manifest)
        if config.get('dropout',0.):
            for epoch,mask in enumerate(schedule.head_masks(seed,args.n_train,config['epochs'],config['head_width'],config['dropout'])):
                row['epochs'][epoch]['head_mask_sha256']=schedule.array_hash(mask)
        records.append(row)
    assert sources=={name:sha(HERE/name) for name in sources},'Generator source changed'
    result={'schema_version':1,'config':config,'config_sha256':sha(args.config),'n_train':args.n_train,
        'seeds':args.seeds,'source_sha256':sources,'records':records,
        'software':{'python':platform.python_version(),'numpy':np.__version__},
        'scope':'Seed-only initial literals and schedules; no dataset, labels, learned parameters, or predictions were loaded.'}
    if args.include_initial:
        import torch
        result['software']['torch']=str(torch.__version__)
    (args.output/'constants.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'seeds':args.seeds,'dataset_loaded':False}))

if __name__=='__main__':main()
