"""Rebuild the frozen eleven datasets, without placing test labels in runner inputs."""
import argparse, gzip, hashlib, json
from pathlib import Path
import numpy as np
from mnist.code import data

def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--a100-record', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    if np.__version__!='2.1.2': raise ValueError('Use NumPy2.1.2')
    for kind in ('train_images','train_labels'):
        name,expected=data.SOURCES[kind]
        if data.file_hash(args.raw_dir/name,'md5')!=expected: raise ValueError(name)
    pixels=data.read_idx(args.raw_dir/data.SOURCES['train_images'][0],60000,True)
    labels=data.read_idx(args.raw_dir/data.SOURCES['train_labels'][0],60000,False)
    with gzip.open(args.a100_record,'rt') as stream: evidence=json.load(stream)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for row in evidence['qualification']['draws']:
        draw,seed=row['draw'],row['seed']
        order=np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train,query=order[:10000],order[10000:20000]
        resize=lambda index:data.area_resize(pixels[index].astype(np.float32)/np.float32(255),9).reshape(10000,81)
        arrays={'train_indices':train,'query_indices':query,'train_pixels':resize(train),'train_labels':labels[train],'query_pixels':resize(query),'test_labels':labels[query]}
        hashes={key:digest(value) for key,value in arrays.items()}
        if hashes!=row['dataset']['array_sha256']: raise ValueError(f'A100 input hash mismatch draw{draw}')
        destination=args.output_dir/f'draw_{draw:02d}.npz'
        np.savez(destination,x=arrays['train_pixels'],y=arrays['train_labels'],q=arrays['query_pixels'])
        manifest.append({'draw':draw,'seed':seed,'filename':destination.name,'file_sha256':data.file_hash(destination),'array_sha256':hashes})
        print(f'draw{draw:02d}: canonical inputs match A100 hashes',flush=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
