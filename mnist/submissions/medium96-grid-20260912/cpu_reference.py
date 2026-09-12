"""Independent exact-FP32 CPU replay; learner inputs exclude test labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from mnist.code import data

def mm(a,b):
    return np.einsum('ik,kj->ij',a,np.ascontiguousarray(b),optimize=False,dtype=np.float32)

def rows(a):return np.einsum('ij->j',a,optimize=False,dtype=np.float32)

def scalar_mm(a,b):
    out=np.zeros((a.shape[0],b.shape[1]),np.float32)
    for k in range(a.shape[1]):out=out+a[:,k,None]*b[None,k,:]
    return out

def parameters(config,seed):
    w=config['width'];rng=np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1/math.sqrt(81),1/math.sqrt(81),(81,w)).astype(np.float32),np.zeros(w,np.float32),
            rng.uniform(-1/math.sqrt(w),1/math.sqrt(w),(w,10)).astype(np.float32),np.zeros(10,np.float32)]

def learn(train_images,train_labels,test_images,config,seed):
    x=train_images.reshape(len(train_images),81)*np.float32(4)-np.float32(.5)
    q=test_images.reshape(len(test_images),81)*np.float32(4)-np.float32(.5)
    target=(train_labels[:,None]==np.arange(10)).astype(np.float32)
    w1,b1,w2,b2=parameters(config,seed)
    batch=config['batch_size'];assert len(x)%batch==0
    step=np.float32(config['learning_rate']/batch)
    for _ in range(config['epochs']):
        for first in range(0,len(x),batch):
            xb=x[first:first+batch];t=target[first:first+batch]
            h=np.maximum(mm(xb,w1)+b1,np.float32(0))
            d2=mm(h,w2)+b2-t
            d1=np.where(h>0,mm(d2,w2.T),np.float32(0))
            w1=w1-step*mm(xb.T,d1);b1=b1-step*rows(d1)
            w2=w2-step*mm(h.T,d2);b2=b2-step*rows(d2)
    scores=mm(np.maximum(mm(q,w1)+b1,np.float32(0)),w2)+b2
    return scores.argmax(1).astype('<i8'),np.concatenate([a.ravel() for a in (w1,b1,w2,b2)]),scores

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draw',type=int,default=0)
    p.add_argument('--output',type=Path,default=HERE/'cpu-verification-draw00.json')
    args=p.parse_args();protocol=json.loads((HERE/'protocol.json').read_text())
    rng=np.random.default_rng(9182)
    for n,k,m in ((25,81,512),(25,512,10),(81,25,512),(512,25,10),(25,10,512)):
        a=rng.normal(size=(n,k)).astype(np.float32);b=rng.normal(size=(k,m)).astype(np.float32)
        assert np.array_equal(mm(a,b).view(np.uint32),scalar_mm(a,b).view(np.uint32))
    paths={name:data.download_source(Path('/tmp/mnist-medium96-current/raw'),*data.SOURCES[name]) for name in ('train_images','train_labels')}
    pixels=data.read_idx(paths['train_images'],60000,True)
    labels=data.read_idx(paths['train_labels'],60000,False)
    ds=protocol['dataset_seeds'][args.draw];ls=protocol['learner_seeds'][args.draw]
    order=np.random.Generator(np.random.PCG64(ds)).permutation(60000)
    def resize(ix):
        a=data.area_resize(pixels[ix].astype(np.float32)/np.float32(255),9);np.clip(a,0,1,out=a);return a[:,None,:,:]
    arrays={'train_images':resize(order[:10000]),'train_labels':labels[order[:10000]],'test_images':resize(order[10000:20000])}
    start=time.perf_counter();pred,params,scores=learn(**arrays,config=protocol['configuration'],seed=ls);seconds=time.perf_counter()-start
    result=json.loads((HERE/'results'/f'draw-{args.draw:02d}.json').read_text())
    assert {k:data.array_hash(v) for k,v in arrays.items()}==result['input_sha256']
    assert data.array_hash(params)==result['final_parameter_sha256']
    assert data.array_hash(scores)==result['validation']['captured_task']['scores_sha256_float32_le']
    assert np.array_equal(pred,np.load(HERE/'predictions'/f'draw-{args.draw:02d}.npy',allow_pickle=False))
    record={'draw':args.draw,'dataset_seed':ds,'learner_seed':ls,'numpy':np.__version__,'cpu_reference_seconds':seconds,
            'scope':'Complete200epoch CPU replay, all final parameters, all100000 scores and10000 predictions bitwise equal to A100; no test labels used',
            'parameters_compared':params.size,'scores_compared':scores.size,'predictions_compared':pred.size,
            'parameter_sha256':data.array_hash(params),'scores_sha256':data.array_hash(scores),'predictions_sha256_int64':data.array_hash(pred),
            'arithmetic_checks':'Five actual training matrix shapes equal separate ascending scalar FP32 reduction',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
