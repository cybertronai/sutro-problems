"""Numerically expand conv forward/backward/update leaves and audit every access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
import ir_conv as emit
from ir_core import Program, expand, score, make_program, ref as R, ins as I, loop as L
from ir_machine import SemanticMachine

HERE=Path(__file__).resolve().parent


def reference(x,w,b,delta):
    n,ci,h,ww=x.shape
    co=w.shape[0]
    px=np.pad(x,((0,0),(0,0),(1,1),(1,1)))
    pd=np.pad(delta,((0,0),(0,0),(1,1),(1,1)))
    out=np.zeros((n,co,h,ww),np.float32)
    dx=np.zeros_like(x);dw=np.zeros_like(w);db=np.zeros_like(b)
    for c in range(ci):
        for ky in range(3):
            for kx in range(3):
                out=out+px[:,c:c+1,ky:ky+h,kx:kx+ww]*w[None,:,c,ky,kx,None,None]
    out=out+b[None,:,None,None]
    for c in range(co):
        for ky in range(3):
            for kx in range(3):
                dx=dx+pd[:,c:c+1,ky:ky+h,kx:kx+ww]*w[None,c,:,2-ky,2-kx,None,None]
    for sample in range(n):
        for y in range(h):
            for xx in range(ww):
                dw=dw+delta[sample,:,y,xx,None,None,None]*px[sample,None,:,y:y+3,xx:xx+3]
                db=db+delta[sample,:,y,xx]
    return out,dx,dw,db


def case(n,ci,co,h,w,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,ci,h,w)).astype(np.float32)
    weights=rng.normal(size=(co,ci,3,3)).astype(np.float32)
    bias=rng.normal(size=co).astype(np.float32)
    delta=rng.normal(size=(n,co,h,w)).astype(np.float32)
    velocity=rng.normal(size=weights.shape).astype(np.float32)
    regions=[('s',3),('k',4),('x',x.size),('w',weights.size),('b',co),('delta',delta.size),
             ('v',weights.size),('px',n*ci*(h+2)*(w+2)),('pd',n*co*(h+2)*(w+2)),
             ('out',delta.size),('dx',x.size),('dw',weights.size),('db',co),
             ('relu',delta.size),('relud',delta.size)]
    body=[]
    for name,words in regions:
        body.append(L('init',words,[I('set',R(name,init=1),0)]))
    for i,value in enumerate([0,.0001,.9,.03]):
        body.append(I('set',R('k',i),int(np.float32(value).view(np.uint32))))
    arrays={'x':x,'w':weights,'b':bias,'delta':delta,'v':velocity}
    for name,a in arrays.items():
        body.append(L('recv_i',a.size,[I('recv',R(name,recv_i=1))]))
    body+=emit.pad_nchw('x','px',n,ci,h,w,'padx')
    body+=emit.pad_nchw('delta','pd',n,co,h,w,'padd')
    body+=emit.conv_forward('px','w','out',n,ci,co,h,w,'forward',bias='b')
    body+=emit.conv_input_gradient('pd','w','dx',n,ci,co,h,w,'dx')
    body+=emit.conv_weight_gradient('px','delta','dw',n,ci,co,h,w,'dw')
    body+=emit.conv_bias_gradient('delta','db',n,co,h,w,'db')
    body+=emit.relu_forward('out','relu',delta.size,'relu')
    body+=emit.relu_backward('relu','delta','relud',delta.size,'relud')
    body+=emit.momentum_update('w','dw','v',weights.size,'update')
    body.append(I('send',R('w',0)))
    document=make_program(regions,body,{'purpose':'Synthetic exact conv primitive validation'})
    parsed=Program(document)
    tape=np.concatenate([a.ravel().view(np.uint32) for a in arrays.values()])
    machine=SemanticMachine(parsed.coordinates,tape)
    machine.run(expand(document))
    out,dx,dw,db=reference(x,weights,bias,delta)
    adjusted=dw+np.float32(.0001)*weights
    next_velocity=np.float32(.9)*velocity+adjusted
    next_weights=weights-np.float32(.03)*next_velocity
    expected={'out':out,'dx':dx,'dw':dw,'db':db,'v':next_velocity,'w':next_weights,
              'relu':np.where(out>0,out,np.float32(0)),
              'relud':np.where(out>0,delta,np.float32(0))}
    for name,value in expected.items():
        start,words=parsed.regions[name]
        actual=np.asarray(machine.memory[start:start+words],np.uint32)
        np.testing.assert_array_equal(actual,value.ravel().view(np.uint32),err_msg=name)
    result,reads,writes=score(document,include_counts=True)
    np.testing.assert_array_equal(reads,machine.read_counts)
    np.testing.assert_array_equal(writes,machine.write_counts)
    assert result['time_ticks_0_2_ps']==machine.time_ticks
    assert result['energy_fj']==machine.energy_fj
    assert result['instructions']==dict(machine.instructions)
    return {'n':n,'cin':ci,'cout':co,'height':h,'width':w,
            'all_forward_backward_relu_optimizer_bits_match':True,
            'every_address_count_and_physical_score_matches':True,
            'expanded_instructions':result['total_instructions'],'program_sha256':result['program_sha256']}


if __name__=='__main__':
    started=time.perf_counter()
    cases=[case(*dimensions,seed=76+i) for i,dimensions in enumerate(
           [(1,1,1,1,1),(2,2,3,3,4),(2,3,2,9,9)])]
    result={'all_passed':True,'cases':cases,'wall_seconds':time.perf_counter()-started,
            'source_sha256':{str(path.relative_to(HERE.parents[2])):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__),HERE/'ir_conv.py',HERE/'ir_core.py',HERE/'ir_tables.py',HERE/'ir_machine.py',
                             HERE.parents[1]/'submissions/1nn-v4-20260911/score_v4.py')},
            'scope':'Synthetic fully expanded 3x3 NCHW conv forward/dInput/dWeight/dBias, ReLU and coupled-decay momentum update; all raw FP32 bits, every access count, exact geometry costs. This is not a full classifier or an accuracy claim.'}
    (HERE/'ir-conv-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
