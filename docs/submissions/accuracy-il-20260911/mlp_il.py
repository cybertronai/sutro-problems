"""Lower the exact fixed-batch ReLU/SGD learner to affine-loop v4 primitives.

No learned values enter the program: initial weights are seed-only raw literals.
Gradient entries are streamed through one accumulator once hidden deltas exist,
so all updates use the same pre-update dependency values without gradient arrays.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from il import ref as R, ins as I, loop as L, make_program, score
from accuracy_study import parameters

HERE = Path(__file__).resolve().parent


def raw(value):
    return int(np.float32(value).view(np.uint32))


def build_mlp(width, epochs, learning_rate, seed=101, n_train=600, n_test=600, batch=30):
    if n_train % batch:
        raise ValueError('Only complete fixed-size minibatches are supported')
    if min(width, epochs, n_train, n_test, batch) <= 0:
        raise ValueError('All dimensions and epochs must be positive')
    H, D, C, B, N, Q = width, 9, 10, batch, n_train, n_test
    regions = [('s',5),('k',15),('w1',D*H),('b1',H),('w2',H*C),('b2',C),
               ('h',B*H),('d1',B*H),('d2',B*C),('x',N*D),('labels',N),
               ('target',N*C),('q',Q*D)]
    t, a, cond, best, label = [R('s',i) for i in range(5)]
    zero, one, four, half, step = [R('k',i) for i in range(5)]
    body = []
    # Explicit allocation initialization is modeled work, not a free assumption.
    for region, words in regions:
        body.append(L('init',words,[I('set',R(region,init=1),0)]))
    for index, value in enumerate([raw(0),raw(1),raw(4),raw(.5),raw(learning_rate/B)]+list(range(C))):
        body.append(I('set',R('k',index),value))
    initial = parameters(H,seed)
    for region, values in [('w1',initial[0]),('w2',initial[2])]:
        body += [I('set',R(region,index),int(value))
                 for index,value in enumerate(values.reshape(-1).view(np.uint32))]
    # Fixed tape: all train pixels, raw uint32 train labels, all test pixels.
    for region, words in [('x',N*D),('labels',N),('q',Q*D)]:
        body.append(L('recv_i',words,[I('recv',R(region,recv_i=1))]))
    for region, words in [('x',N*D),('q',Q*D)]:
        v=R(region,norm_i=1)
        body.append(L('norm_i',words,[I('mul',v,v,four),I('sub',v,v,half)]))
    # Raw integer label words compare equal to raw integer class literals;
    # select converts the boolean to the FP32 one-hot representation.
    for c in range(C):
        body.append(L('target_i',N,[I('cmp',cond,R('labels',target_i=1),R('k',5+c),predicate='eq'),
                      I('select',R('target',c,target_i=C),cond,one,zero)]))

    batch_body=[]
    hidden=R('h',b=H,h=1)
    # X_batch @ W1 + b1, ascending feature reduction; ReLU.
    batch_body.append(L('b',B,[L('h',H,[I('set',a,0),L('f',D,[
        I('mul',t,R('x',batch=B*D,b=D,f=1),R('w1',f=H,h=1)),I('add',a,a,t)]),
        I('add',a,a,R('b1',h=1)),I('cmp',cond,zero,a),I('select',hidden,cond,a,zero)])]))
    # H @ W2 + b2, immediately subtract target into delta2.
    delta2=R('d2',b=C,c=1)
    batch_body.append(L('b',B,[L('c',C,[I('set',a,0),L('h',H,[
        I('mul',t,R('h',b=H,h=1),R('w2',h=C,c=1)),I('add',a,a,t)]),
        I('add',a,a,R('b2',c=1)),I('sub',delta2,a,R('target',batch=B*C,b=C,c=1))])]))
    # delta2 @ old W2.T, masked by the ReLU activation. Strict >0 matches z>0.
    batch_body.append(L('b',B,[L('h',H,[I('set',a,0),L('c',C,[
        I('mul',t,R('d2',b=C,c=1),R('w2',h=C,c=1)),I('add',a,a,t)]),
        I('cmp',cond,zero,R('h',b=H,h=1)),I('select',R('d1',b=H,h=1),cond,a,zero)])]))
    # Each gradient uses saved activations/deltas; streaming it avoids a
    # materialized gradient matrix and preserves all pre-update dependencies.
    w=R('w1',f=H,h=1)
    batch_body.append(L('f',D,[L('h',H,[I('set',a,0),L('b',B,[
        I('mul',t,R('x',batch=B*D,b=D,f=1),R('d1',b=H,h=1)),I('add',a,a,t)]),
        I('mul',t,step,a),I('sub',w,w,t)])]))
    w=R('b1',h=1)
    batch_body.append(L('h',H,[I('set',a,0),L('b',B,[I('add',a,a,R('d1',b=H,h=1))]),
                              I('mul',t,step,a),I('sub',w,w,t)]))
    w=R('w2',h=C,c=1)
    batch_body.append(L('h',H,[L('c',C,[I('set',a,0),L('b',B,[
        I('mul',t,R('h',b=H,h=1),R('d2',b=C,c=1)),I('add',a,a,t)]),
        I('mul',t,step,a),I('sub',w,w,t)])]))
    w=R('b2',c=1)
    batch_body.append(L('c',C,[I('set',a,0),L('b',B,[I('add',a,a,R('d2',b=C,c=1))]),
                              I('mul',t,step,a),I('sub',w,w,t)]))
    body.append(L('epoch',epochs,[L('batch',N//B,batch_body)]))

    # Inference streams queries using the first activation-buffer row.
    query_body=[L('h',H,[I('set',a,0),L('f',D,[
        I('mul',t,R('q',query=D,f=1),R('w1',f=H,h=1)),I('add',a,a,t)]),
        I('add',a,a,R('b1',h=1)),I('cmp',cond,zero,a),I('select',R('h',h=1),cond,a,zero)])]
    for c in range(C):
        query_body += [I('set',a,0),L('h',H,[I('mul',t,R('h',h=1),R('w2',c,h=C)),I('add',a,a,t)]),
                       I('add',a,a,R('b2',c))]
        if c==0:
            query_body += [I('copy',best,a),I('copy',label,R('k',5))]
        else:
            query_body += [I('cmp',cond,best,a),I('select',best,cond,a,best),
                           I('select',label,cond,R('k',5+c),label)]
    query_body.append(I('send',label))
    body.append(L('query',Q,query_body))
    return make_program(regions,body,{
        'algorithm':'ordered-FP32 9-H-10 ReLU squared-error minibatch SGD',
        'width':H,'epochs':epochs,'learning_rate':learning_rate,'seed':seed,
        'train_examples':N,'test_examples':Q,'batch_size':B,
        'training_order':'fixed cyclic supplied-row order, no shuffle',
        'tape':'train pixels FP32 bits, train labels uint32, test pixels FP32 bits',
        'normalization':'x*float32(4)-float32(0.5)',
        'gradient_lowering':'stream gradient entries after all delta1 values computed; pre-update dependencies preserved',
        'initialization':'all scratch explicitly zeroed (charged); seeded initial weights set as raw FP32 literals',
    })


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--learning-rate',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    document=build_mlp(args.width,args.epochs,args.learning_rate,args.seed)
    name=f'mlp-h{args.width}-e{args.epochs}-s{args.seed}'
    path=args.output/(name+'.il.json');path.write_text(json.dumps(document,indent=2)+'\n')
    result=score(document)
    result['program_file']=path.name
    result['program_bytes']=path.stat().st_size
    result['lowering_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/(name+'.score.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
