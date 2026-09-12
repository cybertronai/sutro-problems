"""Lower the exact fixed-batch ReLU/SGD learner to affine-loop v4 primitives.

No learned values enter the program: initial weights are seed-only raw literals.
Gradient entries are streamed through one accumulator once hidden deltas exist,
so all updates use the same pre-update dependency values without gradient arrays.
"""
from __future__ import annotations
import math
import numpy as np

from affine import ref as R, ins as I, loop as L, make_program

def initial_parameters(features, width, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1 / math.sqrt(features), 1 / math.sqrt(features), (features, width)).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width), (width, 10)).astype(np.float32),
            np.zeros(10, dtype=np.float32)]


def raw(value):
    return int(np.float32(value).view(np.uint32))


def build_mlp(width, epochs, learning_rate, seed=101, n_train=6000, n_test=6000, batch=30, features=81, stream_queries=True):
    if min(width, epochs, n_train, n_test, batch, features) <= 0:
        raise ValueError('All dimensions and epochs must be positive')
    if n_train % batch:
        raise ValueError('Only complete fixed-size minibatches are supported')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    H, D, C, B, N, Q = width, features, 10, batch, n_train, n_test
    regions = [('s',5),('k',15),('w1',D*H),('b1',H),('w2',H*C),('b2',C),
               ('h',B*H),('d1',B*H),('d2',B*C),('x',N*D),('labels',N),
               ('target',N*C),('q',Q*D)]
    if stream_queries:
        # Reuse one query beside the hot scalars. Training data and targets stay
        # resident; query tape words are consumed only after training finishes.
        regions = regions[:2] + [('q', D)] + regions[2:-1]
    t, a, cond, best, label = [R('s',i) for i in range(5)]
    zero, one, four, half, step = [R('k',i) for i in range(5)]
    body = []
    # Explicit allocation initialization is modeled work, not a free assumption.
    for region, words in regions:
        body.append(L('init',words,[I('set',R(region,init=1),0)]))
    for index, value in enumerate([raw(0),raw(1),raw(4),raw(.5),raw(learning_rate/B)]+list(range(C))):
        body.append(I('set',R('k',index),value))
    initial = initial_parameters(D,H,seed)
    for region, values in [('w1',initial[0]),('w2',initial[2])]:
        body += [I('set',R(region,index),int(value))
                 for index,value in enumerate(values.reshape(-1).view(np.uint32))]
    # Fixed tape: all train pixels, raw uint32 train labels, all test pixels.
    input_regions = [('x',N*D),('labels',N)] + ([] if stream_queries else [('q',Q*D)])
    for region, words in input_regions:
        body.append(L('recv_i',words,[I('recv',R(region,recv_i=1))]))
    normalization_regions = [('x',N*D)] + ([] if stream_queries else [('q',Q*D)])
    for region, words in normalization_regions:
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
    query_pixel = R('q',f=1) if stream_queries else R('q',query=D,f=1)
    query_body = []
    if stream_queries:
        query_body.append(L('recv_query', D, [I('recv', R('q', recv_query=1))]))
        v = R('q', norm_query=1)
        query_body.append(L('norm_query', D, [I('mul', v, v, four), I('sub', v, v, half)]))
    query_body += [L('h',H,[I('set',a,0),L('f',D,[
        I('mul',t,query_pixel,R('w1',f=H,h=1)),I('add',a,a,t)]),
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
        'algorithm':f'ordered-FP32 {D}-H-10 ReLU squared-error minibatch SGD',
        'width':H,'epochs':epochs,'learning_rate':learning_rate,'seed':seed,
        'train_examples':N,'test_examples':Q,'batch_size':B,
        'training_order':'fixed cyclic supplied-row order, no shuffle',
        'tape':'train pixels FP32 bits, train labels uint32, test pixels FP32 bits',
        'normalization':'x*float32(4)-float32(0.5)',
        'gradient_lowering':'stream gradient entries after all delta1 values computed; pre-update dependencies preserved',
        'initialization':'all scratch explicitly zeroed (charged); seeded initial weights set as raw FP32 literals',
        **({'features': D, 'stream_queries': True,
            'query_layout': 'one query immediately after hot scalars/constants; receive, normalize, infer, and send after training'} if stream_queries else {}),
    })

