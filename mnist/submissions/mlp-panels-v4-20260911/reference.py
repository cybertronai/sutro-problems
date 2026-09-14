"""Verified ordered-FP32 CPU implementation and bounded IL verifier."""
import time
import numpy as np
import panels as p
from validate_mlp_il import SemanticMachine
import accuracy_study as study

def panel_mm(a,b,cfg):
    if cfg is None:
        return study.ordered_mm(a,b)
    # Vectorize across independent tiles, never across K. Padding is a CPU-only
    # verification convenience; generated IL executes exact valid tails.
    mr = cfg['m']
    widths = [6,10] if cfg['n']=='6+10' else [cfg['n']]
    left = np.zeros(((len(a)+mr-1)//mr*mr,a.shape[1]),np.float32)
    left[:len(a)] = a
    result = np.empty((len(a),b.shape[1]),np.float32)
    offset = 0
    wi = 0
    while offset<b.shape[1]:
        nc = min(widths[wi%len(widths)],b.shape[1]-offset)
        right = b[:,offset:offset+nc]
        out = np.zeros((len(left)//mr,mr,nc),np.float32)
        for k in range(a.shape[1]):
            av = left[:,k].reshape(-1,mr,1)
            bv = right[k][None,None,:]
            out = out + av*bv
        result[:,offset:offset+nc] = out.reshape(-1,nc)[:len(a)]
        offset += nc
        wi += 1
    return result

def cpu(data,config,epochs,check_baseline=False):
    cfgs = config['products']
    def mm(name,a,b):
        return panel_mm(a,b,cfgs[name])
    params = study.parameters(32,101)
    x,q = study.transform(data['train_images']),study.transform(data['test_images'])
    target = (data['train_labels'][:,None]==np.arange(10)).astype(np.float32)
    step = np.float32(.2/30)
    reference = study.parameters(32,101)
    epoch_checks = []
    for epoch in range(epochs):
        for first in range(0,len(x),30):
            w1,b1,w2,b2 = params
            xb = x[first:first+30]
            z = mm('hidden',xb,w1)+b1
            h = np.where(z>0,z,np.float32(0))
            d2 = mm('output',h,w2)+b2-target[first:first+30]
            d1 = np.where(z>0,mm('backward',d2,w2.T),np.float32(0))
            params = [w1-step*mm('grad_w1',xb.T,d1),b1-step*study.ordered_rows(d1),
                      w2-step*mm('grad_w2',h.T,d2),b2-step*study.ordered_rows(d2)]
        if check_baseline:
            reference = study.epoch(x,target,reference,.2,study.ordered_mm,study.ordered_rows)
            for actual,expected in zip(params,reference):
                assert actual.tobytes()==expected.tobytes(),('epoch',epoch)
            epoch_checks.append(epoch)
    w1,b1,w2,b2 = params
    scores = []
    ib = config['inference_batch']
    for first in range(0,len(q),ib):
        z = mm('infer_hidden',q[first:first+ib],w1)+b1
        h = np.where(z>0,z,np.float32(0))
        scores.append(mm('infer_output',h,w2)+b2)
    scores = np.concatenate(scores)
    return {'params':np.concatenate([a.ravel() for a in params]), 'scores':scores,
            'predictions':np.argmax(scores,axis=1).astype(np.int64), 'epochs_checked_against_original':epoch_checks}

def execute(config,data,n_train=30,n_test=7,epochs=1,mutated=False):
    inputs = {k:v[:n_train if k.startswith('train') else n_test].copy() for k,v in data.items()}
    if mutated:
        inputs['train_labels'] = (inputs['train_labels']+1)%10
        inputs['test_images'] = np.random.default_rng(7845).uniform(0,1,inputs['test_images'].shape).astype(np.float32)
    document = p.build(config,epochs,n_train,n_test)
    program = p.il.Program(document)
    tape = np.concatenate([inputs['train_images'].ravel().view(np.uint32),inputs['train_labels'].astype(np.uint32),
                           inputs['test_images'].ravel().view(np.uint32)])
    b2,size = program.regions['b2']
    grouped_inference = not (config['inference_batch']==1 and config['products']['infer_hidden'] is None
                            and config['products']['infer_output'] is None and not config['cache_x'])
    score_base = program.regions['d2'][0]
    class Machine(SemanticMachine):
        def step(self,node):
            super().step(node)
            if node[0]=='add' and b2<=node[-1]<b2+size:
                self.scores.append(self.memory[node[1]])
            if node[0]=='send':
                if grouped_inference:
                    row = len(self.score_rows)%config['inference_batch']
                    first = score_base+row*10
                    self.score_rows.append(list(self.memory[first:first+10]))
                else:
                    self.score_rows.append(self.scores[-10:])
    machine = Machine(program.coordinates,tape)
    machine.scores = []
    machine.score_rows = []
    start = time.perf_counter()
    predictions = machine.run(p.il.expand(document)).astype(np.int64)
    elapsed = time.perf_counter()-start
    expected = cpu(inputs,p.default_config(),epochs)
    words = []
    for name in ('w1','b1','w2','b2'):
        offset,length = program.regions[name]
        words.extend(machine.memory[offset:offset+length])
    actual = np.asarray(words,np.uint32).view(np.float32)
    assert actual.tobytes()==expected['params'].tobytes(),'params'
    scores = np.asarray(machine.score_rows,np.uint32).view(np.float32)
    assert scores.tobytes()==expected['scores'].tobytes(),'scores'
    np.testing.assert_array_equal(predictions,expected['predictions'])
    static,reads,writes = p.il.score(document,include_counts=True)
    assert machine.energy_fj==static['energy_fj']
    assert machine.time_ticks==static['time_ticks_0_2_ps']
    assert dict(machine.instructions)==static['instructions']
    np.testing.assert_array_equal(reads,machine.read_counts)
    np.testing.assert_array_equal(writes,machine.write_counts)
    return {'train':n_train,'test':n_test,'epochs':epochs,'mutated':mutated,
            'parameter_bits_and_predictions_equal_original':True,'all_score_bits_equal_original':True,
            'all_address_counts_and_costs_equal':True,
            'instructions':static['total_instructions'],'interpreter_seconds':elapsed,
            'parameter_sha256':study.digest(actual)}
