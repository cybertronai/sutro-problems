"""Expand BN primitives and compare directly with the independently owned oracle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from ir_core import Program,score,expand,make_program,ref as R,ins as I,loop as L
from ir_machine import SemanticMachine
import ir_bn as emit

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_bn',HERE/'ordered_backend/bn_ref.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

def case(n,c,h,w):
    rng=np.random.default_rng(72+n+h);shape=(n,c,h,w);size=int(np.prod(shape))
    arrays={'raw':rng.normal(size=shape).astype(np.float32),'gamma':rng.normal(size=c).astype(np.float32),
        'beta':rng.normal(size=c).astype(np.float32),'rm':rng.normal(size=c).astype(np.float32),
        'rv':rng.random(c,dtype=np.float32),'dy':rng.normal(size=shape).astype(np.float32)}
    regions=[('s',8),('k',24)]+[(name,a.size) for name,a in arrays.items()]
    regions += [(name,c) for name in ('mean','var','inv','dg','db')]+[(name,size) for name in ('xhat','out','dx','inferred','inferred_xhat')]
    body=[L('zero',words,[I('set',R(name,zero=1),0)]) for name,words in regions]
    for j,value in {2:.9,4:1,6:.5,20:1e-5,21:1/(n*h*w),22:(n*h*w)/(n*h*w-1),23:.1}.items():
        body.append(I('set',R('k',j),int(np.float32(value).view(np.uint32))))
    for name,a in arrays.items():body.append(L('read',a.size,[I('recv',R(name,read=1))]))
    body+=emit.forward('raw','gamma','beta','mean','var','inv','xhat','out','rm','rv',n,c,h,w,'train',True)
    body+=emit.backward('dy','xhat','gamma','inv','dg','db','dx',n,c,h,w,'back')
    body+=emit.forward('raw','gamma','beta','mean','var','inv','inferred_xhat','inferred','rm','rv',n,c,h,w,'infer',False)
    doc=make_program(regions,body);program=Program(doc)
    machine=SemanticMachine(program.coordinates,np.concatenate([a.ravel().view(np.uint32) for a in arrays.values()]))
    machine.run(expand(doc))
    out,xhat,mean,inverse,rm,rv=reference.train(arrays['raw'],arrays['gamma'],arrays['beta'],arrays['rm'],arrays['rv'])
    dx,dg,db=reference.backward(arrays['dy'],xhat,arrays['gamma'],inverse)
    inferred,ixhat,iinverse=reference.infer(arrays['raw'],arrays['gamma'],arrays['beta'],rm,rv)
    expected={'out':out,'xhat':xhat,'dx':dx,'dg':dg,'db':db,'rm':rm,'rv':rv,'inferred':inferred,'inferred_xhat':ixhat,'inv':iinverse}
    for name,a in expected.items():
        first,words=program.regions[name]
        np.testing.assert_array_equal(np.asarray(machine.memory[first:first+words],np.uint32),a.ravel().view(np.uint32),err_msg=name)
    result,reads,writes=score(doc,include_counts=True)
    np.testing.assert_array_equal(reads,machine.read_counts);np.testing.assert_array_equal(writes,machine.write_counts)
    assert result['energy_fj']==machine.energy_fj and result['time_ticks_0_2_ps']==machine.time_ticks
    return {'shape':list(shape),'all_training_backward_inference_running_state_bits_match':True,
        'every_address_and_physical_score_matches':True,'expanded_instructions':result['total_instructions']}

if __name__=='__main__':
    result={'all_passed':True,'cases':[case(2,3,3,4),case(2,2,9,9)],
        'source_sha256':{str(p.relative_to(HERE.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__),HERE/'ir_bn.py',HERE/'ir_core.py',HERE/'ir_machine.py',HERE/'ordered_backend/bn_ref.py']}}
    (HERE/'ir-bn-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
