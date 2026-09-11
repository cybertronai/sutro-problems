"""Execute small expanded MLP IL programs and compare every learned FP32 bit.

The comparison uses explicit ordered-reduction training, separate from the IL
lowering and static histogram scorer, and the previously reviewed v4 machine.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import il
import accuracy_study as study
from mlp_il import build_mlp

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('old_v4',HERE.parents[1]/'submissions/1nn-v4-20260911/score_v4.py')
old=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(old)


class SemanticMachine(old.Machine):
    def step(self,instruction):
        if instruction[0].startswith('cmp_'):
            opcode,d,a,b=instruction
            left,right=old.bits_to_float(self.memory[a]),old.bits_to_float(self.memory[b])
            # The existing machine performs the same charged accesses/checks;
            # the alternate predicate changes only the boolean result.
            super().step(('cmp',d,a,b))
            result={'eq':left==right,'ne':left!=right,'le':left<=right,
                    'gt':left>right,'ge':left>=right}[opcode[4:]]
            self.memory[d]=int(result)
        else:super().step(instruction)


def validate_case(data,width,train_count,batch,test_count,epochs,seed=101,rate=.2):
    document=build_mlp(width,epochs,rate,seed,train_count,test_count,batch)
    parsed=il.Program(document)
    train=data['train_images'][:train_count]
    labels=data['train_labels'][:train_count]
    queries=data['test_images'][:test_count]
    tape=np.concatenate([train.reshape(-1).view(np.uint32),labels.astype(np.uint32),
                         queries.reshape(-1).view(np.uint32)])
    t0=time.perf_counter()
    machine=SemanticMachine(parsed.coordinates,tape)
    observed=machine.run(il.expand(document)).astype(np.int64)
    execution_seconds=time.perf_counter()-t0
    x,q=study.transform(train),study.transform(queries)
    target=(labels[:,None]==np.arange(10)).astype(np.float32)
    expected_params=study.parameters(width,seed)
    for _ in range(epochs):
        for start in range(0,train_count,batch):
            expected_params=study.update(x[start:start+batch],target[start:start+batch],
                expected_params,rate,study.ordered_mm,study.ordered_rows)
    parameter_checks={}
    for name,expected in zip(['w1','b1','w2','b2'],expected_params):
        start,length=parsed.regions[name]
        actual=np.asarray(machine.memory[start:start+length],dtype=np.uint32).view(np.float32).reshape(expected.shape)
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),name
        parameter_checks[name]=study.digest(actual)
    expected=study.predict(q,expected_params,study.ordered_mm)
    assert np.array_equal(observed,expected),'predictions'
    static=il.score(document)
    assert machine.energy_fj==static['energy_fj'],'energy'
    assert machine.time_ticks==static['time_ticks_0_2_ps'],'time'
    assert dict(machine.instructions)==static['instructions'],'opcode counts'
    assert sum(machine.read_counts)==static['charged_reads'],'read counts'
    assert sum(machine.write_counts)==static['charged_writes'],'write counts'
    return {'width':width,'train_count':train_count,'batch':batch,'test_count':test_count,
            'epochs':epochs,'all_parameter_bits_match':True,'all_predictions_match':True,
            'static_and_expanded_costs_counts_match':True,'parameter_sha256':parameter_checks,
            'expanded_instructions':static['total_instructions'],
            'reference_execution_seconds':execution_seconds,
            'static_score_seconds':static['time_to_score_seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('mnist/data/small.npz'))
    p.add_argument('--output',type=Path,default=HERE/'mlp_validation.json')
    a=p.parse_args()
    with np.load(a.data,allow_pickle=False) as z:
        data={key:z[key] for key in ('train_images','train_labels','test_images')}
    cases=[validate_case(data,4,6,3,7,3),validate_case(data,16,30,30,17,1)]
    result={'checks':'Complete expanded IL training/inference versus explicit ordered FP32 reference',
            'cases':cases,'all_passed':True,
            'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ['il.py','mlp_il.py','accuracy_study.py','validate_mlp_il.py']}}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
