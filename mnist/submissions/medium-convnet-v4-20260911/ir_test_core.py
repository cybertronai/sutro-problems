"""Adversarial validity checks, integer convolution regression and geometry audit."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ir_core import Program,score,make_program,ref as R,ins as I,loop as L,FORMAT
from ir_tables import packed_table,lookup

HERE=Path(__file__).resolve().parent

def rejected(document,text):
    try:Program(document)
    except ValueError as error:
        assert text in str(error),(text,str(error));return
    raise AssertionError('Malformed program accepted: '+text)

def checks():
    # Prefix initialization never excuses an earlier read, nor a missing word.
    rejected(make_program([('x',2)],[I('copy',R('x'),R('x',1))]),'Uninitialized')
    rejected(make_program([('x',2)],[L('i',1,[I('set',R('x',i=1),0)]),I('send',R('x',1))]),'Uninitialized')
    rejected(make_program([('x',2)],[L('i',2,[I('set',R('x',i=1),0)]),I('send',R('x',2))]),'escapes')
    doc=make_program([('x',2)],[I('set',R('x'),0),I('send',{**R('x'),'lookup':lookup('index')})])
    doc['tables']={'index':packed_table([1])};rejected(doc,'Uninitialized')
    malformed=copy.deepcopy(doc);malformed['body'][1]['src'][0]['lookup']['offset']=1
    rejected(malformed,'Literal index outside')
    malformed=copy.deepcopy(doc);malformed['tables']['index']['sha256']='0'*64
    rejected(malformed,'Literal table identity')
    malformed=copy.deepcopy(doc);malformed['tables']['index']['length']=2
    rejected(malformed,'Literal table identity')
    # Compare arbitrary positive/negative-stride histograms to Cartesian sums.
    rng=np.random.default_rng(731);words=211
    for _ in range(150):
        counts=rng.integers(1,7,3).tolist();coefficients=rng.integers(-7,8,3).tolist()
        scope=tuple((v,(int(rng.integers(-2,3)),count)) for v,count in zip(('i','j','k'),counts))
        coefficients=dict(zip(('i','j','k'),coefficients));operand=R('x',100,**coefficients)
        parsed=Program(make_program([('x',words)],[L('init',words,[I('set',R('x',init=1),0)])]))
        base,hist=parsed.histogram(operand,scope)
        actual=np.zeros(words+1,np.int64);actual[base:base+len(hist)]=hist
        expected=np.zeros(words+1,np.int64)
        for values in itertools.product(*(range(start,start+count) for _,(start,count) in scope)):
            expected[parsed.address(operand,dict(zip(('i','j','k'),values)))]+=1
        np.testing.assert_array_equal(actual,expected)
    # Shell boundaries and every access floor checked by direct integer geometry.
    words=1_000_001
    doc=make_program([('x',words)],[L('i',words,[I('set',R('x',i=1),0)]),I('send',R('x',words-1))])
    result,reads,writes=score(doc,include_counts=True)
    expected_energy=expected_ticks=0
    for address in range(1,words+1):
        h=math.isqrt(address-1)+1
        x=address-(h-1)*(h-1)-h;y=h-abs(x)
        assert -16000<=x<=16000 and 1<=y<=16000
        expected_energy+=max(50,2*(abs(x)+y));expected_ticks+=max(250,2*(abs(x)+y))
    assert result['energy_fj']==expected_energy and result['time_ticks_0_2_ps']==expected_ticks
    assert result['peak_initialized_scratch_words']==words
    # Historical complete 1NN exact totals must survive generic engine changes.
    # Resolve from mnist/ so this regression works from submissions/ as well as
    # the original experiments/ directory, independently of the caller's cwd.
    historical=HERE.parents[1]/'experiments/accuracy-il-20260911'
    baseline=json.loads((historical/'1nn.il.json').read_text());baseline['format']=FORMAT
    result=score(baseline);frozen=json.loads((historical/'1nn.il.score.json').read_text())
    keys=['total_instructions','instructions','charged_reads','charged_writes','energy_fj','time_ticks_0_2_ps','area_um2_occupied_cells']
    assert all(result[key]==frozen[key] for key in keys)
    return {'all_passed':True,'rejected_invalid_programs':7,'cartesian_histogram_cases':150,
        'independently_priced_initialized_words':words,'historical_1nn_exact_regression':{key:result[key] for key in keys},
        'scope':'Validity rejection tests; affine counts versus Cartesian enumeration; >1M-word interval initialization proof and independent physical prices; complete historical 1NN exact totals.'}

if __name__=='__main__':
    result=checks()
    paths=[Path(__file__),HERE/'ir_core.py',HERE/'ir_tables.py']
    result['source_sha256']={str(p.relative_to(HERE.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (HERE/'ir-core-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
