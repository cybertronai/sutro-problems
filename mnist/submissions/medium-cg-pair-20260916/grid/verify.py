"""Verify packaged grid accuracy, source bindings, inputs, and score arithmetic."""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from mnist.code import data

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def check(condition,message):
    if not condition:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True)
    args=parser.parse_args()
    check(np.__version__=='2.1.2','Use NumPy2.1.2 and OPENBLAS_CORETYPE=Haswell')
    evidence=HERE/'evidence'
    manifest=json.loads((evidence/'run_manifest.json').read_text())
    for name,expected in manifest['identity']['source_sha256'].items():
        check(sha(HERE/name)==expected,f'Frozen grid source changed: {name}')
    program_bytes=gzip.decompress((evidence/'program.json.gz').read_bytes())
    check(hashlib.sha256(program_bytes).hexdigest()==manifest['identity']['program_file_sha256'],'Program file differs')
    program=json.loads(program_bytes)
    canonical=hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check(canonical==manifest['identity']['program_canonical_sha256'],'Canonical program differs')
    check(sha(evidence/'input_manifest.json')==manifest['identity']['input_manifest_sha256'],'Input manifest changed')
    check(sha(evidence/'conformance/parity.json')==manifest['identity']['parity_report_sha256'],'Initial parity report changed')
    inputs=json.loads((evidence/'input_manifest.json').read_text())
    with gzip.open(HERE.parent/'evidence/a100/croatia.json.gz','rt') as stream:a100=json.load(stream)
    check(sha(ROOT/'mnist/code/data.py')==a100['dataset']['data_helper_sha256'],'Canonical data helper changed')
    for kind in ('train_images','train_labels'):
        path=args.raw_dir/data.SOURCES[kind][0]
        check(data.file_hash(path,'md5')==data.SOURCES[kind][1],f'Canonical {kind} changed')
        check(sha(path)==a100['dataset']['source_sha256'][kind],f'Raw {kind} SHA differs')
    pixels=data.read_idx(args.raw_dir/data.SOURCES['train_images'][0],60000,True)
    labels=data.read_idx(args.raw_dir/data.SOURCES['train_labels'][0],60000,False)
    for index,record in enumerate(inputs):
        check(record['draw']==index and record['seed']==2026091600+index,'Unexpected grid draw')
        order=np.random.Generator(np.random.PCG64(record['seed'])).permutation(60000)
        train,query=order[:10000],order[10000:20000]
        resize=lambda rows:data.area_resize(pixels[rows].astype(np.float32)/np.float32(255),9).reshape(10000,81)
        arrays={'train_indices':train,'query_indices':query,'train_pixels':resize(train),'train_labels':labels[train],'query_pixels':resize(query),'test_labels':labels[query]}
        check({key:ah(value) for key,value in arrays.items()}==record['array_sha256'],f'Input draw{index} differs; pin OPENBLAS_CORETYPE=Haswell')
        check(record['array_sha256']==a100['qualification']['draws'][index]['dataset']['array_sha256'],'Grid and A100 inputs differ')
    check(len(inputs)==11,'Expected all11 datasets')
    accuracy=json.loads(subprocess.check_output([sys.executable,str(HERE/'verify_accuracy.py'),'--results-dir',str(evidence),'--raw-dir',str(args.raw_dir)],text=True))
    check(accuracy==json.loads((evidence/'accuracy.json').read_text()),'Saved accuracy report differs')
    score=json.loads((evidence/'grid-score.json').read_text())
    check(score['program_sha256']==canonical and score['configuration']==program['metadata'],'Score belongs to another program')
    shared=HERE.parent.parent/'grid-mlp-scoring-20260912'
    for name,expected in score['reproduction']['source_sha256'].items():
        path=HERE/'scorer'/name if name=='score_program.py' else shared/name
        check(sha(path)==expected,f'Scoring source changed: {name}')
    instructions=collections.Counter()
    def count(body,multiplicity=1):
        for node in body:
            if 'loop' in node:count(node['body'],multiplicity*node['count'])
            else:instructions[node['op']]+=multiplicity
    count(program['body'])
    check(dict(instructions)==score['logical_instructions'],'Instruction counts differ')
    extra=instructions['recv']+instructions['send']
    executed=dict(instructions);executed['copy']=executed.get('copy',0)+extra
    check(executed==score['executed_instructions'] and sum(executed.values())==score['total_executed_instructions'],'Expanded instruction counts differ')
    words=sum(region['words'] for region in program['regions'])
    check(words==score['program_scratch_words'] and words+250==score['peak_allocated_scratch_words'],'Scratch counts differ')
    check(4*(words+250)==score['peak_allocated_scratch_bytes'] and words+250<=384000000,'Scratch capacity/bytes differ')
    check(instructions['recv']==score['input_tape_words']==1630000 and instructions['send']==score['output_tape_words']==10000,'Tape scope differs')
    for kind,total in [('input',1630000),('output',10000)]:
        check(score['per_port_words'][kind]==[total//250+(port<total%250) for port in range(250)],'Tape port allocation differs')
    check(sum(row['energy_fj'] for row in score['components'].values())==score['energy_fj'],'Energy component sum differs')
    check(sum(row['cycles'] for row in score['components'].values())==score['cycles'],'Cycle component sum differs')
    check(score['energy_mj']==score['energy_fj']/1e12 and score['time_ms']==score['cycles']/1e6,'Model unit conversion differs')
    check(score['word_node_hops']==score['energy_fj'],'Movement energy differs')
    check(score['max_simultaneous_instructions']==score['max_simultaneous_outstanding_accesses']==1,'Unexpected schedule concurrency')
    for name in ('parity','parity_medium','parity_wide'):
        doc=json.loads((evidence/f'conformance/{name}.json').read_text())
        check(doc['all_compared_regions_bitwise_equal'] is True,f'Numerical parity failed: {name}')
    for name in ('tiny','edges','grid'):
        check(json.loads((evidence/f'conformance/interpreter-{name}.json').read_text())['pass'] is True,f'Interpreter parity failed: {name}')
    full=json.loads((evidence/'conformance/full-draw.json').read_text())
    check(full['all_compared_regions_bitwise_equal'] and full['all_compared_float_regions_finite'],'Full-size numerical parity failed')
    check(full['program_canonical_sha256']==canonical and full['program_file_sha256']==manifest['identity']['program_file_sha256'],'Full-size conformance program differs')
    check(full['reference_run_fingerprint']==manifest['fingerprint'] and full['reference_record_sha256']==sha(evidence/'draw_00.json'),'Full-size conformance reference differs')
    check(full['compiler_source_sha256']==sha(HERE/'il_executor/compiler.py') and full['verifier_source_sha256']==sha(HERE/'il_executor/verify_full_draw.py'),'Independent executor source differs')
    check(full['input_file_sha256']==inputs[0]['file_sha256'],'Independent executor input differs')
    check(all(full['shape'][key]==value for key,value in {'N':10000,'Q':10000,'F':512,'T':300}.items()),'Independent executor was not full-size')
    reference=json.loads((evidence/'draw_00.json').read_text())
    check(len(full['regions'])==26 and full['regions']['out']['different_words']==0 and full['regions']['out']['words']==10000,'Missing full-size comparisons')
    for name,item in full['regions'].items():
        check(item['sha256_equal'] is True,'Independent executor hash mismatch')
        if name=='out':
            check(item['sha256_int64']==reference['predictions']['array_sha256'],'Independent predictions differ')
        else:
            source_name='sc2_normalized' if name=='sc2' else name
            check(item['sha256']==reference['snapshots'][source_name]['sha256'] and item['all_finite'],'Independent numerical region differs')
    score_audit=json.loads((evidence/'conformance/full-score-audit.json').read_text())
    check(score_audit['passed'] and score_audit['full_score_sha256']==sha(evidence/'grid-score.json'),'Score audit differs')
    report={'verified':True,'program_sha256':canonical,'frozen_source_hashes_verified':True,
            'all11_input_datasets_match_a100':True,'accuracy_and_argmax_verified':True,
            'score_source_hashes_and_arithmetic_verified':True,'reduced_numerical_parity_verified':True,
            'full_size_numerical_parity_verified':True,
            'total_correct':accuracy['total_correct'],'total_predictions':110000,
            'meets_2_percent_target':accuracy['meets_2_percent_target'],
            'grid_energy_mj':score['energy_mj'],'grid_time_ms':score['time_ms'],
            'grid_peak_scratch_bytes':score['peak_allocated_scratch_bytes']}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
