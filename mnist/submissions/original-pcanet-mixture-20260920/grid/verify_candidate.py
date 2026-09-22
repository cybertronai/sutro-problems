"""Independent numerical qualification and spatial-score checks.

Run only after execution. Test labels enter here, never the grid input tape.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent


def digest_bytes(data):return hashlib.sha256(data).hexdigest()
def document(path):return json.loads(gzip.decompress(path.read_bytes()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--data-module',type=Path,required=True)
    p.add_argument('--a100-predictions',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.data_module))
    from data import official
    arrays,manifest=official(a.raw)
    train,labels,query,test_labels=arrays
    original=document(HERE/'program.json.gz')
    placed=document(HERE/'placed-program.json.gz')
    score=json.loads((HERE/'placed-score.json').read_text())
    execution=json.loads((HERE/'full-execution/execution.json').read_text())
    provenance=json.loads((HERE/'full-execution/build.json').read_text())
    stored_manifest=json.loads((HERE/'full-execution/data-manifest.json').read_text())
    frozen=json.loads((HERE/'frozen-protocol.json').read_text())
    assertions=[]
    def check(condition,label):
        if not condition:raise AssertionError(label)
        assertions.append(label)
    for path,sha in frozen['sources'].items():
        check(digest_bytes((HERE/path).read_bytes())==sha,'frozen source '+path)
    check(original['body']==placed['body'],'placement preserves every logical instruction')
    check(sorted(original['regions'],key=lambda r:r['name'])==sorted(placed['regions'],key=lambda r:r['name']),
          'placement preserves all logical regions and sizes')
    check(original['metadata']==placed['metadata'],'placement preserves metadata')
    check(score['program_sha256']==digest_bytes(gzip.decompress((HERE/'placed-program.json.gz').read_bytes())),
          'score bound to placed program')
    expected=digest_bytes(gzip.decompress((HERE/'program.json.gz').read_bytes()))
    check(execution['program_sha256']==expected==provenance['program_sha256'],'execution bound to original program')
    check(stored_manifest==manifest,'canonical dataset manifests')
    tape=np.load(HERE/'full-execution/input.npy',mmap_mode='r',allow_pickle=False)
    offset=0
    for part in [train.reshape(-1).view(np.uint32),query.reshape(-1).view(np.uint32),labels.astype(np.uint32)]:
        check(np.array_equal(tape[offset:offset+len(part)],part),'exact input tape segment at '+str(offset))
        offset+=len(part)
    check(offset==len(tape)==score['input_tape_words']==54940000,'exact input count; no test labels')
    predictions=np.load(HERE/'full-execution/predictions.npy',allow_pickle=False)
    check(predictions.dtype==np.uint32 and predictions.shape==(10000,),'prediction shape and dtype')
    check(bool(np.all(predictions<10)),'prediction digits')
    check(score['output_tape_words']==10000,'exact output count')
    check(score['energy_fj']==sum(v['energy_fj'] for v in score['components'].values()),'energy component sum')
    check(score['cycles']==sum(v['cycles'] for v in score['components'].values()),'cycle component sum')
    check(score['peak_allocated_scratch_words']==sum(r['words'] for r in placed['regions'])+250,'all scratch and tape stages charged')
    check(score['peak_allocated_scratch_words']<=384000000,'scratch capacity')
    memory=np.load(HERE/'full-execution/memory.npy',mmap_mode='r',allow_pickle=False)
    base=1;region_checks={};scores=None
    for region in original['regions']:
        name,size=region['name'],region['words'];data=memory[base:base+size]
        observed=execution['regions'][name]
        check(observed['sha256']==digest_bytes(data.tobytes()),'numeric region hash '+name)
        nonfinite=int((~np.isfinite(data.view(np.float32))).sum())
        check(nonfinite==observed['nonfinite_as_fp32']==0,'finite numeric region '+name)
        region_checks[name]=size
        if name=='sc':scores=data.view(np.float32).reshape(10000,10)
        if name=='out':check(np.array_equal(data,predictions),'output region matches output tape')
        base+=size
    check(scores is not None and np.array_equal(scores.argmax(1),predictions),'class-score argmax matches predictions')
    correct=int(np.count_nonzero(predictions==test_labels))
    report={'checks_passed':len(assertions),'assertions':assertions,'numerically_finite':True,
            'correct':correct,'total':10000,'accuracy_pct':correct/100,
            'qualifies_99_percent':correct>=9900,'prediction_sha256':digest_bytes(predictions.tobytes()),
            'original_program_sha256':expected,'placed_program_sha256':score['program_sha256'],
            'energy_mj':score['energy_mj'],'time_ms':score['time_ms'],
            'peak_scratch_bytes':score['peak_allocated_scratch_bytes'],
            'numeric_execution_seconds':execution['elapsed_seconds'],
            'limitations':original['metadata']['differences_from_A100']}
    if a.a100_predictions:
        a100=np.load(a.a100_predictions,allow_pickle=False)
        report['a100_prediction_matches']=int(np.count_nonzero(a100==predictions))
        report['a100_correct']=int(np.count_nonzero(a100==test_labels))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('assertions','limitations')}),flush=True)


if __name__=='__main__':main()
