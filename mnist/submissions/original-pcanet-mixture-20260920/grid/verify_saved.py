"""Verify published grid evidence without executing the model or grid scorer."""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()
def read(name):return json.loads((HERE/name).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--data-module',type=Path,default=HERE.parent)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.data_module))
    from data import official
    arrays,manifest=official(a.raw)
    checks=[]
    def check(value,message):
        if not value:raise AssertionError(message)
        checks.append(message)
    for path,expected in read('frozen-protocol.json')['sources'].items():
        check(sha((HERE/path).read_bytes())==expected,'frozen source '+path)
    original_bytes=gzip.decompress((HERE/'program.json.gz').read_bytes())
    placed_bytes=gzip.decompress((HERE/'placed-program.json.gz').read_bytes())
    original=json.loads(original_bytes);placed=json.loads(placed_bytes)
    score=read('placed-score.json');execution=read('full-execution/execution.json')
    verified=read('verification.json');build=read('full-execution/build.json')
    check(sha(original_bytes)==execution['program_sha256']==build['program_sha256'],
          'numerical execution bound to original program')
    check(sha(placed_bytes)==score['program_sha256'],'cost bound to placed program')
    check(original['body']==placed['body'],'placement preserves logical instructions')
    check(sorted(original['regions'],key=lambda r:r['name'])==sorted(placed['regions'],key=lambda r:r['name']),
          'placement preserves region names and sizes')
    check(original['metadata']==placed['metadata'],'placement preserves model metadata')
    check(manifest==read('full-execution/data-manifest.json'),'canonical MNIST hashes')
    cfg=original['metadata']['config']
    for key,value in {'N':60000,'Q':10000,'side':28,'L1':8,'L2':5,'kernel':7,
                      'block':14,'stride':7,'blocks':9,'features':2304,'K':100,
                      'components':8,'kmeans_steps':8,'em_steps':8}.items():
        check(cfg[key]==value,'configuration '+key)
    check(cfg['mixture_seed_by_class']==list(range(900,910)),'per-class seeds')
    check(score['input_tape_words']==54940000 and score['output_tape_words']==10000,'complete tape counts')
    check(score['energy_fj']==sum(v['energy_fj'] for v in score['components'].values()),'energy component sum')
    check(score['cycles']==sum(v['cycles'] for v in score['components'].values()),'cycle component sum')
    check(np.isclose(score['energy_mj'],score['energy_fj']/1e12,rtol=1e-15),'energy unit conversion')
    check(score['time_ms']==score['cycles']/1e6,'time unit conversion')
    words=sum(r['words'] for r in placed['regions'])+250
    check(words==score['peak_allocated_scratch_words']<=384000000,'scratch accounting and capacity')
    check(4*words==score['peak_allocated_scratch_bytes'],'scratch bytes')
    check(score['model_spec_commit']=='01a0bd5e0d2564825b0f53dd766f763c82dbc7c0','model revision')
    for region in original['regions']:
        observed=execution['regions'][region['name']]
        check(observed['words']==region['words'] and observed['nonfinite_as_fp32']==0,
              'recorded finite region '+region['name'])
    pred=np.load(HERE/'full-execution/predictions.npy',allow_pickle=False)
    scores=np.load(HERE/'full-execution/class-scores.npy',allow_pickle=False)
    check(pred.shape==(10000,) and pred.dtype==np.uint32 and bool(np.all(pred<10)),'prediction format')
    check(scores.shape==(10000,10) and scores.dtype==np.float32 and bool(np.isfinite(scores).all()),'finite class scores')
    check(sha(scores.tobytes())==execution['regions']['sc']['sha256'],'class scores match executed region hash')
    check(np.array_equal(scores.argmax(1),pred),'score argmax equals saved predictions')
    correct=int(np.count_nonzero(pred==arrays[3]))
    check(correct==verified['correct']==9911,'independent canonical test accuracy')
    check(correct>=9900 and verified['qualifies_99_percent'],'original-tier target')
    check(sha(pred.tobytes())==verified['prediction_sha256'],'prediction hash')
    for key in ['energy_mj','time_ms']:
        check(score[key]==verified[key],'qualification report '+key)
    for name in ['conformance/interpreter-conformance.json','conformance/parallel-conformance.json',
                 'conformance/sparse/parallel-conformance.json','rng-verification.json']:
        check(read(name)['passed'],'retained conformance '+name)
    result={'passed':True,'checks':len(checks),'correct':correct,'total':10000,
            'accuracy_pct':correct/100,'energy_mj':score['energy_mj'],'time_ms':score['time_ms'],
            'peak_scratch_bytes':score['peak_allocated_scratch_bytes'],
            'scope':'Independent saved-prediction accuracy, saved score argmax, program/source bindings, placement identity, arithmetic and retained conformance reports.',
            'limitations':['Does not rerun the learner or recompute affine access histograms.',
                'Finite observations for non-score regions are retained scalar records; the full 1.25 GB memory dump is excluded from the submission. The original full checker verified every region against that dump.']}
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
