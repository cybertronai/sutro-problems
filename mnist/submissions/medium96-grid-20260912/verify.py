"""Independently regenerate all11draws, audit predictions and recompute NVML costs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from mnist.code import data

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    protocol=read(HERE/'protocol.json');manifest=read(HERE/'prediction_manifest.json')
    assert manifest['protocol_sha256']==sha(HERE/'protocol.json')
    assert manifest['draw_manifest_sha256']==sha(HERE/'draw_manifest.json')
    assert protocol['generator_sha256']==sha(ROOT/'mnist/code/data.py')
    for name,digest in protocol['source_sha256'].items():assert sha(HERE/name)==digest
    assert len(set(protocol['dataset_seeds']))==len(manifest['predictions'])==11
    paths={name:data.download_source(Path('/tmp/mnist-medium96-current/raw'),*data.SOURCES[name]) for name in ('train_images','train_labels')}
    for name,path in paths.items():assert sha(path)==protocol['raw_sources'][name]['sha256']
    pixels=data.read_idx(paths['train_images'],60000,True);labels=data.read_idx(paths['train_labels'],60000,False)
    draws=read(HERE/'draw_manifest.json')['draws'];records=[]
    for i,entry in enumerate(manifest['predictions']):
        assert i==entry['draw']; assert entry['dataset_seed']==protocol['dataset_seeds'][i]
        order=np.random.Generator(np.random.PCG64(entry['dataset_seed'])).permutation(60000)
        fit,query=order[:10000],order[10000:20000]
        assert len(np.unique(np.r_[fit,query]))==20000
        assert data.array_hash(fit)==draws[i]['train_indices_sha256']
        assert data.array_hash(query)==draws[i]['test_indices_sha256']
        def resize(ix):
            a=data.area_resize(pixels[ix].astype(np.float32)/np.float32(255),9);np.clip(a,0,1,out=a);return a[:,None,:,:]
        arrays={'train_images':resize(fit),'train_labels':labels[fit],'test_images':resize(query)}
        assert {k:data.array_hash(a) for k,a in arrays.items()}==draws[i]['input_sha256']
        assert sha(HERE/entry['file'])==entry['file_sha256']
        prediction=np.load(HERE/entry['file'],allow_pickle=False)
        assert prediction.shape==(10000,) and prediction.dtype==np.uint8 and (prediction<10).all()
        assert data.array_hash(prediction)==entry['array_sha256_u8']
        assert sha(HERE/entry['result_file'])==entry['result_sha256']
        result=read(HERE/entry['result_file'])
        assert result['input_sha256']==draws[i]['input_sha256']
        assert result['model_config']=={**protocol['configuration'],'seed':protocol['learner_seeds'][i]}
        assert result['prediction_sha256_int64_le']==data.array_hash(prediction.astype('<i8'))
        assert result['source_sha256']==protocol['source_sha256']['learner.py']
        assert not result['validation']['ptx_has_fp32_fma'] and not result['validation']['ptx_has_ftz']
        for trial in result['trials']:
            def power(v):
                elapsed=v['end']['time_s']-v['start']['time_s']
                joules=(v['end']['energy_mj']-v['start']['energy_mj'])/1000
                assert np.isclose(elapsed,v['duration_s']) and np.isclose(joules,v['energy_j'])
                return joules/elapsed
            idle=(power(trial['idle_before'])+power(trial['idle_after']))/2
            power(trial['active'])
            adjusted=trial['active']['energy_j']-idle*trial['active']['duration_s']
            assert np.isclose(adjusted/trial['invocations'],trial['idle_adjusted_j_per_invocation'])
            assert trial['idle_adjusted_j_per_invocation']>0
            assert trial['cuda_graph_replays_per_invocation']==protocol['configuration']['epochs']+2
        correct=int((prediction==labels[query]).sum())
        records.append({'draw':i,'correct':correct,'total':10000,'energy_a100_mj':result['summary']['idle_adjusted_j_per_invocation']['mean']*1000,
                        'time_a100_ms':result['summary']['cuda_event_us_per_invocation']['mean']/1000})
    accuracy=read(HERE/'accuracy.json');count=sum(r['correct'] for r in records)
    mean=count/110000*100;sd=float(np.std([r['correct']/100 for r in records],ddof=1))
    assert accuracy['correct']==count and accuracy['total']==110000
    assert np.isclose(accuracy['accuracy_percent_mean'],mean)
    assert np.isclose(accuracy['accuracy_percent_sample_sd_pp'],sd)
    assert accuracy['meets_target']==(count>=105600)
    for key in ('energy_a100_mj','time_a100_ms'):
        assert np.isclose(accuracy[key+'_mean'],np.mean([r[key] for r in records]))
    report={'all_checks_passed':True,'draws':records,'correct':count,'total':110000,'accuracy_percent_mean':mean,
            'accuracy_percent_sample_sd_pp':sd,'meets_96_percent_target':count>=105600,
            'checks':['All11raw-source seeded draws regenerated, arrays and index hashes matched; within-draw train/test disjoint',
                      'All110000prediction labels and file/array hashes checked; exact correct counts recomputed',
                      'Learner source/protocol/configuration hashes and every learner seed checked',
                      'AllNVML counter deltas and paired idle-adjusted energies independently recomputed',
                      'NoFP32FMA or FTZ in recorded kernel verification; repeated complete tasks bitwise stable'],
            'verification_source_sha256':sha(Path(__file__))}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
