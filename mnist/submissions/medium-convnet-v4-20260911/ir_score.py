#!/usr/bin/env python3
"""Build and exactly score a complete fixed CNN using retained seed-only constants.

The constants directory is produced by ordered_backend/export_constants.py with
--include-initial. This command reads no MNIST arrays, labels or learned weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import numpy as np
from ir_model import build
from ir_core import score

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value,dtype='<f4').tobytes()).hexdigest()

def load_constants(config,path,regenerate=False):
    manifest=json.loads(path.read_text())
    if manifest['seeds']!=config['seeds'] or manifest['n_train']!=config['n_train']:raise ValueError('Constants task dimensions/seeds mismatch')
    for key,value in manifest['config'].items():
        if config.get(key)!=value:raise ValueError('Constants configuration differs at '+key)
    for name,digest in manifest['source_sha256'].items():
        if name not in ('schedule.py','export_constants.py'):raise ValueError('Unexpected constants source')
        if sha(HERE/'ordered_backend'/name)!=digest:raise ValueError('Constants generator source changed: '+name)
    initial=[];schedules=[]
    for seed,record in zip(config['seeds'],manifest['records'],strict=True):
        if record['seed']!=seed:raise ValueError('Constants record order')
        filename=record['initial']['path']
        if Path(filename).name!=filename:raise ValueError('Initial file must be adjacent to manifest')
        file=path.parent/filename
        if sha(file)!=record['initial']['file_sha256']:raise ValueError('Initial file bytes changed')
        with np.load(file,allow_pickle=False) as values:arrays={name:values[name].copy() for name in values.files}
        if set(arrays)!=set(record['initial']['arrays']):raise ValueError('Initial member names differ')
        for name,value in arrays.items():
            expected=record['initial']['arrays'][name]
            if value.dtype!=np.float32 or list(value.shape)!=expected['shape'] or str(value.dtype)!=expected['dtype'] or array_sha(value)!=expected['sha256']:
                raise ValueError('Initial array differs: '+name)
        if regenerate:
            import importlib.util
            spec=importlib.util.spec_from_file_location('seed_reproduction',HERE/'ordered_backend/schedule.py')
            generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
            fresh=generator.initial_parameters(config,seed)
            for name,value in arrays.items():np.testing.assert_array_equal(value.view(np.uint32),fresh[name].view(np.uint32))
        initial.append(arrays);schedules.append(record['epochs'])
    return initial,schedules,manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--constants',type=Path,required=True,help='Retained seed-only constants.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--regenerate-initial',action='store_true',help='Requires the matching CPU PyTorch; independently reruns initialization from each seed')
    args=parser.parse_args()
    if not 1<=args.repeats<=10:parser.error('--repeats must be 1..10')
    config=json.loads(args.config.read_text())
    sources=sorted(p for p in HERE.glob('ir_*.py') if not p.name.startswith('ir_test') and p.name not in ('ir_bn_reference.py','ir_machine.py'))
    sources += [HERE/'ordered_backend/schedule.py',HERE/'ordered_backend/export_constants.py']
    before={str(p.relative_to(HERE.parents[2])):sha(p) for p in sources}
    initial,manifests,constants=load_constants(config,args.constants,args.regenerate_initial)
    document=build(config,initial,manifests)
    results=[score(document) for _ in range(args.repeats)]
    stable=[{k:v for k,v in row.items() if k!='time_to_score_seconds'} for row in results]
    if not all(row==stable[0] for row in stable):raise AssertionError('Repeated exact scores differ')
    result=results[0];times=[row['time_to_score_seconds'] for row in results]
    result.update(time_to_score_seconds=statistics.median(times),time_to_score_samples_seconds=times,
        time_ms=result['time_ticks_0_2_ps']/5/1e9,energy_mj=result['energy_fj']/1e12,
        area_mm2=result['area_um2_occupied_cells']/1e6,
        config=config,constants_manifest_sha256=sha(args.constants),source_sha256=before,
        all_regenerated_schedule_bytes_match_pinned_gpu_manifests=True,
        initial_seed_generation_reexecuted_here=args.regenerate_initial,
        initial_scope='All raw initial parameter words embedded and checked against retained seed-only exporter records; --regenerate-initial additionally reruns the pinned PyTorch generation recipe.',
        software={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
        numerical_validation_scope='Exact scores are static and input-independent. Separate retained expanded-IL/CPU/GPU numerical tests establish arithmetic equivalence; this command does not retrain or evaluate classification accuracy.')
    if before!={str(p.relative_to(HERE.parents[2])):sha(p) for p in sources}:raise AssertionError('Source changed during scoring')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'program.il.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    (args.output/'model-score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('program_sha256','total_instructions','time_ms','energy_mj','area_mm2','time_to_score_seconds')},indent=2))

if __name__=='__main__':main()
