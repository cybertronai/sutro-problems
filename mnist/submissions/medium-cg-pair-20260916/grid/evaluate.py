"""Execute all frozen grid draws without reading evaluation labels."""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.util, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import spatial_program

HERE=Path(__file__).resolve().parent

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()

def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--executor-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--program',type=Path,required=True)
    p.add_argument('--parity-report',type=Path,required=True)
    p.add_argument('--threads',type=int,default=16)
    args=p.parse_args()
    if np.__version__!='2.1.2':p.error('Use NumPy2.1.2')
    if not args.parity_report.is_file():p.error('Numerical parity report is required')
    protocol=json.loads(args.protocol.read_text())
    if file_hash(HERE/'spatial_program.py')!=protocol['spatial_program_sha256']:
        raise ValueError('Frozen grid source changed')
    records=json.loads((args.inputs_dir/'manifest.json').read_text())
    seeds=list(range(2026091600,2026091611))
    if protocol['dataset_seeds']!=seeds or [r['seed'] for r in records]!=seeds:
        raise ValueError('All11 frozen draws are required')
    source_paths={'spatial_program.py':HERE/'spatial_program.py','evaluate.py':Path(__file__),
                  'executor/executor.py':args.executor_dir/'executor.py',
                  'executor/grid_pair.cpp':args.executor_dir/'grid_pair.cpp',
                  'protocol.json':args.protocol}
    sources={name:file_hash(path) for name,path in source_paths.items()}
    identity={'source_sha256':sources,'protocol':protocol,
              'program_file_sha256':file_hash(args.program),
              'program_canonical_sha256':hashlib.sha256(json.dumps(json.loads(args.program.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'input_manifest_sha256':file_hash(args.inputs_dir/'manifest.json'),
              'parity_report_sha256':file_hash(args.parity_report),
              'library_sha256':file_hash(args.executor_dir/'grid_pair.so'),
              'python':platform.python_version(),'numpy':np.__version__,
              'machine':platform.machine(),'platform':platform.platform(),'threads':args.threads,
              'arithmetic':'FP32RNE; FTZ/DAZ disabled; FMA/reassociation disabled; independent outer work parallel, each reduction retains IL order'}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output_dir/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume identity differs')
    else:write_json(manifest_path,{'created_utc':utc(),'fingerprint':fingerprint,'identity':identity})
    spec=importlib.util.spec_from_file_location('grid_executor',args.executor_dir/'executor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    weights,biases=spatial_program.filters(512,0)
    complete=[]
    for record in records:
        draw=record['draw'];path=args.inputs_dir/record['filename']
        if file_hash(path)!=record['file_sha256']:raise ValueError(f'Input changed draw{draw}')
        row_path=args.output_dir/f'draw_{draw:02d}.json'
        if row_path.exists():
            row=json.loads(row_path.read_text())
            if row['run_fingerprint']!=fingerprint:raise ValueError('Mixed evaluation')
            for item in ('predictions','scores'):
                if file_hash(args.output_dir/row[item]['filename'])!=row[item]['file_sha256']:raise ValueError('Saved artifact changed')
            complete.append(row);continue
        with np.load(path,allow_pickle=False) as data:
            x,y,q=data['x'],data['y'],data['q']
        for key,array in [('train_pixels',x),('train_labels',y),('query_pixels',q)]:
            if array_hash(array)!=record['array_sha256'][key]:raise ValueError(f'Input array mismatch {key}')
        snapshots={}
        def snapshot(name,array):
            snapshots[name]={'shape':list(array.shape),'dtype':str(array.dtype),
                             'sha256':array_hash(array),'all_finite':bool(np.isfinite(array).all())}
        print(f'Draw{draw:02d} seed{record["seed"]}: starting fresh grid execution',flush=True)
        start=time.perf_counter()
        predictions,scores=module.run(x,y,q,weights,biases,iterations=300,threads=args.threads,snapshot=snapshot,library=args.executor_dir/'grid_pair.so')
        elapsed=time.perf_counter()-start
        if predictions.shape!=(10000,) or np.any((predictions<0)|(predictions>9)):raise ValueError('Invalid predictions')
        artifacts={}
        for name,array in [('predictions',predictions),('scores',scores)]:
            dest=args.output_dir/f'draw_{draw:02d}.{name}.npy';np.save(dest,array,allow_pickle=False)
            artifacts[name]={'filename':dest.name,'file_sha256':file_hash(dest),'array_sha256':array_hash(array),'shape':list(array.shape),'dtype':str(array.dtype)}
        row={'draw':draw,'seed':record['seed'],'run_fingerprint':fingerprint,'completed_utc':utc(),
             'input_arrays':record['array_sha256'],'host_execution_seconds':elapsed,
             'all_scores_finite':bool(np.isfinite(scores).all()),'snapshots':snapshots,**artifacts}
        write_json(row_path,row);complete.append(row)
        print(f'Draw{draw:02d}: predictions frozen; host execution {elapsed:.3f}s; finite={row["all_scores_finite"]}',flush=True)
    write_json(args.output_dir/'prediction_freeze.json',{'completed_utc':utc(),'run_fingerprint':fingerprint,'complete':True,'draws':len(complete),'predictions':110000,'test_labels_read':False,'all_scores_finite':all(r['all_scores_finite'] for r in complete),'draw_record_sha256':{f'draw_{r["draw"]:02d}.json':file_hash(args.output_dir/f'draw_{r["draw"]:02d}.json') for r in complete}})

if __name__=='__main__':main()
