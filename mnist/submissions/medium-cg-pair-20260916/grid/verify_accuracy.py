"""Independently score frozen spatial-grid predictions against canonical labels."""
import argparse, gzip, hashlib, io, json, statistics
from pathlib import Path
import numpy as np
from mnist.code import data

def artifact_bytes(path):
    path=Path(path)
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes())

def fh(path):return hashlib.sha256(artifact_bytes(path)).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--raw-dir',type=Path,required=True)
    args=parser.parse_args();root=args.results_dir
    freeze=json.loads((root/'prediction_freeze.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    identity=manifest['identity']
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if manifest['fingerprint']!=fingerprint or freeze['run_fingerprint']!=fingerprint:raise ValueError('Fingerprint mismatch')
    if not freeze['complete'] or freeze['draws']!=11 or freeze['predictions']!=110000 or freeze['test_labels_read'] is not False:raise ValueError('Incomplete prediction freeze')
    raw=args.raw_dir/data.SOURCES['train_labels'][0]
    if data.file_hash(raw,'md5')!=data.SOURCES['train_labels'][1]:raise ValueError('MNIST label hash mismatch')
    truth=data.read_idx(raw,60000,False)
    rows=[]
    for draw,seed in enumerate(range(2026091600,2026091611)):
        path=root/f'draw_{draw:02d}.json'
        if fh(path)!=freeze['draw_record_sha256'][path.name]:raise ValueError('Draw record changed afterfreeze')
        doc=json.loads(path.read_text())
        if doc['draw']!=draw or doc['seed']!=seed or doc['run_fingerprint']!=fingerprint:raise ValueError('Draw identity mismatch')
        arrays={}
        for kind in ('predictions','scores'):
            item=doc[kind];path=root/item['filename']
            if fh(path)!=item['file_sha256']:raise ValueError('Artifact hash mismatch')
            a=np.load(io.BytesIO(artifact_bytes(path)),allow_pickle=False)
            if ah(a)!=item['array_sha256'] or list(a.shape)!=item['shape'] or str(a.dtype)!=item['dtype']:raise ValueError('Artifact metadata mismatch')
            arrays[kind]=a
        pred,scores=arrays['predictions'],arrays['scores']
        if pred.shape!=(10000,) or pred.dtype!=np.int64 or scores.shape!=(10000,10) or scores.dtype!=np.float32:raise ValueError('Wrong output shape/dtype')
        if np.any((pred<0)|(pred>9)) or not np.isfinite(scores).all() or not np.array_equal(scores.argmax(1),pred):raise ValueError('Invalid grid scores/predictions')
        if doc['all_scores_finite'] is not True or not all(s['all_finite'] for s in doc['snapshots'].values()):raise ValueError('Nonfinite intermediate')
        order=np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train,query=order[:10000],order[10000:20000]
        if np.intersect1d(train,query).size or ah(train)!=doc['input_arrays']['train_indices'] or ah(query)!=doc['input_arrays']['query_indices'] or ah(truth[query])!=doc['input_arrays']['test_labels']:raise ValueError('Frozen dataset identity mismatch')
        correct=int(np.count_nonzero(pred==truth[query]))
        rows.append({'draw':draw,'seed':seed,'correct':correct,'total':10000,'errors':10000-correct,'host_execution_seconds':doc['host_execution_seconds']})
    total=sum(r['correct'] for r in rows)
    report={'run_fingerprint':fingerprint,'prediction_freeze_sha256':fh(root/'prediction_freeze.json'),
            'independently_verified':True,'all_scores_finite':True,'all_argmax_predictions_verified':True,
            'completed_draws':11,'total_correct':total,'total_predictions':110000,'total_errors':110000-total,
            'mean_accuracy_percent':total/1100,'mean_error_percent':(110000-total)/1100,
            'sample_sd_percentage_points':statistics.stdev(r['correct']/100 for r in rows),
            'minimum_correct_for_2_percent':107800,'meets_2_percent_target':total>=107800,
            'mean_host_execution_seconds':statistics.mean(r['host_execution_seconds'] for r in rows),
            'draws':rows}
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
