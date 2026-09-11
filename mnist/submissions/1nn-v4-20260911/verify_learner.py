"""Independently verify the submitted learner and canonical dataset integrity.

Run from the repository root. Only the three allowed arrays are supplied to
learner.predict; the all-six identity audit is separate from model evaluation.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=Path('mnist/data/small.npz'))
parser.add_argument('--manifest', type=Path, default=Path('mnist/doc/dataset_manifest.json'))
parser.add_argument('--output', type=Path, help='Output directory; defaults to a new temporary directory')
args = parser.parse_args()
args.data = args.data.resolve()
args.manifest = args.manifest.resolve()
output = args.output or Path(tempfile.mkdtemp(prefix='mnist-learner-verification-'))
output.mkdir(parents=True, exist_ok=True)
SUB=Path(__file__).resolve().parent
SOURCE=SUB/'learner.py'
spec=importlib.util.spec_from_file_location('submitted_learner',SOURCE)
learner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(learner)
checks={}

def fixture(train,labels,queries):
    return learner.predict(np.asarray(train,dtype=np.float32), np.asarray(labels,dtype=np.int64), np.asarray(queries,dtype=np.float32))

pred,idx=fixture([[0.,0.],[2.,0.]],[8,3],[[1.,0.]])
assert idx.tolist()==[0] and pred.tolist()==[8]
checks['equal_distance_uses_first_training_row']=True
pred,idx=fixture([[1.,1.],[1.,1.],[0.,0.]],[7,2,9],[[1.,1.]])
assert idx.tolist()==[0] and pred.tolist()==[7]
checks['duplicate_images_keep_first_label']=True
pred,idx=fixture([[0.,0.]],[6],[[1.,1.],[0.,0.]])
assert idx.tolist()==[0,0] and pred.tolist()==[6,6]
checks['one_training_example']=True
for tr,q in [([[np.nan]],[[0.]]),([[0.]],[[np.inf]])]:
    try: fixture(tr,[0],q)
    except ValueError: pass
    else: raise AssertionError('Nonfinite data accepted')
checks['nonfinite_input_rejected']=True
with np.load(args.data,allow_pickle=False) as z:
    arrays={k:z[k] for k in ('train_images','train_labels','test_images')}
original={k:v.copy() for k,v in arrays.items()}
pred,idx=learner.predict(**arrays)
assert all(np.array_equal(v,original[k]) for k,v in arrays.items())
checks['inputs_unchanged']=True
saved=np.load(SUB/'predictions.npy',allow_pickle=False)
assert np.array_equal(pred,saved)
checks['canonical_predictions_match_saved']=True
# Independent scalar oracle, preserving explicit float32 rounding after each primitive.
# Check all 600 training candidates for 12 deterministic, spread-out test rows.
qrows=[0,1,2,10,59,100,199,299,399,499,598,599]
x=arrays['train_images'].reshape(600,9)
q=arrays['test_images'].reshape(600,9)
for j in qrows:
    best=np.float32(np.inf)
    selected=-1
    for i in range(600):
        distance=np.float32(0)
        for k in range(9):
            delta=np.float32(q[j,k]-x[i,k])
            distance=np.float32(distance+np.float32(delta*delta))
        if distance<best:
            best=distance
            selected=i
    assert idx[j]==selected
checks['scalar_float32_oracle_12_queries_all_training_rows']=True
mutated_labels=(arrays['train_labels']+1)%10
p2,i2=learner.predict(arrays['train_images'],mutated_labels,arrays['test_images'])
assert np.array_equal(i2,idx) and np.array_equal(p2,(pred+1)%10)
checks['predictions_depend_on_supplied_training_labels']=True
metadata=json.loads((SUB/'cpu_results.json').read_text())
assert learner.array_hash(pred)==metadata['predictions_sha256_int64_le']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==metadata['source_sha256']
checks['saved_source_and_prediction_hashes_match']=True
v=metadata['training_only_validation']
train=np.asarray(v['training_rows']); valid=np.asarray(v['validation_rows'])
assert len(train)==480 and len(valid)==120 and len(set(train)&set(valid))==0
assert sorted(np.concatenate((train,valid)).tolist())==list(range(600))
vp,_=learner.predict(arrays['train_images'][train],arrays['train_labels'][train],arrays['train_images'][valid])
assert int((vp==arrays['train_labels'][valid]).sum())==v['correct']==63
checks['training_validation_partition_and_score']=True
# Run the actual CLI on an archive with no test_labels or index arrays at all.
with tempfile.TemporaryDirectory(prefix='mnist-learner-review-') as td:
    td=Path(td)
    stripped=td/'allowed-only.npz'
    np.savez_compressed(stripped,**arrays)
    cmd=[sys.executable,str(SOURCE),'--data',str(stripped),'--output',str(td/'valid'),'--manifest',str(args.manifest)]
    run=subprocess.run(cmd,cwd=Path.cwd(),capture_output=True,text=True,check=True)
    assert np.array_equal(np.load(td/'valid/predictions.npy',allow_pickle=False),pred)
    checks['full_cli_succeeds_without_test_labels_or_indices']=True
    tampered={k:v.copy() for k,v in arrays.items()}
    tampered['train_images'].flat[0]=np.nextafter(tampered['train_images'].flat[0],np.float32(1))
    np.savez_compressed(td/'tampered.npz',**tampered)
    cmd[cmd.index('--data')+1]=str(td/'tampered.npz')
    cmd[cmd.index('--output')+1]=str(td/'invalid')
    run=subprocess.run(cmd,cwd=Path.cwd(),capture_output=True,text=True)
    assert run.returncode!=0 and 'Noncanonical data: train_images' in run.stderr
    assert not (td/'invalid/predictions.npy').exists()
    checks['single_ulp_input_tamper_rejected_before_prediction']=True

# Separate integrity audit: hashes only; test labels are not passed to learner.
manifest_path=args.manifest
manifest=json.loads(manifest_path.read_text())
with np.load(args.data,allow_pickle=False) as z:
    data_checks={}
    for key,expected in manifest['tiers']['small']['arrays'].items():
        array=z[key]
        actual={'shape':list(array.shape),'dtype':str(array.dtype),'sha256_c_order_little_endian':learner.array_hash(array)}
        data_checks[key]={'actual':actual,'matches_canonical':actual==expected}
    overlap=len(set(z['train_indices'])&set(z['test_indices']))
assert all(v['matches_canonical'] for v in data_checks.values()) and overlap==0
check_result={'audit_type':'independent dataset identity check; not learner input',
 'source_baseline':'e70f9c9e1db65b62d9256f7b1f9b668cf4c48909',
 'profile':manifest['profile'],'seed':manifest['seed'],
 'manifest_file_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
 'train_test_source_index_overlap_count':overlap,'arrays':data_checks,
 'all_six_match':True}
(output/'dataset_verification.json').write_text(json.dumps(check_result,indent=2)+'\n')
result={'learner_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'checks_passed':len(checks)}
(output/'learner_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(f'Verification outputs: {output}')
