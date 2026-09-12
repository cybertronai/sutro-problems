"""Audit evidence and ordered arithmetic; optionally verify actual A100 outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import reference
import run

HERE=Path(__file__).resolve().parent
run.check_protocol()
manifest=json.loads((HERE/'draw_manifest.json').read_text())
draw=manifest['draws'][0]
with np.load(HERE/draw['archive'],allow_pickle=False) as z:
    x=reference.transform(z['train_images'])
    y=(z['train_labels'][:,None]==np.arange(10)).astype(np.float32)
params=reference.parameters(32,101)
fast=reference.epoch(x,y,params,.2)
slow=reference.epoch(x,y,params,.2,reference.ordered_mm,reference.ordered_rows)
assert all(np.array_equal(a.view(np.uint32),b.view(np.uint32)) for a,b in zip(fast,slow,strict=True))
result={'ordered_FP32_full_epoch':True,'minibatches':40,'parameter_words':650}
gpu_path=HERE/'gpu_results.json'
if gpu_path.exists():
    g=json.loads(gpu_path.read_text())
    assert g['source_sha256']==run.sha(HERE/'gpu_benchmark.py')
    assert g['model_config']==dict(width=32,epochs=300,learning_rate=.2,seed=101,batch_size=25)
    for name,spec in draw['arrays'].items(): assert g['input_sha256'][name]==spec['sha256']
    pred=np.asarray(g['predictions'],dtype=np.int64)
    assert np.array_equal(pred,np.load(HERE/'predictions'/'draw-00.npy',allow_pickle=False))
    assert g['prediction_sha256_int64_le']==run.generator.array_hash(pred)
    p=np.asarray(g['final_parameter_bits_u32'],dtype=np.uint32)
    cpu=json.loads((HERE/'prediction_manifest.json').read_text())['draws'][0]
    boundaries=(0,288,320,640,650)
    for i in range(4):
        assert hashlib.sha256(p[boundaries[i]:boundaries[i+1]].astype('<u4').tobytes()).hexdigest()==cpu['parameter_sha256'][i]
    assert g['validation']['canonical']['scores_sha256_float32_le']==cpu['scores_sha256']
    for trial in g['trials']:
        a=trial['active']; baseline=trial['paired_idle_power_w']
        energy=(a['energy_j']-baseline*a['duration_s'])/trial['invocations']
        assert abs(energy-trial['idle_adjusted_j_per_invocation'])<1e-12
    result.update(a100_matches_cpu_all_predictions=True,a100_matches_cpu_all_parameter_bits=True,
                  a100_matches_cpu_all_score_bits=True,nvml_energy_recomputed=True)
run.write(HERE/'verification.json',result)
print(json.dumps(result,indent=2))
