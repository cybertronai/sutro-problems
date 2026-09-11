"""Freeze the eleven-draw protocol and prepare isolated learner inputs.

Raw labels are read by trusted preparation solely to extract each draw's allowed
training labels. Per-draw query labels, statistics, and checksums are never
constructed here; a separate evaluator derives them after predictions freeze.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(HERE))
from mnist.code import data as generator
import learner

DATASET_SEEDS = list(range(20261001,20261012))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def spec(array):
    return {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':generator.array_hash(array)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists(): raise FileExistsError('Protocol already frozen; use a fresh output directory')
    sources={name:sha(HERE/name) for name in ('learner.py','prepare_draws.py','modal_accuracy.py')}
    raw_sources={}
    for key in ('train_images','train_labels'):
        filename,expected=generator.SOURCES[key]; path=args.raw/filename
        assert generator.file_hash(path,'md5')==expected
        raw_sources[key]={'filename':filename,'md5':expected,'sha256':sha(path)}
    protocol={'schema_version':1,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'tier':'medium','dataset_profile':'competition-v2','target_percent':'98',
        'dataset_seeds':DATASET_SEEDS,'training_examples':6000,'test_examples':6000,
        'source_pool_size':60000,'source_sha256':sources,'generator_sha256':sha(Path(generator.__file__)),
        'raw_sources':raw_sources,'container_image':learner.IMAGE_REF,'config':learner.CONFIG,
        'preparation_software':{'python':platform.python_version(),'numpy':np.__version__},
        'epochs':learner.EPOCHS,'schedule_epochs':100,'member_seeds':learner.MEMBER_SEEDS,
        'selected_inference':'ensemble','ensemble':'float64 arithmetic mean of three float32 raw-logit arrays; argmax breaks ties toward smallest digit',
        'draw_sampling':'Independent SeedSequence(dataset_seed).spawn(2)[0], PCG64 permutation of all60000 original training rows; positions0:6000 fit and6000:12000 query, disjoint within draw; overlap across draws allowed',
        'resize':'Canonical data.py: uint8 converted to float32 /255; exact separable box-area overlap averaging28→9; clip[0,1]; N,1,9,9',
        'learner_input_allowlist':['train_images','train_labels','test_images'],
        'normalization':'Mean and population standard deviation from each draw6000 training images only; max(std,1e-6)',
        'optimizer':'AdamW betas(.9,.999),eps1e-8,lr.001,wd.001; cosine horizon100 to2% initial lr, step each epoch',
        'minibatches':'New seeded CUDA permutation every epoch; B128 with final partial112 retained',
        'augmentation':{'probability':.5,'rotation_degrees':[-8,8],'inverse_sampling_scale':[.94,1.06],
            'translation_pixels_per_axis':[-.35,.35],'interpolation':'bilinear','padding':'zeros',
            'align_corners':False,'applied_to':'raw training images before normalization, no input gradients'},
        'initialization':'Fresh model, optimizer, scheduler and RNG reset for every draw/member; no pretrained or search checkpoints; no state or data reused across draws',
        'selection':'No search, early stopping or checkpoint selection on these11draws; final fixed71st epoch only',
        'development_disclosure':'Architecture and71epochs came from prior training-only search. Ensemble101/102/103 chosen for this new attempt after disclosed old single-dataset feasibility results. New dataset seeds and all choices fixed before any new-draw evaluation.',
        'prediction_freeze':'All11ensemble predictions and diagnostic member predictions/logits must be hashed in a manifest before ANY per-draw query labels are derived/opened or scored',
        'accuracy_rule':'Unrounded arithmetic mean across 11 draws; sample SD (ddof=1) across dataset draws; sum(correct) >= 64680 out of 66000, inclusive 98%',
        'determinism':{'torch_deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True,
            'tf32_matmul':False,'tf32_cudnn':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8','precision':'float32, no autocast'},
        'retention':'All predictions/logits/training histories/state tensor hashes retained; checkpoint binaries retained only for draw00',
        'phase_gate':'Accuracy first: do not benchmark energy/runtime or translate until all11draws are frozen and aggregate accuracy passes',
        'per_draw_test_labels_created':False}
    write(out/'protocol.json',protocol)  # Freeze before sampling/read_idx.
    images=generator.read_idx(args.raw/raw_sources['train_images']['filename'],60000,True)
    labels=generator.read_idx(args.raw/raw_sources['train_labels']['filename'],60000,False)
    (out/'data').mkdir(exist_ok=True); (out/'draws').mkdir(exist_ok=True)
    manifests=[]
    for draw,seed in enumerate(DATASET_SEEDS):
        order,_=generator.source_permutations(seed=seed)
        fit,query=order[:6000],order[6000:12000]
        assert len(np.unique(np.r_[fit,query]))==12000
        allowed={'train_labels':labels[fit].copy()}
        for name,positions in [('train_images',fit),('test_images',query)]:
            pixels=images[positions].astype(np.float32); pixels/=np.float32(255)
            resized=generator.area_resize(pixels,9); np.clip(resized,0.,1.,out=resized)
            allowed[name]=resized[:,None,:,:]
        path=out/'data'/f'draw-{draw:02d}.npz'
        np.savez_compressed(path,**allowed)
        manifest={'draw_index':draw,'dataset_seed':seed,'profile':'competition-v2',
            'protocol_sha256':sha(out/'protocol.json'),'generator_sha256':protocol['generator_sha256'],
            'allowed_archive':f'data/draw-{draw:02d}.npz','allowed_archive_sha256':sha(path),
            'arrays':{name:spec(array) for name,array in allowed.items()},
            'train_indices':fit.tolist(),'test_indices':query.tolist(),
            'train_indices_spec':spec(fit),'test_indices_spec':spec(query),
            'train_test_disjoint':True,'source_pool_size':60000,'test_labels_created':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat()}
        mpath=out/'draws'/f'draw-{draw:02d}.json'; write(mpath,manifest)
        manifests.append({'draw_index':draw,'dataset_seed':seed,'path':f'draws/draw-{draw:02d}.json','sha256':sha(mpath)})
        print(f'Prepared isolated draw{draw:02d}, dataset seed{seed}',flush=True)
    write(out/'draw_manifest.json',{'protocol_sha256':sha(out/'protocol.json'),'draws':manifests,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'test_labels_created':False})


if __name__=='__main__':main()
