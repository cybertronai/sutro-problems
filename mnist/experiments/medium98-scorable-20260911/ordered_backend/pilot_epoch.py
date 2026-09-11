"""One training-only ordered CNN epoch to estimate validation-run duration.

No validation labels or formal query dataset enter the remote job. This is a
feasibility pilot, not a submission time/energy or accuracy measurement.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys
import modal
HERE=Path(__file__).resolve().parent
SOURCE_NAMES=['ops.py','network.py','schedule.py','bn_ops.py','pilot_epoch.py']
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6')
for name in SOURCE_NAMES:image=image.add_local_file(str(HERE/name),remote_path='/root/backend/'+name)
app=modal.App('sutro-ordered-cnn-one-epoch-pilot')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,
              startup_timeout=300,max_containers=1,retries=0)
def run(payload,config,seed,hashes):
    import time
    import numpy as np
    import torch
    import triton
    sys.path.insert(0,'/root/backend')
    from network import Trainer
    from schedule import array_hash
    for name,digest in hashes.items():assert sha(Path('/root/backend')/name)==digest
    assert set(payload)=={'train_images','train_labels','query_images'}
    arrays={name:np.frombuffer(row['bytes'],dtype=row['dtype']).reshape(row['shape']).copy() for name,row in payload.items()}
    torch.set_num_threads(4)
    trainer=Trainer(arrays['train_images'],arrays['train_labels'],config,seed)
    started=time.perf_counter();trainer.prepare(arrays['query_images']);torch.cuda.synchronize()
    preparation=time.perf_counter()-started
    trainer.initialize();torch.cuda.synchronize();started=time.perf_counter()
    trainer.train_epoch(1);torch.cuda.synchronize();train_seconds=time.perf_counter()-started
    started=time.perf_counter();trainer.inference_graph.replay();torch.cuda.synchronize();inference_seconds=time.perf_counter()-started
    output=trainer.outputs()
    assert all(np.isfinite(a).all() for field in ('parameters','buffers','velocities') for a in output[field].values())
    assert np.isfinite(output['scores']).all()
    result={'scope':'One actual training-only4800-row epoch and1200-row unlabeled validation inference; neither submission benchmark nor accuracy measurement.',
        'config':config,'seed':seed,'source_sha256':hashes,
        'input_sha256':{name:array_hash(a) for name,a in arrays.items()},
        'preparation_seconds_for_one_epoch_schedule':preparation,'training_epoch_seconds':train_seconds,
        'unlabeled_inference_seconds':inference_seconds,
        'projected150epoch_training_seconds_excluding_validation_and_full_schedule_setup':150*train_seconds,
        'parameter_count':sum(a.size for a in trainer.initial_arrays.values()),
        'initial_parameter_sha256':{name:array_hash(a) for name,a in trainer.initial_arrays.items()},
        'final_parameter_sha256':{name:array_hash(a) for name,a in output['parameters'].items()},
        'final_buffer_sha256':{name:array_hash(a) for name,a in output['buffers'].items()},
        'scores_sha256':array_hash(output['scores']),'schedule_manifests':trainer.schedule_manifests,
        'hardware':{'gpu':torch.cuda.get_device_name(0),'uuid':str(torch.cuda.get_device_properties(0).uuid)},
        'software':{'torch':str(torch.__version__),'numpy':np.__version__,'triton':triton.__version__,'cuda':torch.version.cuda},
        'test_dataset_accessed':False,'query_labels_supplied':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    print(json.dumps(result),flush=True)
    return result

@app.local_entrypoint()
def main(data:str='',split:str='',output:str=''):
    import numpy as np
    source=Path(data);split_path=Path(split)
    destination=Path(output) if output else HERE/'pilot-results'
    destination.mkdir(parents=True,exist_ok=True);assert not any(destination.iterdir())
    with np.load(source,allow_pickle=False) as archive:
        assert set(archive.files)<={'train_images','train_labels','train_indices'}
        images=archive['train_images'];labels=archive['train_labels']
    canonical=json.loads((HERE.parents[2]/'doc/dataset_manifest.json').read_text())['tiers']['medium']['arrays']
    for name,a in [('train_images',images),('train_labels',labels)]:
        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==canonical[name]['sha256_c_order_little_endian']
    split_data=json.loads(split_path.read_text());fit=np.asarray(split_data['fit_positions']);val=np.asarray(split_data['validation_positions'])
    assert len(fit)==4800 and len(val)==1200 and len(set(fit)&set(val))==0
    config={'width':32,'depth':3,'head_width':128,'image_size':9,'batch_size':128,'epochs':1,
        'learning_rate':.03,'momentum':.9,'weight_decay':.0001,'augmentation':'mild_affine',
        'loss':'approx_softmax_gradient','batch_norm':True,'dropout':.2}
    arrays={'train_images':images[fit],'train_labels':labels[fit],'query_images':images[val]}
    payload={name:{'dtype':str(a.dtype),'shape':a.shape,'bytes':np.ascontiguousarray(a).tobytes()} for name,a in arrays.items()}
    hashes={name:sha(HERE/name) for name in SOURCE_NAMES}
    (destination/'source-freeze.json').write_text(json.dumps({'sources':hashes,'config':config,
        'split_sha256':sha(split_path),'training_archive_sha256':sha(source)},indent=2)+'\n')
    result=run.remote(payload,config,11,hashes)
    assert hashes=={name:sha(HERE/name) for name in SOURCE_NAMES}
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('training_epoch_seconds','unlabeled_inference_seconds',
        'projected150epoch_training_seconds_excluding_validation_and_full_schedule_setup')},indent=2))
