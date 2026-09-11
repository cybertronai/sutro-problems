"""Frozen ConvNet refit and prediction only; no test labels or model selection.

Run after selection.json is frozen. Every model starts fresh, learns from all
6,000 training rows, and stops at the prescribed epoch. No search checkpoint is
accepted. The caller separately freezes and evaluates the returned predictions.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
import random
import sys
import time
import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner

IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = (modal.Image.from_registry(IMAGE_REF).pip_install('numpy==2.2.6')
         .add_local_file(str(HERE/'runner.py'), remote_path='/root/runner.py'))
app = modal.App('sutro-mnist-medium-convnet-frozen-refit')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=8192, timeout=1200,
              startup_timeout=300, min_containers=0, max_containers=2,
              buffer_containers=0, scaledown_window=2, retries=0)
def refit_remote(payload, selection, seed, provenance):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import torch.nn.functional as F
    import importlib.util
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('frozen_runner', '/root/runner.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert sha(Path('/root/runner.py')) == provenance['source_sha256']['runner.py']
    assert set(payload) == {'train_images', 'train_labels', 'test_images'}
    arrays = {name: np.frombuffer(value['bytes'], dtype=value['dtype']).reshape(value['shape']).copy()
              for name, value in payload.items()}
    for name, value in arrays.items():
        assert module.array_hash(value) == provenance['input_sha256'][name], name
    assert seed in selection['seeds'] == [101, 102, 103]
    config = selection['config']
    assert config['epochs'] == selection['schedule_epochs'] == 100
    assert 1 <= selection['epochs'] <= selection['schedule_epochs']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = module.build_model(config).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    raw = torch.from_numpy(arrays['train_images']).cuda()
    y = torch.from_numpy(arrays['train_labels']).long().cuda()
    mean, std = raw.mean().item(), max(raw.std(unbiased=False).item(), 1e-6)
    clean = (raw-mean)/std
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=selection['schedule_epochs'], eta_min=config['learning_rate']*.02)

    def augmented(batch):
        # Identical random draws and arithmetic to the frozen search runner.
        if config['augmentation'] == 'none': return batch
        with torch.no_grad():
            count = len(batch)
            angle = (torch.rand(count,device='cuda')*16-8)*(np.pi/180)
            scale = torch.rand(count,device='cuda')*.12+.94
            translation = (torch.rand((count,2),device='cuda')*.7-.35)*(2/9)
            theta = torch.zeros((count,2,3),dtype=torch.float32,device='cuda')
            theta[:,0,0] = scale*angle.cos(); theta[:,0,1] = -scale*angle.sin()
            theta[:,1,0] = scale*angle.sin(); theta[:,1,1] = scale*angle.cos()
            theta[:,:,2] = translation
            grid = F.affine_grid(theta,batch.shape,align_corners=False)
            changed = F.grid_sample(batch,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            mask = (torch.rand(count,device='cuda')<.5)[:,None,None,None]
            return torch.where(mask,changed,batch)

    def training_metrics():
        model.eval(); total_loss, correct = 0., 0
        with torch.inference_mode():
            for first in range(0,len(clean),512):
                out = model(clean[first:first+512]); truth = y[first:first+512]
                total_loss += F.cross_entropy(out,truth,reduction='sum').item()
                correct += int((out.argmax(1)==truth).sum().item())
        return {'correct':correct,'total':len(clean),'loss':total_loss/len(clean)}

    history = []; started = time.monotonic()
    for epoch in range(1, selection['epochs']+1):
        model.train(); order = torch.randperm(len(raw),device='cuda')
        lr = optimizer.param_groups[0]['lr']
        for positions in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            batch = (augmented(raw[positions])-mean)/std
            loss = F.cross_entropy(model(batch),y[positions])
            loss.backward(); optimizer.step()
        scheduler.step()
        training = training_metrics()
        if not np.isfinite(training['loss']):
            raise RuntimeError(f'Nonfinite training loss for seed {seed}, epoch {epoch}')
        history.append({'epoch':epoch,'learning_rate':lr,'train':training,
                        'elapsed_seconds':time.monotonic()-started})
        if epoch == 1 or epoch%10 == 0 or epoch == selection['epochs']:
            print(f'Frozen refit seed{seed} epoch{epoch}/{selection["epochs"]}: '
                  f'train {training["correct"]}/{len(raw)}',flush=True)
    # Query images are only transferred/normalized after all updates are complete.
    # No statistics are estimated from them and they never enter an optimizer step.
    model.eval(); query = (torch.from_numpy(arrays['test_images']).cuda()-mean)/std
    query_logits = []
    with torch.inference_mode():
        for first in range(0,len(query),512):
            query_logits.append(model(query[first:first+512]).cpu().numpy())
    logits = np.concatenate(query_logits)
    assert logits.shape == (6000,10) and logits.dtype == np.float32 and np.isfinite(logits).all()
    predictions = logits.argmax(1).astype(np.int64)
    state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save({'state_dict':state,'config':config,'seed':seed,'epoch':selection['epochs'],
                'normalization':{'mean':mean,'std':std},'schedule_epochs':selection['schedule_epochs'],
                'phase':'frozen_refit','provenance':provenance},buffer)
    checkpoint = buffer.getvalue()
    result = {'id':f'seed{seed}','phase':'frozen_refit','config':config,'seed':seed,
        'epochs':selection['epochs'],'schedule_epochs':selection['schedule_epochs'],
        'parameter_count':parameter_count,'training_examples':len(raw),'query_examples':len(query),
        'normalization':{'mean':mean,'std':std,'source':'all 6000 training images only'},
        'history':history,'checkpoint_policy':'last fixed epoch only; no validation or query labels',
        'predictions_sha256':module.array_hash(predictions),'logits_sha256':module.array_hash(logits),
        'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
        'state_tensor_sha256':{key:module.array_hash(value.numpy()) for key,value in state.items()},
        'provenance':provenance,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_job_elapsed_seconds':time.monotonic()-started,
        'elapsed_time_scope':'progress metadata only; not a controlled hardware benchmark',
        'software':{'python':platform.python_version(),'numpy':np.__version__,
                    'torch':str(torch.__version__),'cuda':torch.version.cuda,
                    'cudnn':torch.backends.cudnn.version()},
        'hardware':{'gpu':torch.cuda.get_device_name(0)},'container_image':IMAGE_REF,
        'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),
                       'cudnn_benchmark':torch.backends.cudnn.benchmark,
                       'cudnn_deterministic':torch.backends.cudnn.deterministic,
                       'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                       'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
    return result,checkpoint,logits.tobytes()


@app.local_entrypoint()
def main(data: str, selection: str, output: str):
    import numpy as np
    selection_path = Path(selection)
    evidence_dir = selection_path.parent
    frozen = json.loads(selection_path.read_text())
    assert frozen['seeds'] == [101,102,103]
    assert frozen['selected_inference'] in ('single','ensemble')
    assert frozen['selected_name'] == ('ensemble' if frozen['selected_inference']=='ensemble' else 'seed101')
    for name in ('runner.py','modal_train.py','refit.py'):
        assert frozen['source_sha256'][name] == sha(HERE/name), name+' changed after freeze'
    assert frozen['protocol_sha256'] == sha(evidence_dir/'protocol.json')
    assert frozen['validation_selection_sha256'] == sha(evidence_dir/'validation_selection.json')
    preferred = json.loads((evidence_dir/'validation_selection.json').read_text())['preferred_candidate']
    assert frozen['config'] == preferred['config']
    assert frozen['epochs'] == preferred['refit_epochs']
    assert frozen['selected_inference'] == preferred['selected_inference']
    assert frozen['schedule_epochs'] == frozen['config']['epochs'] == 100
    allowed = {'train_images','train_labels','test_images'}
    with np.load(data,allow_pickle=False) as archive:
        assert set(archive.files) == allowed, 'Refit requires exactly the three allowed input arrays'
        arrays = {name:archive[name] for name in sorted(allowed)}
    canonical = json.loads((HERE.parents[1]/'doc'/'dataset_manifest.json').read_text())['tiers']['medium']['arrays']
    input_sha = {}
    for name,array in arrays.items():
        spec = canonical[name]; digest = runner.array_hash(array)
        assert list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype']
        assert digest == spec['sha256_c_order_little_endian']
        assert digest == frozen['input_sha256'][name]
        input_sha[name] = digest
    destination = Path(output); destination.mkdir(parents=True,exist_ok=True)
    # No silent resume or replacement of retained predictions/checkpoints.
    if any(destination.iterdir()):
        raise FileExistsError('Refit output directory must be empty')
    payload = {name:{'shape':array.shape,'dtype':str(array.dtype),
                     'bytes':np.ascontiguousarray(array).tobytes()} for name,array in arrays.items()}
    provenance = {'selection_sha256':sha(selection_path),'protocol_sha256':frozen['protocol_sha256'],
        'source_sha256':frozen['source_sha256'],'input_sha256':input_sha,
        'validation_selection_sha256':frozen['validation_selection_sha256'],
        'allowed_input_arrays':sorted(allowed),'initialization':'fresh random parameters per fixed seed',
        'checkpoint_inputs':[]}
    write_json(destination/'refit_plan.json',{'selection':frozen,'provenance':provenance,
        'started_at_utc':datetime.now(timezone.utc).isoformat()})
    jobs = [(payload,frozen,seed,provenance) for seed in frozen['seeds']]
    results, all_logits = {}, {}
    for result,checkpoint,logit_bytes in refit_remote.starmap(jobs,order_outputs=False):
        seed = result['seed']; name = f'seed{seed}'
        assert seed in frozen['seeds'] and seed not in results
        assert result['provenance'] == provenance and result['config'] == frozen['config']
        assert hashlib.sha256(checkpoint).hexdigest() == result['checkpoint_sha256']
        logits = np.frombuffer(logit_bytes,dtype=np.float32).reshape(6000,10)
        assert np.isfinite(logits).all() and runner.array_hash(logits) == result['logits_sha256']
        predictions = logits.argmax(1).astype(np.int64)
        assert runner.array_hash(predictions) == result['predictions_sha256']
        files = {'predictions':f'predictions-{name}.npy','logits':f'logits-{name}.npy',
                 'checkpoint':f'checkpoint-{name}.pt'}
        np.save(destination/files['predictions'],predictions)
        np.save(destination/files['logits'],logits)
        (destination/files['checkpoint']).write_bytes(checkpoint)
        result['files'] = files
        result['file_sha256'] = {key:sha(destination/value) for key,value in files.items()}
        write_json(destination/f'result-{name}.json',result)
        results[seed] = result; all_logits[seed] = logits
        print(f'Saved frozen seed{seed} predictions; no query labels accessed.',flush=True)
    assert sorted(results) == frozen['seeds']
    ensemble = np.mean(np.stack([all_logits[seed] for seed in frozen['seeds']]),axis=0,dtype=np.float64)
    predictions = ensemble.argmax(1).astype(np.int64)
    np.save(destination/'logits-ensemble.npy',ensemble)
    np.save(destination/'predictions-ensemble.npy',predictions)
    write_json(destination/'refit_summary.json',{'provenance':provenance,
        'selected_name':frozen['selected_name'],'seeds':frozen['seeds'],'epochs':frozen['epochs'],
        'config':frozen['config'],'ensemble_accumulation':'float64 arithmetic mean of three float32 raw-logit arrays',
        'ensemble_predictions_sha256':runner.array_hash(predictions),'ensemble_logits_sha256':runner.array_hash(ensemble),
        'ensemble_file_sha256':{name:sha(destination/name) for name in ('logits-ensemble.npy','predictions-ensemble.npy')},
        'test_labels_opened':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print('Frozen refits complete. Freeze every prediction file before separate evaluation.',flush=True)


if __name__ == '__main__':
    raise SystemExit('Run with: uvx --with numpy==2.2.6 modal==1.5.5 run '+__file__)
