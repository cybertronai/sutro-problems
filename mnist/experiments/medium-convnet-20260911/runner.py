"""Accuracy-only ConvNet search: training arrays in, validation evidence out.

There is deliberately no test-data loader or test-evaluation entry point here.
Architecture selection, checkpoint selection, and normalization use only the
fixed training/validation partition. All models start from random weights.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import time
import numpy as np

SPLIT_SEED = 20260914
SEARCH_SEED = 11
REPLICATION_SEEDS = (22, 33)
FINAL_SEEDS = (101, 102, 103)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def candidates():
    # width, depth, activation, pooling, dropout, lr, weight_decay, augmentation
    grid = [
        (32, 3, 'gelu', 'none', .2, .001, .001, 'none'),
        (48, 3, 'gelu', 'none', .2, .001, .001, 'none'),
        (64, 3, 'gelu', 'none', .2, .001, .001, 'none'),
        (32, 4, 'gelu', 'none', .2, .001, .001, 'none'),
        (32, 3, 'gelu', 'max', .2, .003, .001, 'none'),
        (32, 2, 'gelu', 'max', .1, .001, .01, 'none'),
        (64, 3, 'relu', 'none', .2, .001, .001, 'none'),
        (32, 3, 'gelu', 'none', .2, .001, .001, 'mild_affine'),
        (64, 3, 'gelu', 'none', .2, .001, .001, 'mild_affine'),
        (64, 4, 'gelu', 'none', .2, .001, .001, 'mild_affine'),
    ]
    rows = []
    for i, (width, depth, activation, pooling, dropout, lr, wd, augmentation) in enumerate(grid, 1):
        rows.append({'id': f'cnn-{i:02d}', 'architecture': 'cnn', 'width': width,
            'depth': depth, 'activation': activation, 'pooling': pooling,
            'dropout': dropout, 'learning_rate': lr, 'weight_decay': wd,
            'augmentation': augmentation, 'batch_size': 128, 'epochs': 100,
            'head_width': width * 4, 'image_size': 9})
    return rows


def split_indices(labels):
    """Exact 4800/1200 stratification with stable largest-remainder ties."""
    assert labels.shape == (6000,)
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    counts = np.bincount(labels, minlength=10)
    allocations = np.floor(counts * .2).astype(int)
    order = np.argsort(-(counts * .2 - allocations), kind='stable')
    allocations[order[:1200 - allocations.sum()]] += 1
    fit, val = [], []
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        val.extend(indices[:allocations[digit]])
        fit.extend(indices[allocations[digit]:])
    fit, val = rng.permutation(fit), rng.permutation(val)
    assert len(fit) == 4800 and len(val) == 1200
    assert len(set(fit) & set(val)) == 0
    assert sorted(np.r_[fit, val].tolist()) == list(range(6000))
    return fit, val


def load_training(path, canonical):
    with np.load(path, allow_pickle=False) as archive:
        if not set(archive.files) <= {'train_images', 'train_labels', 'train_indices'}:
            raise ValueError('Search requires a physically training-only archive')
        # This allowlist is the only archive array access in the search runner.
        images, labels = archive['train_images'], archive['train_labels']
    for name, array in [('train_images', images), ('train_labels', labels)]:
        spec = canonical['tiers']['medium']['arrays'][name]
        assert list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype']
        assert array_hash(array) == spec['sha256_c_order_little_endian'], name
    return images, labels


def protocol_document(images, labels, source_hashes):
    return {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Accuracy-only canonical MNIST-medium ConvNet search; no hardware benchmarking or translation',
        'dataset_profile': 'competition-v2', 'dataset_seed': 20260910,
        'learner_input_allowlist': ['train_images', 'train_labels'],
        'input_sha256': {'train_images': array_hash(images), 'train_labels': array_hash(labels)},
        'source_sha256': source_hashes, 'split_seed': SPLIT_SEED,
        'training_examples': 4800, 'validation_examples': 1200,
        'initial_seed': SEARCH_SEED, 'candidates': candidates(),
        'optimizer': 'AdamW, betas=(0.9,0.999), eps=1e-8, no label smoothing',
        'scheduler': 'CosineAnnealingLR, horizon100 epochs, minimum2% of initial learning rate; step after each epoch',
        'minibatches': 'New seeded CUDA permutation every epoch; retain the final partial batch',
        'normalization': 'One mean and population standard deviation from unaugmented fit images only; max(std,1e-6)',
        'augmentation': {'mild_affine': {'probability': .5, 'degrees': [-8, 8],
            'translation_pixels_per_axis': [-.35, .35], 'inverse_sampling_scale': [.94, 1.06],
            'interpolation': 'bilinear', 'padding': 'zeros', 'align_corners': False,
            'ordering': 'raw pixels before normalization; training only; no input gradients'}},
        'checkpoint_selection': ['maximum validation correct count', 'minimum validation cross-entropy',
                                 'minimum parameter count', 'earliest epoch'],
        'initial_ranking': ['maximum best-checkpoint validation correct count', 'minimum corresponding validation loss',
                            'minimum parameter count', 'configuration id'],
        'replication': {'top_initial_candidates': 3, 'additional_seeds': list(REPLICATION_SEEDS),
                        'same_split_and_schedule': True},
        'replicated_ranking': ['maximum mean best-checkpoint validation correct across seeds11/22/33',
                              'minimum mean corresponding validation cross-entropy', 'minimum parameter count',
                              'configuration id'],
        'refit_epoch_rule': 'Median of the selected architecture best-checkpoint epochs for seeds11/22/33; preserve100-epoch cosine horizon',
        'final_seed_rule': {'primary_single_seed': 101, 'replication_seeds': [102, 103],
                           'never_select_best_test_seed': True},
        'ensemble_rule': 'For the preferred validation-ranked architecture, compare the float64 arithmetic mean of raw float32 validation logits from best checkpoints at seeds11/22/33 against the fixed primary seed11 checkpoint. Select the ensemble only for strictly higher validation correct count; ties favor the single model. Final ensemble uses the same frozen architecture/epoch at seeds101/102/103, averaged identically; primary single uses seed101. No learned weights or test-time augmentation.',
        'phase_gate': 'Search and replicate stop before any test arrays are opened; root must freeze architecture/epoch/seeds/ensemble policy before a separate refit or test process',
        'determinism': {'torch_use_deterministic_algorithms': True, 'cudnn_benchmark': False,
                        'cudnn_deterministic': True, 'tf32_matmul': False, 'tf32_cudnn': False,
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'precision': 'float32, no autocast'},
        'prohibited': ['pretrained weights', 'historical checkpoints', 'test arrays during search',
                       'test-label model/seed/checkpoint selection', 'W&B integration', 'energy/runtime benchmark or IL translation']}


def build_model(config):
    from torch import nn
    def activation():
        return nn.GELU(approximate='none') if config['activation'] == 'gelu' else nn.ReLU()
    layers, channels, spatial = [], 1, 9
    for layer in range(config['depth']):
        layers += [nn.Conv2d(channels, config['width'], 3, padding=1, bias=False),
                   nn.BatchNorm2d(config['width']), activation()]
        channels = config['width']
        if config['pooling'] == 'max' and layer == min(1, config['depth'] - 1):
            layers.append(nn.MaxPool2d(2)); spatial //= 2
    return nn.Sequential(nn.Sequential(*layers), nn.Flatten(),
        nn.Linear(config['width'] * spatial * spatial, config['head_width']), activation(),
        nn.Dropout(config['dropout']), nn.Linear(config['head_width'], 10))


def train_run(images, labels, split, config, seed, phase, provenance):
    """Return serializable epoch curves, best checkpoint, and validation logits."""
    import torch
    import torch.nn.functional as F
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_model(config).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    fit, val = np.asarray(split['fit_positions']), np.asarray(split['validation_positions'])
    raw = torch.from_numpy(np.ascontiguousarray(images[fit])).cuda()
    y = torch.from_numpy(np.ascontiguousarray(labels[fit])).long().cuda()
    vraw = torch.from_numpy(np.ascontiguousarray(images[val])).cuda()
    vy = torch.from_numpy(np.ascontiguousarray(labels[val])).long().cuda()
    mean, std = raw.mean().item(), max(raw.std(unbiased=False).item(), 1e-6)
    clean, vclean = (raw-mean)/std, (vraw-mean)/std
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=config['epochs'], eta_min=config['learning_rate']*.02)

    def augmented(batch):
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

    def evaluate(x, target, return_logits=False):
        model.eval(); total_loss, correct, logits = 0., 0, []
        with torch.inference_mode():
            for first in range(0,len(x),512):
                out = model(x[first:first+512]); truth = target[first:first+512]
                total_loss += F.cross_entropy(out,truth,reduction='sum').item()
                correct += int((out.argmax(1)==truth).sum().item())
                if return_logits: logits.append(out.cpu().numpy())
        return {'correct':correct,'total':len(x),'loss':total_loss/len(x)}, np.concatenate(logits) if return_logits else None

    history, best, best_state, best_logits = [], None, None, None
    started = time.monotonic()
    for epoch in range(1,config['epochs']+1):
        model.train(); order = torch.randperm(len(raw),device='cuda')
        lr = optimizer.param_groups[0]['lr']
        for positions in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            batch = (augmented(raw[positions])-mean)/std
            loss = F.cross_entropy(model(batch),y[positions])
            loss.backward(); optimizer.step()
        scheduler.step()
        training,_ = evaluate(clean,y)
        validation, logits = evaluate(vclean,vy,True)
        row = {'epoch':epoch,'learning_rate':lr,'train':training,'validation':validation,
               'elapsed_seconds':time.monotonic()-started}
        if not np.isfinite([training['loss'],validation['loss']]).all():
            raise RuntimeError(f'Nonfinite loss for {config["id"]}, seed{seed}, epoch{epoch}')
        history.append(row)
        rank = (-validation['correct'],validation['loss'],parameter_count,epoch)
        if best is None or rank < (-best['validation']['correct'],best['validation']['loss'],parameter_count,best['epoch']):
            best = row.copy(); best_logits = logits.copy()
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch == 1 or epoch%10 == 0:
            print(f'{phase} {config["id"]} seed{seed} epoch{epoch}/{config["epochs"]}: '
                  f'validation {validation["correct"]}/1200; best {best["validation"]["correct"]}/1200',flush=True)
    state_buffer = io.BytesIO()
    torch.save({'state_dict':best_state,'config':config,'seed':seed,'epoch':best['epoch'],
                'normalization':{'mean':mean,'std':std},'schedule_epochs':config['epochs'],
                'phase':phase,'provenance':provenance},state_buffer)
    checkpoint = state_buffer.getvalue()
    result = {'id':f'{phase}-{config["id"]}-s{seed}','phase':phase,'config':config,'seed':seed,
        'parameter_count':parameter_count,'normalization':{'mean':mean,'std':std},
        'best':best,'last':history[-1],'history':history,
        'best_validation_predictions':best_logits.argmax(1).astype(int).tolist(),
        'best_validation_logits_sha256':array_hash(best_logits),
        'best_checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
        'best_state_tensor_sha256':{key:array_hash(value.numpy()) for key,value in best_state.items()},
        'provenance':provenance,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_elapsed_seconds':time.monotonic()-started,
        'software':{'python':platform.python_version(),'numpy':np.__version__,
                    'torch':str(torch.__version__),'cuda':torch.version.cuda,
                    'cudnn':torch.backends.cudnn.version()},
        'hardware':{'gpu':torch.cuda.get_device_name(0)},
        'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),
                       'cudnn_benchmark':torch.backends.cudnn.benchmark,
                       'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                       'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
    return result,checkpoint,best_logits.tobytes()
