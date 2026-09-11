"""Fresh fixed ConvNet training for one allowed draw; no evaluation labels.

The algorithm is the disclosed prior C64/D3 ensemble, fixed before eleven new
resampled draws. Every member initializes anew and learns only its supplied
6,000 training rows. Training diagnostics never select a checkpoint.
"""
from datetime import datetime, timezone
import hashlib
import io
import platform
import random
import time
import numpy as np

IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
CONFIG = {'id':'cnn-09','architecture':'cnn','width':64,'depth':3,'activation':'gelu',
          'pooling':'none','dropout':0.2,'learning_rate':0.001,'weight_decay':0.001,
          'augmentation':'mild_affine','batch_size':128,'epochs':100,
          'head_width':256,'image_size':9}
EPOCHS = 71
MEMBER_SEEDS = [101,102,103]


def array_hash(array):
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'),copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


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

def train_member(arrays, config, seed, provenance, keep_checkpoint=False):
    import torch
    import torch.nn.functional as F
    assert set(arrays) == {'train_images','train_labels','test_images'}
    assert config == CONFIG and seed in MEMBER_SEEDS
    selection = {'epochs':EPOCHS,'schedule_epochs':100}
    for name, value in arrays.items():
        assert array_hash(value) == provenance['input_sha256'][name]
    assert arrays['train_images'].shape == arrays['test_images'].shape == (6000,1,9,9)
    assert arrays['train_labels'].shape == (6000,)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_model(config).cuda()
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
        'predictions_sha256':array_hash(predictions),'logits_sha256':array_hash(logits),
        'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
        'state_tensor_sha256':{key:array_hash(value.numpy()) for key,value in state.items()},
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
    return result, checkpoint if keep_checkpoint else b'', logits
