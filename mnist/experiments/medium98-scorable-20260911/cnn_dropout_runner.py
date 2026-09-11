"""Preliminary training-only head-dropout extension to the selected BN family.

Native PyTorch/cuDNN reductions screen candidates only. A final ordered-FP32
backend must retrain and validate them before any new-dataset test evaluation.
No test-image or test-label input is accepted by this module.
"""
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np

SPLIT_SEED=20260914
IMAGE_REF=('ghcr.io/ab-10/wikitext-bench@'
 'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')

def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

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


def candidates():
    rows=[]
    grid=[(32,3,.03)]
    for width,depth,rate in grid:
        rows.append({'id':f'dropout-c{width}-d{depth}-lr{rate:g}',
            'width':width,'depth':depth,'learning_rate':rate,'head_width':4*width,
            'epochs':150,'batch_size':128,'momentum':.9,'weight_decay':.0001,
            'loss':'approx_softmax_gradient','normalization':'4*x-0.5',
            'augmentation':'mild_affine','dropout':.2,'batchnorm':True,'bn_epsilon':1e-5,'bn_momentum':.1,
            'pooling':'none','conv_bias':False,'head_bias':True,
            'initialization':'He uniform fan_in for all weights; zero biases'})
    return rows


def approximate_softmax(scores):
    maximum=scores.max(dim=1,keepdim=True).values
    shifted=(scores-maximum).clamp(min=-16.,max=0.)
    probability=shifted*float(np.float32(1/1024))+1.
    for _ in range(10):probability=probability*probability
    return probability/probability.sum(dim=1,keepdim=True)


def build_model(config):
    import torch
    from torch import nn
    layers=[];channels=1
    for _ in range(config['depth']):
        layers.extend([nn.Conv2d(channels,config['width'],3,padding=1,bias=False),
                       nn.BatchNorm2d(config['width'],eps=config['bn_epsilon'],momentum=config['bn_momentum']),
                       nn.ReLU()])
        channels=config['width']
    model=nn.Sequential(nn.Sequential(*layers),nn.Flatten(),
        nn.Linear(channels*81,config['head_width']),nn.ReLU(),nn.Dropout(config['dropout']),
        nn.Linear(config['head_width'],10))
    for layer in model.modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight,a=0,mode='fan_in',nonlinearity='relu')
            if layer.bias is not None:nn.init.zeros_(layer.bias)
    return model


def train_run(images,labels,split,config,seed,provenance):
    import torch
    import torch.nn.functional as F
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build_model(config).cuda()
    parameter_count=sum(p.numel() for p in model.parameters())
    params=list(model.parameters());velocity=[torch.zeros_like(p) for p in params]
    fit=np.asarray(split['fit_positions']);val=np.asarray(split['validation_positions'])
    raw=torch.from_numpy(np.ascontiguousarray(images[fit])).cuda()
    y=torch.from_numpy(np.ascontiguousarray(labels[fit])).long().cuda()
    target=F.one_hot(y,10).to(torch.float32)
    vraw=torch.from_numpy(np.ascontiguousarray(images[val])).cuda()
    vy=torch.from_numpy(np.ascontiguousarray(labels[val])).long().cuda()
    vtarget=F.one_hot(vy,10).to(torch.float32)
    clean=raw*4-.5;vclean=vraw*4-.5

    def augmented(batch):
        with torch.no_grad():
            count=len(batch)
            angle=(torch.rand(count,device='cuda')*16-8)*(np.pi/180)
            scale=torch.rand(count,device='cuda')*.12+.94
            translation=(torch.rand((count,2),device='cuda')*.7-.35)*(2/9)
            theta=torch.zeros((count,2,3),dtype=torch.float32,device='cuda')
            theta[:,0,0]=scale*angle.cos();theta[:,0,1]=-scale*angle.sin()
            theta[:,1,0]=scale*angle.sin();theta[:,1,1]=scale*angle.cos()
            theta[:,:,2]=translation
            grid=F.affine_grid(theta,batch.shape,align_corners=False)
            changed=F.grid_sample(batch,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            mask=(torch.rand(count,device='cuda')<.5)[:,None,None,None]
            return torch.where(mask,changed,batch)

    def evaluate(x,t,truth):
        model.eval();values=[];loss=0.;correct=0
        with torch.inference_mode():
            for start in range(0,len(x),512):
                out=model(x[start:start+512]);error=approximate_softmax(out)-t[start:start+512]
                loss+=float((.5*error.square().sum()).item())
                correct+=int((out.argmax(1)==truth[start:start+512]).sum().item())
                values.append(out.cpu().numpy())
        return {'correct':correct,'total':len(x),'loss':loss/len(x)},np.concatenate(values)

    history=[];best=None;best_logits=None;best_state=None;started=time.perf_counter()
    status='complete'
    for epoch in range(1,config['epochs']+1):
        multiplier=1. if epoch<=90 else .1 if epoch<=127 else .01
        lr=float(np.float32(config['learning_rate']*multiplier))
        model.train();order=torch.randperm(len(raw),device='cuda')
        for positions in order.split(config['batch_size']):
            model.zero_grad(set_to_none=True)
            scores=model(augmented(raw[positions])*4-.5)
            # Defined update direction; no log or transcendental primitive.
            with torch.no_grad():
                gradient=(approximate_softmax(scores)-target[positions])*float(np.float32(1/len(positions)))
            scores.backward(gradient)
            with torch.no_grad():
                for p,v in zip(params,velocity):
                    grad=p.grad+p*float(np.float32(config['weight_decay']))
                    v.mul_(float(np.float32(config['momentum'])))
                    v.add_(grad)
                    p.sub_(v*lr)
        training,_=evaluate(clean,target,y)
        validation,logits=evaluate(vclean,vtarget,vy)
        finite=np.isfinite([training['loss'],validation['loss']]).all() and np.isfinite(logits).all()
        if not finite:
            status='nonfinite';history.append({'epoch':epoch,'learning_rate':lr,'status':'nonfinite'})
            print(f'{config["id"]} seed{seed}: nonfinite at epoch{epoch}',flush=True);break
        row={'epoch':epoch,'learning_rate':lr,'train':training,'validation':validation,
             'elapsed_seconds':time.perf_counter()-started}
        history.append(row)
        rank=(-validation['correct'],validation['loss'],epoch)
        if best is None or rank<(-best['validation']['correct'],best['validation']['loss'],best['epoch']):
            best=row.copy();best_logits=logits.copy()
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%10==0:
            print(f'{config["id"]} seed{seed} epoch{epoch}: validation{validation["correct"]}/1200; '
                  f'best{best["validation"]["correct"]}/1200',flush=True)
    result={'id':f'{config["id"]}-s{seed}','config':config,'seed':seed,'status':status,
        'parameter_count':parameter_count,'best':best,'history':history,'provenance':provenance,
        'arithmetic_scope':'Preliminary native PyTorch/cuDNN FP32 screening; ordered backend validation still required',
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,
        'software':{'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
                    'cudnn':torch.backends.cudnn.version(),'python':platform.python_version()},
        'hardware':{'gpu':torch.cuda.get_device_name(0)},
        'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),
           'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
    if best is None:return result,b'',None
    buffer=io.BytesIO();torch.save({'state_dict':best_state,'config':config,'seed':seed,'epoch':best['epoch']},buffer)
    checkpoint=buffer.getvalue()
    result['checkpoint_sha256']=hashlib.sha256(checkpoint).hexdigest()
    result['validation_logits_sha256']=array_hash(best_logits)
    result['validation_predictions']=best_logits.argmax(1).astype(int).tolist()
    return result,checkpoint,best_logits
