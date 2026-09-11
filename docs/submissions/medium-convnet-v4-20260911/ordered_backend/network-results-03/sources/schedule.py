"""Input-independent constants for the declared ordered CNN.

Initialization exactly reproduces the preliminary architecture's CPU PyTorch
constructor draws followed by He uniform resets and zero biases. The final
algorithm uses separate PCG64 streams for permutation and affine schedules;
these are not claimed to reproduce preliminary CUDA RNG schedules.

Map convention: inverse sampling in pixel coordinates about the image centre.
Each destination gets four neighbours ordered top-left, top-right, bottom-left,
bottom-right; invalid neighbours use index N*H*W, a required zero sentinel.
The learner forms (((p00*c00+p01*c01)+p10*c10)+p11*c11), then *4−0.5,
with separate FP32 operations. Coefficients are seed-only program literals.
"""
import hashlib
import math
import numpy as np

SCHEDULE_DOMAIN=20261202

def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'),copy=False)).tobytes()).hexdigest()

def initial_parameters(config,seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    layers=[];channels=1
    for _ in range(config['depth']):
        layers.extend([nn.Conv2d(channels,config['width'],3,padding=1,bias=False),nn.ReLU()])
        channels=config['width']
    side=config.get('image_size',9)
    model=nn.Sequential(nn.Sequential(*layers),nn.Flatten(),
        nn.Linear(channels*side*side,config['head_width']),nn.ReLU(),nn.Linear(config['head_width'],10))
    for layer in model.modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight,a=0,mode='fan_in',nonlinearity='relu')
            if layer.bias is not None:nn.init.zeros_(layer.bias)
    names=[f'conv{i}.weight' for i in range(config['depth'])]+['head1.weight','head1.bias','head2.weight','head2.bias']
    return {name:p.detach().numpy().copy() for name,p in zip(names,model.parameters())}

def schedules(seed,n,epochs,side=9,augmentation='mild_affine'):
    streams=np.random.SeedSequence([int(seed),SCHEDULE_DOMAIN]).spawn(2)
    order_rng=np.random.Generator(np.random.PCG64(streams[0]))
    affine_rng=np.random.Generator(np.random.PCG64(streams[1]))
    centre=np.float32((side-1)/2)
    for epoch in range(1,epochs+1):
        order=order_rng.permutation(n).astype(np.int32)
        angle=affine_rng.uniform(-8.,8.,n)*(math.pi/180.)
        scale=affine_rng.uniform(.94,1.06,n)
        shift=affine_rng.uniform(-.35,.35,(n,2))
        mask=affine_rng.random(n)<.5
        if augmentation=='none':mask[:]=False
        elif augmentation!='mild_affine':raise ValueError(augmentation)
        # Trigonometry is performed once on seed-only metadata, never on data.
        theta=np.empty((n,2,3),np.float32)
        theta[:,0,0]=(scale*np.cos(angle)).astype(np.float32)
        theta[:,0,1]=(-scale*np.sin(angle)).astype(np.float32)
        theta[:,1,0]=(scale*np.sin(angle)).astype(np.float32)
        theta[:,1,1]=(scale*np.cos(angle)).astype(np.float32)
        theta[:,:,2]=(shift+float(centre)).astype(np.float32)
        theta[~mask]=np.array([[1.,0.,centre],[0.,1.,centre]],np.float32)
        yield {'epoch':epoch,'order':order,'theta':theta,'mask':mask}

def maps(schedule,side=9):
    order=schedule['order'];theta=schedule['theta'];n=len(order)
    centre=np.float32((side-1)/2)
    yy,xx=np.indices((side,side),dtype=np.int32)
    x=xx.reshape(1,-1).astype(np.float32)-centre
    y=yy.reshape(1,-1).astype(np.float32)-centre
    sx=theta[:,0,0,None]*x
    sx=sx+theta[:,0,1,None]*y
    sx=sx+theta[:,0,2,None]
    sy=theta[:,1,0,None]*x
    sy=sy+theta[:,1,1,None]*y
    sy=sy+theta[:,1,2,None]
    ix=np.floor(sx).astype(np.int32);iy=np.floor(sy).astype(np.int32)
    fx=sx-ix.astype(np.float32);fy=sy-iy.astype(np.float32)
    ox=np.float32(1)-fx;oy=np.float32(1)-fy
    coefficients=np.stack((ox*oy,fx*oy,ox*fy,fx*fy),axis=-1).astype(np.float32)
    indices=[]
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        qx=ix+dx;qy=iy+dy
        valid=(qx>=0)&(qx<side)&(qy>=0)&(qy<side)
        value=order[:,None]*side*side+qy*side+qx
        indices.append(np.where(valid,value,n*side*side).astype(np.int32))
    return np.stack(indices,axis=-1),coefficients

def apply_maps(raw,indices,coefficients):
    source=np.concatenate((raw.reshape(-1),np.zeros(1,np.float32)))
    result=source[indices[:,:,0]]*coefficients[:,:,0]
    for neighbor in range(1,4):result=result+source[indices[:,:,neighbor]]*coefficients[:,:,neighbor]
    result=result*np.float32(4)
    result=result-np.float32(.5)
    return result.reshape(len(indices),1,raw.shape[-2],raw.shape[-1])

def learning_rate(config,epoch):
    multiplier=1. if epoch<=90 else .1 if epoch<=127 else .01
    return np.float32(config['learning_rate']*multiplier)

def manifest(schedule,indices,coefficients):
    return {'epoch':schedule['epoch'],'order_sha256':array_hash(schedule['order']),
        'theta_sha256':array_hash(schedule['theta']),'mask_sha256':array_hash(schedule['mask']),
        'indices_sha256':array_hash(indices),'coefficients_sha256':array_hash(coefficients)}

def iter_epochs(seed,n_train,epochs,side=9,augmentation='mild_affine'):
    for row in schedules(seed,n_train,epochs,side,augmentation):
        indices,coefficients=maps(row,side)
        yield row['order'],indices,coefficients,manifest(row,indices,coefficients)

def get_epoch(seed,epoch,n_train,side=9,augmentation='mild_affine'):
    """One-based epoch; streaming iter_epochs is faster for all epochs."""
    if epoch<1:raise ValueError('epoch must be positive')
    for row in schedules(seed,n_train,epoch,side,augmentation):pass
    indices,coefficients=maps(row,side)
    return row['order'],indices,coefficients
