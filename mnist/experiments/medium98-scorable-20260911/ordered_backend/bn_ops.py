"""Triton lowering of the explicit BN formulas in bn_ref.py.

All channel reductions follow n,y,x from +0. Every multiply/add is separate.
Division is inline div.rn.f32, because Triton3.1 div_rn inserts FTZ.
"""
import numpy as np
import triton
import triton.language as tl

KW=dict(enable_fp_fusion=False,num_warps=4)


@triton.jit
def _divide(x,y):
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',
        constraints='=f,f,f',args=[x,y],dtype=tl.float32,is_pure=True,pack=1)


@triton.jit
def _inverse(variance):
    value=variance+0.00001
    root=tl.full(variance.shape,1.,tl.float32)
    for _ in tl.static_range(16):
        quotient=_divide(value,root)
        added=root+quotient
        root=added*0.5
    return _divide(tl.full(variance.shape,1.,tl.float32),root)


@triton.jit
def _stats(X,RM,RV,MEAN,INV,C:tl.constexpr,P:tl.constexpr,M:tl.constexpr,
           INV_M:tl.constexpr,CORRECTION:tl.constexpr,BLOCK:tl.constexpr):
    c=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total=tl.full((BLOCK,),0.,tl.float32)
    for r in range(M):
        value=tl.load(X+(r//P*C+c)*P+r%P,c<C,other=0.)
        total=total+value
    mean=total*INV_M
    total=tl.full((BLOCK,),0.,tl.float32)
    for r in range(M):
        value=tl.load(X+(r//P*C+c)*P+r%P,c<C,other=0.)
        centered=value-mean
        square=centered*centered
        total=total+square
    variance=total*INV_M
    inverse=_inverse(variance)
    tl.store(MEAN+c,mean,c<C);tl.store(INV+c,inverse,c<C)
    old_mean=tl.load(RM+c,c<C,other=0.)
    old_var=tl.load(RV+c,c<C,other=1.)
    old_mean_part=old_mean*0.9;new_mean_part=mean*0.1
    new_mean=old_mean_part+new_mean_part
    unbiased=variance*CORRECTION
    old_var_part=old_var*0.9;new_var_part=unbiased*0.1
    new_var=old_var_part+new_var_part
    tl.store(RM+c,new_mean,c<C);tl.store(RV+c,new_var,c<C)


@triton.jit
def _infer_stats(RV,INV,C:tl.constexpr,BLOCK:tl.constexpr):
    c=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    variance=tl.load(RV+c,c<C,other=1.)
    tl.store(INV+c,_inverse(variance),c<C)


@triton.jit
def _normalize(X,GAMMA,BETA,MEAN,INV,NORM,OUT,SIZE:tl.constexpr,C:tl.constexpr,P:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<SIZE;c=(i//P)%C
    x=tl.load(X+i,valid,other=0.);mean=tl.load(MEAN+c,valid,other=0.)
    inverse=tl.load(INV+c,valid,other=1.)
    gamma=tl.load(GAMMA+c,valid,other=1.);beta=tl.load(BETA+c,valid,other=0.)
    centered=x-mean;normalized=centered*inverse
    scaled=normalized*gamma;out=scaled+beta
    tl.store(NORM+i,normalized,valid);tl.store(OUT+i,out,valid)


@triton.jit
def _grad_stats(DY,NORM,DG,DB,C:tl.constexpr,P:tl.constexpr,M:tl.constexpr,BLOCK:tl.constexpr):
    c=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    beta=tl.full((BLOCK,),0.,tl.float32);gamma=tl.full((BLOCK,),0.,tl.float32)
    for r in range(M):
        index=(r//P*C+c)*P+r%P
        dy=tl.load(DY+index,c<C,other=0.);normalized=tl.load(NORM+index,c<C,other=0.)
        term=dy*normalized
        beta=beta+dy;gamma=gamma+term
    tl.store(DG+c,gamma,c<C);tl.store(DB+c,beta,c<C)


@triton.jit
def _grad_x(DY,NORM,GAMMA,INV,DG,DB,DX,SIZE:tl.constexpr,C:tl.constexpr,P:tl.constexpr,
            INV_M:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<SIZE;c=(i//P)%C
    dy=tl.load(DY+i,valid,other=0.);normalized=tl.load(NORM+i,valid,other=0.)
    gamma=tl.load(GAMMA+c,valid,other=1.);inverse=tl.load(INV+c,valid,other=1.)
    dg=tl.load(DG+c,valid,other=0.);db=tl.load(DB+c,valid,other=0.)
    average=db*INV_M;weighted=dg*INV_M
    centered=dy-average
    term=normalized*weighted
    centered=centered-term
    scaled=centered*gamma
    tl.store(DX+i,scaled*inverse,valid)


class BatchNorm:
    def __init__(self,backend):
        self.backend=backend
    def train(self,x,gamma,beta,running_mean,running_var,mean,inverse,normalized,out):
        n,c,h,w=x.shape;p=h*w;m=n*p
        assert m>1
        self.backend.save('bn_stats',_stats[(triton.cdiv(c,32),)](x,running_mean,running_var,mean,inverse,
            c,p,m,float(np.float32(1/m)),float(np.float32(m/(m-1))),32,**KW))
        self.backend.save('bn_normalize',_normalize[(triton.cdiv(x.numel(),128),)](
            x,gamma,beta,mean,inverse,normalized,out,x.numel(),c,p,128,**KW))
    def backward(self,dy,normalized,gamma,inverse,dx,dgamma,dbeta):
        n,c,h,w=dy.shape;p=h*w;m=n*p
        self.backend.save('bn_grad_stats',_grad_stats[(triton.cdiv(c,32),)](
            dy,normalized,dgamma,dbeta,c,p,m,32,**KW))
        self.backend.save('bn_grad_x',_grad_x[(triton.cdiv(dy.numel(),128),)](
            dy,normalized,gamma,inverse,dgamma,dbeta,dx,dy.numel(),c,p,float(np.float32(1/m)),128,**KW))
    def infer(self,x,gamma,beta,running_mean,running_var,inverse,normalized,out):
        n,c,h,w=x.shape;p=h*w
        self.backend.save('bn_infer_stats',_infer_stats[(triton.cdiv(c,32),)](running_var,inverse,c,32,**KW))
        self.backend.save('bn_infer_normalize',_normalize[(triton.cdiv(x.numel(),128),)](
            x,gamma,beta,running_mean,inverse,normalized,out,x.numel(),c,p,128,**KW))
