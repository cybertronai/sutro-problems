from functools import lru_cache
from pathlib import Path
import os
@lru_cache(None)
def extension():
 os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
 from torch.utils.cpp_extension import load
 p=Path(__file__).resolve().parent
 return load(name='sutro_centered_halfnorm_small_members_bn_stagepack',sources=[str(p/'kernels/centered_halfnorm_small_members_bindings.cpp'),str(p/'kernels/centered_halfnorm_small_members.cu')],extra_cflags=['-O0','-g0'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)
def transform(last,middle,members,true_count):
 import torch
 N=last.shape[0]//members
 out=torch.empty((N,2*members*512),device=last.device,dtype=torch.bfloat16)
 means=torch.empty((2*members,512),device=last.device,dtype=torch.float32)
 partial=torch.empty((2*members,(true_count+1023)//1024,512),device=last.device,dtype=torch.float32)
 extension().into(last,middle,out,means,partial,members,true_count)
 return out
