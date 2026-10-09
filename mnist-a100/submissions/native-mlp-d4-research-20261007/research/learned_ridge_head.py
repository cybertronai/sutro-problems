"""Native BF16 learned-feature ridge readout; experimental, not a submission."""
from pathlib import Path
from functools import lru_cache
import os
@lru_cache(None)
def extension():
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    from torch.utils.cpp_extension import load
    p=Path(__file__).resolve().parent/'kernels'
    return load(name='sutro_learned_ridge_head',sources=[str(p/'learned_ridge_head_bindings.cpp'),str(p/'learned_ridge_head.cu')],extra_cuda_cflags=['-O3','-lineinfo'],extra_ldflags=['-lcublas','-lcusolver'],verbose=False)
def fit(features,labels,members,regularization):
    return extension().fit(features,labels,members,regularization)
