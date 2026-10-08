from pathlib import Path
from functools import lru_cache
import os
@lru_cache(None)
def extension():
 os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
 from torch.utils.cpp_extension import load
 p=Path(__file__).resolve().parent/'kernels'
 return load(name='sutro_embedding_graph_s1000_fixed',sources=[str(p/'embedding_graph_bindings.cpp'),str(p/'embedding_graph.cu')],extra_cuda_cflags=['-O3','-lineinfo'],extra_ldflags=['-lcublas'],verbose=False)
