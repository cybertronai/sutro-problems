from pathlib import Path
from functools import lru_cache
import os
@lru_cache(None)
def extension(gain=.25):
 if gain!=.25:raise ValueError("fixed skip gain .25")
 os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
 from torch.utils.cpp_extension import load
 p=Path(__file__).resolve().parent/'kernels'
 return load(name='sutro_persistent_bf16_deep4_batchnorm_staged_back128_ltsearch_head16_residual_bn_stagepack',sources=[str(p/'persistent_bf16_deep4_batchnorm_staged_back128_ltsearch_head16_residual_bindings.cpp'),str(p/'persistent_bf16_deep4_batchnorm_staged_back128_ltsearch_head16_residual.cu')],extra_cflags=['-O0','-g0'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],extra_ldflags=['-lcublas','-lcublasLt'],verbose=False)
def classify(x,y,q,width=256,members=4,batch=512,views=1,steps=400,lr1=.3,lr2=.3,head_lr=.3,momentum=.9,seed=42,use_graph=True,schedule='none',weight_decay=.0001,dropout=0.,input_noise=0.,input_scale=-1.,optimizer='nesterov',mixup=0.,init_gain=1.,head_gain=0.,initialization='gaussian',swa_fraction=0.,dictionary_scale=1.,data_noise=.5,skip_gain=.25):
 schedule_id={'none':0,'cosine':1,'step':2}[schedule]
 if optimizer not in ('nesterov','momentum','sgd'):raise ValueError('unknown optimizer')
 if optimizer=='momentum':schedule_id+=3
 if optimizer=='sgd':momentum=0.
 init_kind={'gaussian':0,'paired':1,'hadamard':2,'xavier':3,'kaiming':4,'xavier_relu':5,'data_centroid':6,'data_sample':7}[initialization]
 return extension(skip_gain).classify(x,y,q,width,members,batch,views,steps,lr1,lr2,head_lr,momentum,seed,use_graph,schedule_id,weight_decay,dropout,input_noise,input_scale,mixup,init_gain,head_gain,init_kind,swa_fraction,dictionary_scale,data_noise)
