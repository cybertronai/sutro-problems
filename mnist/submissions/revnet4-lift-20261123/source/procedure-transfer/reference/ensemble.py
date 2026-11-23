"""Explicit ordered FP32 ensemble accumulation on the A100."""
import hashlib
import torch
import triton
import triton.language as tl


@triton.jit
def _add(S,X,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    previous=tl.load(S+i,i<N,other=0.)
    value=tl.load(X+i,i<N,other=0.)
    tl.store(S+i,previous+value,i<N)


class Ensemble:
    def __init__(self,trainers):
        self.trainers=trainers
        self.total=torch.zeros_like(trainers[0].output_scores)
        self.predictions=torch.empty(len(self.total),dtype=torch.int64,device='cuda')
        self.compiled={}
    def add(self,trainer):
        kernel=_add[(triton.cdiv(self.total.numel(),128),)](
            self.total,trainer.output_scores,self.total.numel(),128,enable_fp_fusion=False)
        self.compiled['ensemble_add__'+hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest()[:16]]=kernel
    def invoke(self):
        self.total.zero_()
        for trainer in self.trainers:
            trainer.invoke()
            self.add(trainer)
        self.trainers[0].network.backend.argmax(self.total,self.predictions)
    def fingerprint(self):
        import numpy as np
        def digest(value):
            array=value.detach().cpu().numpy()
            return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        members=[]
        for trainer in self.trainers:
            members.append(dict(seed=trainer.seed,
                parameters={name:digest(value) for name,value in trainer.network.params.items()},
                buffers={name:digest(value) for name,value in getattr(trainer.network,'buffers',{}).items()},
                velocities={name:digest(value) for name,value in trainer.network.velocity.items()},
                scores=digest(trainer.output_scores),schedule_position=digest(trainer.offset)))
        return dict(members=members,ensemble_scores=digest(self.total),predictions=digest(self.predictions))
