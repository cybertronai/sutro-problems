"""Native FP32 CTA-local CG for regularized <=64D local hull systems."""
from pathlib import Path
import ctypes as ct,inspect
import torch
from kernels.nvrtc_driver import Module
_modules={}
def solve(a,b,steps=48):
 batch,d,_=a.shape;assert d<=64 and a.is_contiguous() and b.is_contiguous() and b.shape==(batch,d,1)
 dev=a.device.index
 if dev not in _modules:_modules[dev]=Module(Path(__file__).with_suffix('.cu').read_text(),'classical_hull_cg.cu')
 out=torch.empty_like(b);_modules[dev].launch('hull_cg',batch,64,[a,b,out,ct.c_int(d),ct.c_int(steps)]);return out

def install(m,steps=48):
 checks=[]
 def checked(a,b):
  a=a.contiguous();b=b.contiguous();out=solve(a,b,steps)
  if not checks and not torch.cuda.is_current_stream_capturing():
   ad=a.double();bd=b.double();ref=torch.linalg.solve(ad,bd)
   relative=float((out.double()-ref).norm()/ref.norm());residual=float((ad@out.double()-bd).norm()/bd.norm())
   assert bool(torch.isfinite(out).all()) and relative<1e-4 and residual<1e-4,(relative,residual)
   checks.append(dict(shape=tuple(a.shape),steps=steps,relative_error=relative,residual=residual))
  return out
 source=inspect.getsource(m.hknn)
 old='lc = torch.linalg.cholesky_ex(torch.add(eye[None], g, alpha=2.0 ** -20))[0]\n    out = (torch.cholesky_solve(r, lc).square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)'
 assert source.count(old)==1
 m._hull_solve=checked;m._hull_checks=checks
 source=source.replace(old,'solution = _hull_solve(torch.add(eye[None], g, alpha=2.0 ** -20), r)\n    out = (solution.square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)')
 exec(source,m.__dict__)
