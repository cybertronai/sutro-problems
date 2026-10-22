import math
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
GAMMA,RIDGE,RANK,SAMPLES,BLEND,NPOW=0.02,0.1,256,512,0.02,0.6
AGOP_N,M_STEPS,BSTEPS=(3000,10000),(16,16),14
PSEUDO_CONF,P,NS_ITERS,SHIFT=0.15,2000,15,3.0
F32=torch.float32
M,DIMS=1,(60,1024,512,256,10)
STEPS,BATCH,LR,BETA1,DECAY_START=7200,2500,0.01467,0.8,2/3
UF,GK=0.2,30
NOISE,SCALE,RECON,EPS,CSTD,SEED=0.25,0.6,2000.0,1e-10,0.025,11
MIX,MIX_A,MIX_B,MIX_C=0.5,0.1,0.6,0.9
TW,TEACH_TAU=0.25,0.1
ACT=torch.bfloat16
TAN,TAN_K,TAN_R=0.5,10,4
NB=4194304
HK_K,HK_C,HK_M,HK_N,HK_CF=40,3,2048,1024,0.5
IND={'combo_kernels':True,'triton.autotune_pointwise':False,'pattern_matcher':False,'fx_graph_cache':False,'autotune_local_cache':False,'reorder_for_peak_memory':False,'size_asserts':False,'scalar_asserts':False,'alignment_asserts':False}
CACHE={}
FP:tl.constexpr=tl.float32
BF:tl.constexpr=tl.bfloat16
T=torch.Tensor
def chol(a):
 return torch.linalg.cholesky_ex(a)[0]
def rownorm(x):
 return x*(x.shape[1]**0.5/x.norm(dim=1,keepdim=True).clamp_min(1e-06))**NPOW
def aug(a,b,shift=0.0):
 sa=a.square().sum(1,keepdim=True)
 sb=b.square().sum(1,keepdim=True)
 A=torch.cat([a,sa,torch.ones_like(sa)],1)
 B=torch.cat([2*GAMMA*b,torch.full_like(sb,-GAMMA),shift-GAMMA*sb],1).T.contiguous()
 return A,B
def kern(a,b):
 A,B=aug(a,b)
 return (A@B).exp_()
def h3_ops(z):
 A,B=aug(z,z,SHIFT)
 Ah,Bh=A.half(),B.half()
 Al,Bl=(A-Ah.float()).half(),(B-Bh.float()).half()
 return torch.cat([Ah,Ah,Al],1),torch.cat([Bh,Bl,Bh],0)
def panels(A3,B3,pp,rows,diag):
 out=[]
 for i in rows:
  r=min(pp,A3.shape[0]-i)
  pi=torch.mm(A3[i:i+r],B3[:,i:])
  pi.exp_()
  pi[:,:r].diagonal().fill_(diag)
  out.append(pi)
 return out
class Sym:
 def __init__(o,pans,pp,n):
  o.pans,o.pp,o.n=pans,pp,n
 def mv(o,dh):
  pp,n=o.pp,o.n
  y=torch.empty(n,dh.shape[1],device=dh.device)
  p0=o.pans[0]
  r0=min(p0.shape[0],n)
  torch.mm(p0[:r0,:n],dh,out_dtype=F32,out=y[:r0])
  if r0<n:
   torch.mm(p0[:,r0:n].T,dh[:r0],out_dtype=F32,out=y[r0:n])
  for j in range(1,len(o.pans)):
   a=j*pp
   if a>=n:
    break
   pj=o.pans[j]
   e=min(a+pj.shape[0],n)
   torch.addmm(y[a:e],pj[:,:n-a],dh[a:n],out_dtype=F32,out=y[a:e])
   if e<n:
    torch.addmm(y[e:n],pj[:,e-a:n-a].T,dh[a:e],out_dtype=F32,out=y[e:n])
  return y
 def cross(o,w,lo,hi):
  s=w.abs().amax(0,keepdim=True).clamp_min(1e-30)
  wh=(w/s).half()
  y=torch.empty(hi-lo,w.shape[1],device=w.device)
  for j,pj in enumerate(o.pans):
   a=j*o.pp
   if a>=lo:
    break
   if j==0:
    torch.mm(pj[:,lo-a:hi-a].T,wh[a:a+pj.shape[0]],out_dtype=F32,out=y)
   else:
    torch.addmm(y,pj[:,lo-a:hi-a].T,wh[a:a+pj.shape[0]],out_dtype=F32,out=y)
  return y*s
def upd_comp(ap,d,w,r,nrm,v,vt):
 a=nrm/(d*ap).sum(0)
 w=w+d*a
 r=r-ap*a
 p=r-v@(vt@r)
 nxt=(r*p).sum(0)
 d=p+d*(nxt/nrm)
 return w,r,d,nxt,d.half()
def lfac(kmm,jit):
 lm=kmm.clone()
 lm.diagonal().add_(jit)
 eye=torch.eye(len(kmm),device=kmm.device)
 return torch.linalg.solve_triangular(chol(lm),eye,upper=False).T
def pcg(op,t,lam,steps,u):
 small=u.T@u
 small.diagonal().add_(lam)
 eye=torch.eye(len(small),device=u.device)
 v=u@torch.linalg.solve_triangular(chol(small),eye,upper=False).T
 vt=v.T.contiguous()
 w=torch.zeros_like(t)
 r=t.clone()
 d=torch.addmm(r,v,vt@r,alpha=-1)
 nrm=torch.linalg.vecdot(r,d,dim=0)
 dh=d.half()
 for _ in range(steps):
  w,r,d,nrm,dh=upd_comp(op.mv(dh),d,w,r,nrm,v,vt)
 return w
def sqrtm(a):
 c=a.norm()
 n=len(a)
 x=torch.zeros(2*n,2*n,device=a.device)
 x[:n,n:]=a/c
 x[n:,:n]=torch.eye(n,device=a.device)
 i15=1.5*torch.eye(2*n,device=a.device)
 for _ in range(NS_ITERS):
  x=x@torch.addmm(i15,x,x,alpha=-0.5)
 return x[:n,n:]*c.sqrt()
def agop(transform,z,w,fac):
 s=fac.shape[0]
 wp=torch.cat([w,(w[:,:,None]*z[:,None,:]).reshape(len(z),-1)],1)
 r=fac.float()@wp
 c=w.shape[1]
 g=r[:,c:].reshape(s,c,-1)-r[:,:c,None]*z[:s,None,:]
 dim=transform.shape[0]
 g=(g@transform.T).reshape(-1,dim)
 sq=sqrtm(g.T@g)
 def fin(m):
  m=(m+m.T)/2
  m=m*(dim/m.trace().clamp_min(1e-12))
  return chol((1-BLEND)*m+BLEND*torch.eye(dim,device=z.device))
 return fin(sq@sqrtm(sq)),fin(sq)
def sym_scores(train_x,train_y,test_x):
 c=math.exp(SHIFT)
 lam,jit,diag=RIDGE*c,0.0001*c,c*(1.0+RIDGE)
 with torch.no_grad():
  x,q=rownorm(train_x.float()),rownorm(test_x.float())
  n=len(x)
  y=F.one_hot(train_y.long(),10).float()
  y=torch.cat([y,y[:,:6]],1)
  transform=torch.eye(x.shape[1],device=x.device)
  for m,st in zip(AGOP_N,M_STEPS):
   z=x[:m]@transform
   km=kern(z,z[:RANK])*c
   pm=P if m%P==0 else m
   A3,B3=h3_ops(z)
   pan=panels(A3,B3,pm,range(0,m,pm),diag)
   w=pcg(Sym(pan,pm,m),y[:m],lam,st,km@lfac(km[:RANK],jit))
   fac=pan[0][:SAMPLES].clone()
   fac.diagonal().fill_(c)
   del pan
   transform,sf=agop(transform,z,w[:,:10],fac)
  za=torch.cat([x,q])@transform
  N=len(za)
  A3,B3=h3_ops(za)
  pan=panels(A3,B3,P,range(0,n,P),diag)
  km=kern(za,za[:RANK])*c
  opx=Sym(pan,P,n)
  return opx.cross(pcg(opx,y,lam,BSTEPS,km[:n]@lfac(km[:RANK],jit)),n,N)[:,:10],sf
def teacher(train_x,train_y,test_x):
 key='teacher',train_x.shape,test_x.shape
 if key not in CACHE:
  static=[train_x.clone(),train_y.clone(),test_x.clone()]
  side=torch.cuda.Stream()
  side.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side):
   for _ in range(2):
    sym_scores(*static)
  torch.cuda.current_stream().wait_stream(side)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   out=sym_scores(*static)
  CACHE[key]=g,static,out
 g,static,out=CACHE[key]
 static[0].copy_(train_x)
 static[1].copy_(train_y)
 static[2].copy_(test_x)
 g.replay()
 return out[0].clone(),out[1].clone()
def hknn(zq,zv,yv,valid,cls):
 nq,C=cls.shape
 cd,ci=torch.mm(*aug(zq,zv)).topk(HK_M,1)
 ok=valid[ci][:,None,:]&(yv[ci][:,None,:]==cls[:,:,None])
 sv,si=cd[:,None,:].masked_fill(~ok,-float('inf')).topk(HK_K,2)
 nb=ci.gather(1,si.reshape(nq,-1)).reshape(nq,C,HK_K)
 bad=torch.isinf(sv)
 v=zv[torch.where(bad,nb[:,:,:1],nb)].reshape(nq*C,HK_K,-1)
 m=v.mean(1,keepdim=True)
 v=v-m
 r=(zq.repeat_interleave(C,0)[:,None]-m).mT
 g=torch.baddbmm(torch.eye(HK_K,device=zq.device),v,v.mT)
 return (r-v.mT@torch.cholesky_solve(v@r,chol(g))).mT.square().sum(-1).view(nq,C).masked_fill(bad[:,:,0],1e30)
def hk_fix(s,za,ty):
 n=len(ty)
 marg=-s.topk(2).values.diff()[:,0]
 keep=marg>=HK_CF
 uidx=marg.topk(HK_N,largest=False).indices
 cls=s[uidx].topk(HK_C).indices
 d=hknn(za[n:][uidx],za,torch.cat([ty,s.argmax(1)]),torch.cat([keep.new_ones(n),keep]),cls)
 pr=torch.softmax((d.min(1,keepdim=True).values-d)/d.min(1).values.mean(),1)
 return s.index_add(0,uidx,torch.zeros_like(s[uidx]).scatter_(1,cls,pr)*(~keep[uidx]).float()[:,None])
@torch.no_grad()
def tangents(x,y):
 sim=x@x.T-0.5*x.square().sum(1)
 sim.masked_fill_(y[:,None]!=y[None],-float('inf')).fill_diagonal_(-float('inf'))
 d=x[sim.topk(TAN_K,dim=1).indices]-x[:,None]
 v=torch.linalg.eigh(d@d.mT)[1][...,-TAN_R:]
 return d.mT@v/TAN_K**0.5
def bn(u):
 var,mean=torch.var_mean(u,dim=-2,unbiased=False,keepdim=True)
 return (u-mean)*torch.rsqrt(var+EPS)
def lrelu(t):
 return F.leaky_relu(t,0.1)
def combine(u,lat,p):
 return comb_math(bn(u),lat,p)
def comb_math(v,lat,p):
 x2=v*lat
 a0=lrelu(v*p[0]+lat*p[2]+x2*p[4]+p[6])
 a1=lrelu(v*p[1]+lat*p[3]+x2*p[5]+p[7])
 c0=lrelu(a0*p[8]+a1*p[10]+p[12])
 c1=lrelu(a0*p[9]+a1*p[11]+p[13])
 return c0*p[14]+c1*p[15]+p[16]
@triton.jit
def _ld(p,o,m):
 return tl.load(p+o,mask=m,other=0.0)
@triton.jit
def _s(w,a,f=1.0):
 tl.store(w,tl.sum(a,0)*f)
@triton.jit
def _p(P,j,S,c,cm):
 return tl.load(P+j*S+c,mask=cm,other=0.0).to(FP)[None,:]
@triton.jit
def comb_bwd_kernel(GO,U,MEAN,RSTD,LAT,P,DU,DLAT,DP,WS,DV,B,S,ROWS,NRS,NCB:tl.constexpr,BC:tl.constexpr,SUB:tl.constexpr,SB:tl.constexpr,PHASE:tl.constexpr):
 pid=tl.program_id(0)
 cb=pid%NCB
 rs=pid//NCB
 c=cb*BC+tl.arange(0,BC)
 cm=c<S
 r0=rs*ROWS
 r1=tl.minimum(r0+ROWS,B)
 rr=tl.arange(0,SUB)
 NPROG=NCB*NRS
 invB=1.0/B.to(FP)
 mean=tl.load(MEAN+c).to(FP)
 rstd=tl.load(RSTD+c).to(FP)
 rs2=rstd[None,:]
 nm2=(-mean*rstd)[None,:]
 p0=_p(P,0,S,c,cm)
 p1=_p(P,1,S,c,cm)
 p2=_p(P,2,S,c,cm)
 p3=_p(P,3,S,c,cm)
 p4=_p(P,4,S,c,cm)
 p5=_p(P,5,S,c,cm)
 p6=_p(P,6,S,c,cm)
 p7=_p(P,7,S,c,cm)
 p8=_p(P,8,S,c,cm)
 p9=_p(P,9,S,c,cm)
 p10=_p(P,10,S,c,cm)
 p11=_p(P,11,S,c,cm)
 p12=_p(P,12,S,c,cm)
 p13=_p(P,13,S,c,cm)
 p14=tl.load(P+14*S+c).to(FP)
 p15=tl.load(P+15*S+c).to(FP)
 q0=p14[None,:]*p8
 q1=p15[None,:]*p9
 q2=p14[None,:]*p10
 q3=p15[None,:]*p11
 if PHASE==1:
  z=tl.zeros([SUB,BC],FP)
  a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19=z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z
  rows=r0+rr
  m=(rows<r1)[:,None]
  o=rows[:,None]*S+c[None,:]
  gn=_ld(GO,o,m)
  un=_ld(U,o,m)
  ln=_ld(LAT,o,m)
  for r in range(r0,r1,SUB):
   go=gn.to(FP)
   u=un.to(FP)
   lat=ln.to(FP)
   mc,oc=m,o
   rows=r+SUB+rr
   m=(rows<r1)[:,None]
   o=rows[:,None]*S+c[None,:]
   gn=_ld(GO,o,m)
   un=_ld(U,o,m)
   ln=_ld(LAT,o,m)
   v=u*rs2+nm2
   x2=v*lat
   pa0=v*p0+lat*p2+x2*p4+p6
   pa1=v*p1+lat*p3+x2*p5+p7
   k0=tl.where(pa0>0,1.0,0.1)
   k1=tl.where(pa1>0,1.0,0.1)
   h0=pa0*k0
   h1=pa1*k1
   pc0=h0*p8+h1*p10+p12
   pc1=h0*p9+h1*p11+p13
   gj0=go*tl.where(pc0>0,1.0,0.1)
   gj1=go*tl.where(pc1>0,1.0,0.1)
   ga0=(gj0*q0+gj1*q1)*k0
   ga1=(gj0*q2+gj1*q3)*k1
   dx2=ga0*p4+ga1*p5
   dv=ga0*p0+ga1*p1+dx2*lat
   dlat=ga0*p2+ga1*p3+dx2*v
   tl.store(DLAT+oc,dlat.to(BF),mask=mc)
   tl.store(DV+oc,dv.to(BF),mask=mc)
   a0+=ga0*v
   a1+=ga1*v
   a2+=ga0*lat
   a3+=ga1*lat
   a4+=ga0*x2
   a5+=ga1*x2
   a6+=ga0
   a7+=ga1
   a8+=gj0*h0
   a9+=gj1*h0
   a10+=gj0*h1
   a11+=gj1*h1
   a12+=gj0
   a13+=gj1
   a14+=gj0*pc0
   a15+=gj1*pc1
   a16+=go
   a17+=dv
   a18+=dv*v
   a19+=tl.where(mc,v,0.0)
  w=WS+rs*20*S+c
  _s(w,a0)
  _s(w+S,a1)
  _s(w+2*S,a2)
  _s(w+3*S,a3)
  _s(w+4*S,a4)
  _s(w+5*S,a5)
  _s(w+6*S,a6)
  _s(w+7*S,a7)
  _s(w+8*S,a8,p14)
  _s(w+9*S,a9,p15)
  _s(w+10*S,a10,p14)
  _s(w+11*S,a11,p15)
  _s(w+12*S,a12,p14)
  _s(w+13*S,a13,p15)
  _s(w+14*S,a14)
  _s(w+15*S,a15)
  _s(w+16*S,a16)
  _s(w+17*S,a17)
  _s(w+18*S,a18)
  _s(w+19*S,a19)
 else:
  CH=tl.cdiv(17*S,NPROG)
  e_hi=tl.minimum((pid+1)*CH,17*S)
  for e0 in range(pid*CH,e_hi,256):
   e=e0+tl.arange(0,256)
   em=e<e_hi
   acc=tl.zeros([256],FP)
   for k in range(NRS):
    acc+=tl.load(WS+k*20*S+e,mask=em,other=0.0,cache_modifier='.cg')
   tl.store(DP+e,acc,mask=em)
  s1=tl.zeros([BC],FP)
  s2=tl.zeros([BC],FP)
  s3=tl.zeros([BC],FP)
  for k in range(NRS):
   s1+=tl.load(WS+(k*20+17)*S+c,cache_modifier='.cg')
   s2+=tl.load(WS+(k*20+18)*S+c,cache_modifier='.cg')
   s3+=tl.load(WS+(k*20+19)*S+c,cache_modifier='.cg')
  mdv=(s1*invB)[None,:]
  mdvv=(s2*invB)[None,:]
  mv=(s3*invB)[None,:]
  rb=tl.arange(0,SB)
  rows=r0+rb
  m=(rows<r1)[:,None]
  o=rows[:,None]*S+c[None,:]
  un=tl.load(U+o,mask=m,other=0.0)
  dn=tl.load(DV+o,mask=m,other=0.0)
  for r in range(r0,r1,SB):
   u=un.to(FP)
   dv=dn.to(FP)
   mc,oc=m,o
   rows=r+SB+rb
   m=(rows<r1)[:,None]
   o=rows[:,None]*S+c[None,:]
   un=tl.load(U+o,mask=m,other=0.0)
   dn=tl.load(DV+o,mask=m,other=0.0)
   v=u*rs2+nm2
   tl.store(DU+oc,(rs2*(dv-mdv-(v-mv)*mdvv)).to(BF),mask=mc)
KCFG={1024:(16,16,4,4,64),768:(16,16,4,4,64),512:(16,16,4,3,64),256:(16,16,2,4,64)}
def comb_bwd(go:T,u:T,mean:T,rstd:T,lat:T,pb:T)->tuple[T,T,T]:
 B,S=u.shape[-2:]
 BC,SUB,W,k,SB=KCFG[S]
 n=S//BC
 rows=-(-B//max(1,(torch.cuda.get_device_properties(u.device).multi_processor_count*k-(4 if k>1 else 0))//n))
 rows=-(-rows//SB)*SB
 nrs=-(-B//rows)
 go,u,mean,rstd,pb=go.contiguous(),u.contiguous(),mean.contiguous(),rstd.contiguous(),pb.contiguous()
 if lat.stride(-1)!=1 or lat.stride(-2)!=S:
  lat=lat.contiguous()
 e=lambda *s,dtype=ACT:torch.empty(*s,device=u.device,dtype=dtype)
 du,dlat,dp,ws,dv=e(u.shape),e(u.shape),e(pb.shape,dtype=F32),e(nrs,20,S,dtype=F32),e(u.shape)
 for ph in (1,2):
  comb_bwd_kernel[n*nrs,](go,u,mean,rstd,lat,pb,du,dlat,dp,ws,dv,B,S,rows,nrs,NCB=n,BC=BC,SUB=SUB,SB=SB,PHASE=ph,num_warps=W)
 return du,dlat,dp
def register():
 if not hasattr(torch.ops.d5fused,'comb_bwd'):
  e=lambda t,d:t.new_empty(t.shape,dtype=d)
  torch.library.custom_op('d5fused::comb_bwd',comb_bwd,mutates_args=()).register_fake(lambda go,u,mean,rstd,lat,pb:(e(u,ACT),e(u,ACT),e(pb,F32)))
class CombFn(torch.autograd.Function):
 @staticmethod
 def forward(ctx,u,lat,p):
  pb=p.to(ACT)
  var,mean=torch.var_mean(u,dim=-2,unbiased=False,keepdim=True)
  rstd=torch.rsqrt(var+EPS)
  out=comb_math((u-mean)*rstd,lat,pb)
  ctx.save_for_backward(u,mean,rstd,lat,pb)
  return out
 @staticmethod
 def backward(ctx,go):
  u,mean,rstd,lat,pb=ctx.saved_tensors
  du,dlat,dp=torch.ops.d5fused.comb_bwd(go,u,mean,rstd,lat,pb)
  return du,dlat,dp
def mm(h,w,act=None):
 s=h.shape
 with torch.autocast('cuda',dtype=act or torch.float16,enabled=act is not None):
  out=torch.bmm(h.reshape(s[0],-1,s[-1]),w)
 return (out if act is not None else out.float()).reshape(*s[:-1],w.shape[-1])
class Ladder:
 def __init__(o,n,nq,device):
  o.n,o.nq,o.device=n,nq,device
  zeros=lambda *s:torch.zeros(*s,device=device)
  o.enc=[zeros(M,a,b) for a,b in zip(DIMS[:-1],DIMS[1:])]
  o.dec=[zeros(M,b,a) for a,b in zip(DIMS[:-1],DIMS[1:])]
  o.beta=[zeros(M,1,b) for b in DIMS[1:]]
  o.gamma=zeros(M,1,DIMS[-1])
  o.comb=[zeros(17,M,1,s) for s in DIMS]
  o.params=o.enc+o.dec+o.beta+[o.gamma]+o.comb
  for p in o.params:
   p.requires_grad_(True)
  o.x=zeros(n,DIMS[0])
  o.y=zeros(n,10)
  o.pool=zeros(n+nq,DIMS[0])
  o.ptar=zeros(n+nq,10)
  o.tan=zeros(n,DIMS[0],TAN_R)
  o.scores=zeros(M,n)
  o.offs=(torch.arange(M,device=device)*n)[:,None]
  o.counter=torch.zeros((),dtype=torch.long,device=device)
  o.iu=torch.zeros(STEPS,M,BATCH,dtype=torch.long,device=device)
  frac=torch.arange(STEPS,device=device,dtype=torch.float32)/STEPS
  o.lrs=LR*((1-frac)/(1-DECAY_START)).clamp(max=1.0)
  ramp=((MIX_C-frac)/(MIX_C-MIX_B)).clamp(0,1)
  o.mixs=MIX*torch.where(frac<MIX_A,0.0,torch.where(frac<MIX_B,1.0,ramp))
  o.lr=torch.tensor(LR,device=device)
  o.hard=torch.zeros(int(UF*nq),dtype=torch.long,device=device)
  o.bank=zeros(NB).to(ACT)
  o.noff=zeros(STEPS,len(DIMS)).long()
  o.opt=torch.optim.Adam(o.params,lr=o.lr,betas=(BETA1,0.999),foreach=True,capturable=True)
  o.ostep=torch.compile(o.opt.step)
  o.lossfn=torch.compile(o.fwd,dynamic=False,options=IND)
  o.reset()
  side=torch.cuda.Stream()
  side.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side):
   for _ in range(3):
    o.opt.zero_grad(set_to_none=True)
    o.step()
  torch.cuda.current_stream().wait_stream(side)
  o.graph=torch.cuda.CUDAGraph()
  o.opt.zero_grad(set_to_none=True)
  with torch.cuda.graph(o.graph):
   for _ in range(GK):
    for p in o.params:
     p.grad=None
    o.step()
 def loss(o,xl,yl,xu,tu,tg,ofs,lb):
  ofs=list(ofs.unbind())
  nz=lambda t:o.bank[(ofs.pop()+torch.arange(t.numel(),device=t.device))%NB].view(t.shape).to(t.dtype)
  beta,gamma=[b.to(ACT) for b in o.beta],o.gamma.to(ACT)
  comb=[p.to(ACT) for p in o.comb]
  xl=xl+TAN*(tg*torch.randn_like(tg[...,:1,:])).sum(-1)
  h=torch.stack((xl,xu),1)
  h=h+NOISE*nz(h)
  lat=[h[:,1]]
  h=h.to(ACT)
  top=len(o.enc)-1
  for i,w in enumerate(o.enc):
   z=bn(mm(h,w,ACT))
   z=z+NOISE*nz(z)
   lat.append(z[:,1])
   b=beta[i][:,None]
   h=(z+b)*gamma[:,None] if i==top else F.dropout(F.relu(z+b),0.1,True)
  hf=h.float()
  ce=-(yl*F.log_softmax(hf[:,0],-1)).sum(-1)
  teach=-(tu*F.log_softmax(hf[:,1],-1)).sum(-1).mean(-1).sum()
  v=F.softmax(hf[:,1],-1);r=combine((v+(lb*(tu-v)).detach()).to(ACT),lat[-1],comb[-1])
  for i in range(top,0,-1):
   r=CombFn.apply(mm(r,o.dec[i],ACT),lat[i],o.comb[i])
  r=combine(mm(r.float(),o.dec[0]),lat[0],o.comb[0])
  rec=(r.float()-xu).square().mean((1,2)).sum()
  return ce.mean(-1).sum()+TW*teach+RECON*rec,ce.detach()
 def fwd(o):
  at=o.counter.view(1)
  m=o.mixs.index_select(0,at)
  c=(o.scores/o.scores.sum(-1,keepdim=True)*m+(1-m)/o.n).cumsum(-1)
  il=torch.searchsorted(c,torch.rand(M,BATCH,device=o.device)*c[:,-1:]).clamp_max(o.n-1)
  iu=o.iu.index_select(0,at).squeeze(0)
  o.lr.copy_(o.lrs.index_select(0,at).squeeze(0))
  tg=o.tan[il]
  loss,ce=o.loss(o.x[il],o.y[il],o.pool[iu],o.ptar[iu],tg,o.noff.index_select(0,at).squeeze(0),(iu<o.n)[...,None])
  o.scores.view(-1).index_copy_(0,(il+o.offs).view(-1),ce.view(-1))
  o.counter.add_(1)
  return loss
 def step(o):
  o.lossfn().backward()
  o.ostep()
 @torch.no_grad()
 def reset(o):
  g=torch.Generator(device=o.device)
  g.manual_seed(SEED)
  for w in o.enc+o.dec:
   w.copy_(torch.randn(w.shape,generator=g,device=o.device)*w.shape[1]**(-0.5))
  rows=[0,1,2,3,4,5,8,9,10,11,14,15]
  for p in o.comb:
   p.zero_()
   p[rows]=torch.randn(len(rows),*p.shape[1:],generator=g,device=o.device)*CSTD
  for p in o.beta:
   p.zero_()
  o.gamma.fill_(1.0)
  for p in o.params:
   st=o.opt.state.get(p)
   if st:
    st['exp_avg'].zero_()
    st['exp_avg_sq'].zero_()
    st['step'].zero_()
  o.counter.zero_()
  o.scores.fill_(1.0)
  ids=torch.arange(o.n+o.nq,device=o.device)
  iu=[]
  for t,rows in ((STEPS//2,ids),(STEPS-STEPS//2,torch.cat([ids,o.hard]))):
   u=torch.rand(math.ceil(t*BATCH/len(rows)),len(rows),generator=g,device=o.device).argsort(-1)
   iu.append(rows[u.view(-1)[:t*BATCH]].view(t,1,BATCH))
  o.iu.copy_(torch.cat(iu))
  o.bank.copy_(torch.randn(NB,generator=g,device=o.device))
  o.noff.copy_(torch.randint(NB,o.noff.shape,generator=g,device=o.device))
  torch.cuda.manual_seed(SEED+1)
 def act(o,i,t,b,gm):
  return (t+b)*gm if i==len(o.enc)-1 else F.relu(t+b)
 @torch.no_grad()
 def predict(o,q):
  hs=[]
  g=torch.Generator(device=o.device)
  g.manual_seed(1)
  nb=o.n//BATCH
  h=o.x[torch.randperm(o.n,generator=g,device=o.device)[:nb*BATCH]]
  h=h.reshape(1,nb,BATCH,-1).expand(M,-1,-1,-1)
  q=q.expand(M,-1,-1)
  for i,w in enumerate(o.enc):
   raw=mm(h,w)
   var,mean=torch.var_mean(raw,dim=2,unbiased=False,keepdim=True)
   h=o.act(i,(raw-mean)*torch.rsqrt(var+EPS),o.beta[i][:,None],o.gamma[:,None])
   mu,vv=mean.mean(1),var.mean(1)*BATCH/(BATCH-1)
   z=(mm(q,w)-mu)*torch.rsqrt(vv+EPS)
   hs.append((z+o.beta[i])[0])
   q=o.act(i,z,o.beta[i],o.gamma)
  return hs+[q[0]]
GUARD_T,GUARD_E=60,40
def guard(s,p,e):
 v,t=s.topk(2,1)
 t=t[:,0]
 k=v[:,0]-v[:,1]>=PSEUDO_CONF
 c=torch.zeros(100,device=s.device).index_add_(0,p*10+t,k.float())
 c[::11]=0
 return torch.where(k & ((c.max()>GUARD_T) | (e>GUARD_E)),t,p)
def _predict(train_x,train_y,test_x):
 os.environ.setdefault('TRITON_CACHE_DIR','/tmp/triton-cache')
 os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/inductor-cache')
 register()
 torch.backends.cuda.matmul.allow_tf32=False
 s,transform=teacher(train_x,train_y,test_x)
 torch.backends.cuda.matmul.allow_tf32=True
 with torch.no_grad():
  x,q=rownorm(train_x.float())@transform,rownorm(test_x.float())@transform
  c=(train_x.float()*SCALE).std()/x.std()
  x,q=x*c,q*c
 key='ladder',len(x),len(q)
 if key not in CACHE:
  CACHE[key]=Ladder(x.shape[0],q.shape[0],x.device)
 net=CACHE[key]
 with torch.no_grad():
  net.x.copy_(x)
  net.tan.copy_(tangents(x,train_y))
  net.y.copy_(F.one_hot(train_y.long(),10).float())
  net.pool.copy_(torch.cat((x,q)))
  net.ptar.copy_(torch.cat((net.y,F.softmax(s/TEACH_TAU,1))))
  top=s.topk(2,1).values
  net.hard.copy_((top[:,0]-top[:,1]).topk(len(net.hard),largest=False).indices+len(x))
 net.reset()
 for _ in range(STEPS//GK):
  net.graph.replay()
 with torch.no_grad():
  hq,hx=net.predict(q),net.predict(x)
  e=(hx[-1].argmax(1)!=train_y).sum()
  za=torch.cat([x,q])
  hz=torch.cat([hx[2],hq[2]]).float()
  hz=hz*za.norm()/hz.norm()
 torch.backends.cuda.matmul.allow_tf32=False
 p=hk_fix(F.softmax(hq[-1],-1),hz,train_y.long())
 p=p/p.sum(1,keepdim=True)
 return p,e
