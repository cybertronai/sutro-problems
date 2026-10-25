"""Compact class-conditional decoder readout for standalone integration."""
import torch
from kernels.d5_decoder_readout import comb as comb_math

def decoder(r,latent,weights,params,stats=None):
 out=[]
 for j,i in enumerate(range(4,-1,-1)):
  if i<4:r=r@weights[i]
  if stats is None:
   var,mean=torch.var_mean(r,dim=0,unbiased=False,keepdim=True)
   inv=torch.rsqrt(var+1e-10)
  else:mean,inv=stats[j]
  out.append((mean,inv));r=comb_math((r-mean)*inv,latent[i],params[i])
 return r,out

def transform(source,alpha=.5):
 old='  o.lossfn=torch.compile(o.fwd,dynamic=False,options=IND)';assert source.count(old)==1
 source=source.replace(old,old+'\n  o.read=torch.compile(decoder,dynamic=False)')
 line=next(s for s in source.splitlines()if s.startswith(' return guard(s,hk_fix('))
 score=line[len(' return guard(s,hk_fix('):line.index(',hz,train_y.long())')]
 old=' torch.backends.cuda.matmul.allow_tf32=False\n'+line
 assert source.count(old)==1
 block=f'''  torch.backends.cuda.matmul.allow_tf32=False
  p={score}
  latent=[net.pool]+[h-net.beta[i][0]for i,h in enumerate(both[:-1])]
  weights=[w[0]for w in net.dec];params=[v[:,0]for v in net.comb]
  _,stats=net.read(F.softmax(both[-1][:len(x)],-1),[v[:len(x)]for v in latent],weights,params)
  idx=(-p.topk(2).values.diff()[:,0]).topk(1024,largest=False).indices
  cls=p[idx].topk(3).indices
  lat=[v[len(x):][idx,None].expand(-1,3,-1).reshape(-1,v.shape[-1])for v in latent]
  r,_=net.read(F.one_hot(cls.flatten(),10).float(),lat,weights,params,stats)
  r=r.reshape(1024,3,60);a=q[idx,None]
  err=(r-a).square().mean(-1);dot=(r*a).sum(-1)
  scale=(err[:,1]-err[:,0]).abs().median().clamp_min(1e-9)
  dscale=(dot[:,1]-dot[:,0]).abs().median().clamp_min(1e-9)
  post=(-.5*err/scale+.5*dot/dscale).softmax(-1)
  p[idx]*=1-{alpha};p[idx[:,None],cls]+={alpha}*post
 return guard(s,hk_fix(p,hz,train_y.long()).argmax(1),e)'''
 source=source.replace(old,block)
 return 'from kernels.d5_decoder_readout_integrated import decoder\n'+source
