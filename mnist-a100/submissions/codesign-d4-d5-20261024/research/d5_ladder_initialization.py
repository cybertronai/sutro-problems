"""Transpose decoder and near-identity denoiser initialization, inspired byPR127.

Retain every original RNG draw before replacing initial values so sampling and
corruption tables stay paired. Parameters train independently after reset.
"""
def transform(source,tied=False,affine=False):
 assert tied or affine
 if tied:
  old='  rows=[0,1,2,3,4,5,8,9,10,11,14,15]';assert source.count(old)==1
  source=source.replace(old,'  for a,b in zip(o.enc,o.dec):b.copy_(a.mT*math.sqrt(a.shape[1]/a.shape[2]))\n'+old)
 if affine:
  old='  o.reset()';assert source.count(old)==1
  source=source.replace(old,'  o.cinit=torch.tensor([.1,-.1,.9,-.9,0,0,0,0,1,0,0,1,0,0,1/1.01,-1/1.01,0],device=device)[:,None,None,None]\n'+old)
  old='   p[rows]=torch.randn(len(rows),*p.shape[1:],generator=g,device=o.device)*CSTD';assert source.count(old)==1
  source=source.replace(old,old+'\n   p.copy_(o.cinit)')
 return source
