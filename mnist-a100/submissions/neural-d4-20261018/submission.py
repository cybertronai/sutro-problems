from triton import jit as ad;from torch import cuda as i,bfloat16 as G,float32 as av,long as R,no_grad as o,empty as q,cat as L,arange as T,zeros as Y,zeros_like as aQ,eye as A,randn as aS,linalg as B,autocast as Z,searchsorted as aV,empty_like as aX,linspace as aY;import math;import torch as aa,torch;import torch.nn.functional as F;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;f,aF,l,a=.02,.1,.02,.6;P,aI=2000,3.
def aw(a):return B.cholesky_ex(a)[0]
def t(x):return x*(x.shape[1]**.5/x.norm(dim=1,keepdim=True).clamp_min(1e-6))**a
def _b(a,b,e=.0):c=a.square().sum(1,keepdim=True);d=b.square().sum(1,keepdim=True);A=L([a,c,aa.ones_like(c)],1);B=L([2*f*b,aa.full_like(d,-f),e-f*d],1).T.contiguous();return A,B
def _c(a,b):A,B=_b(a,b);return(A@B).exp_()
I=True;z=False;import triton as ab;import triton.language as tl;aG=2048;torch.backends.cuda.matmul.allow_tf32=True;E,W,J,D,C=4,256,3,60,10;at,BL,BU=196,2048,512;ar,aB,u,X=.07,.0175,.914147570219414,.1;ay,c,d=.998,2.,2;aJ,U=1.,10;ax=.9;aA,v,au,al,az,aE,ap,aq=5,5,20.,.9,3,.25,.3,10;_S={};m=.5;w=W//2;from torch.cuda import current_stream as zc
def g(value):aa.backends.cuda.matmul.allow_tf32=value
@ad
def M(p,m=None,o=None):return tl.load(p,m,o)
@ad
def S(p,v,m=None):tl.store(p,v,m)
def k(a,b=7):
 c=a.norm();n=len(a);t=Y(2*n,2*n,device=a.device);t[:n,n:]=a/c;t[n:,:n]=A(n,device=a.device);i=1.5*A(2*n,device=a.device)
 for _ in range(b):t=t@aa.addmm(i,t,t,alpha=-.5)
 return t[:n,n:]*c.sqrt()
def _h(i,z,w,a,j,h):
 s=len(a);f=L([w,(w[:,:,None]*z[:,None,:]).reshape(len(z),-1)],1);d=f.abs().amax(0,keepdim=I).clamp_min(1e-30);r=aa.mm(a,(f/d).half(),out_dtype=av)*d;c=w.shape[1];g=r[:,c:].reshape(s,c,-1)-r[:,:c,None]*h[:,None,:];g=(g@i.T).reshape(-1,z.shape[1]);b=g.T@g
 def factor(m):m=(m+m.T)/2;m=m*(z.shape[1]/m.trace().clamp_min(1e-12));return aw((1-l)*m+l*A(z.shape[1],device=z.device))
 if j:return factor(k(b,15))
 e=k(b);return factor(e@k(e))
def _i(x,y):c=math.exp(aI);k,o=aF*c,.0001*c;e=t(x.float());m=F.one_hot(y,10).float();i=e[:r];b=_c(e,i)*c;h=(b[:r]+b[:r].T)*.5;h.diagonal().add_(o);aw,q=B.cholesky_ex(h);g=B.solve_triangular(aw,A(r,device=x.device),upper=False).T;d=b@g;a=d.T@d;a=(a+a.T)*.5;a.diagonal().add_(k);j,p=B.cholesky_ex(a);n=g@aa.cholesky_solve(d.T@m,j);_S['coarse_info']=aa.stack([q,p]);f=e[:aG];l=b[:len(f)].half();return _h(A(x.shape[1],device=x.device),i,n,l,True,f)
def _j(x):d=x.shape[1];b=x.T@(x.square().sum(1,True)*x)/len(x);a=aa.cholesky_inverse(B.cholesky_ex(b)[0]);a=a*(d/a.trace());return B.cholesky_ex(.98*a+.02*A(d,device=x.device))[0]
def _k(x,y,q):g(z);a=_j(x);c,e=H(x)@a,H(q)@a;b=_i(c,y);_S['f']=None,None,(a,b);d=t(c)@b,t(e)@b;g(I);return d
@ad
def s(A,B,O,U,M:'constexpr',N:'constexpr',K:'constexpr',e,X,Y,Z,J,a:'constexpr'):
 g:tl.constexpr=a[9];q:tl.constexpr=64 if g==0 else 16 if g==2 else 32;x:tl.constexpr=a[12];b=ae(0)*q+af(0,q);c=af(0,x);rk=af(0,32)
 if g==0:H=ae(2);d=H%a[7];T=H//a[7];c=c+ae(1)*x
 else:
  d=ae(1);T=0
 if g==2:m=tl.load(e+d).to(tl.int32);hi=tl.load(e+d+1).to(tl.int32);f=hi-m;o=f;s=tl.cdiv(f,512);s=ac(ae(0)*q<f,s,0);A=A+m*K;B=B+m*K;d=0;R=tl.load(e+11+M+m+b,b<f,0).to(tl.int32)
 else:
  f=M;o=N;s=1
 di=tl.cdiv(K,32*a[6])*32
 for Q in range(s):
  h=c+Q*512;i=tl.full((q,x),0,tl.float32)
  for j in range(tl.cdiv(K,32*a[6])):
   l=T*di+j*32+rk;D=tl.load(A+d*a[0]+b[:,None]*a[1]+l[None,:]*a[2],(b[:,None]<f)&(l[None,:]<K),0)
   if a[8]:a2=tl.load(e+d*a[0]+b[:,None]*a[1]+l[None,:]*a[2],(b[:,None]<f)&(l[None,:]<K),0);D=(-4.775*D.to(tl.float32)+2.0315*a2.to(tl.float32)).to(tl.bfloat16)
   bb=tl.load(B+d*a[3]+l[:,None]*a[4]+h[None,:]*a[5],(l[:,None]<K)&(h[None,:]<o),0);i+=tl.dot(D,bb,input_precision='tf32x3'if g==2 or A.dtype.element_ty==tl.float32 else'tf32')
  if g==0:
   if a[8]:bO=tl.load(B+d*a[3]+b[:,None]*a[4]+c[None,:]*a[5],(b[:,None]<M)&(c[None,:]<N),0).to(tl.float32);i=3.4445*bO+i.to(tl.bfloat16).to(tl.float32)
   tl.store(O+H*M*N+b[:,None]*N+c[None,:],i,(b[:,None]<M)&(c[None,:]<N))
  elif g==1:
   L:tl.constexpr=a[10];G:tl.constexpr=a[11];u=_C(i)
   if L==0:
    u=u+.5
    if not G:u=_C(u)
   r=libdevice.rsqrt(tl.sum(u*u,1)/N+1e-6)
   if not G:r=_C(r)
   v=u*r[:,None];p=tl.maximum(v,.0);n=tl.maximum(-v,.0)
   if G:p=_C(p);n=_C(n);t=tl.load(e);I=(d*M+b[:,None])*(2*N)+c[None,:];p=ac(_D(I,t,L),p/.9,.0);n=ac(_D(I+N,t,L),n/.9,.0);tl.store(U+d*M*N+b[:,None]*N+c[None,:],u,(b[:,None]<M)&(c[None,:]<N))
   tl.store(O+d*M*2*N+b[:,None]*2*N+c[None,:],p,(b[:,None]<M)&(c[None,:]<N));tl.store(O+d*M*2*N+b[:,None]*2*N+c[None,:]+N,n,(b[:,None]<M)&(c[None,:]<N))
  elif g==2:
   E=tl.load(e+11+M+m+h,h<o,0).to(tl.int32);bW=tl.load(e+11+m+h,h<o,.0);k=i-.5*bW[None,:];k=ac((h[None,:]<o)&(R[:,None]!=E[None,:]),k,-float('inf'))
   for dA in range(10):value=tl.max(k,1);W=tl.min(ac(k==value[:,None],E[None,:],2147483647),1);S=R*tl.cdiv(M,512)*10+Q*10+dA;tl.store(O+S,value,b<f);tl.store(O+M*tl.cdiv(M,512)*10+S,W.to(tl.float32),b<f);k=ac(E[None,:]==W[:,None],-float('inf'),k)
  else:
   BL:tl.constexpr=a[10];bx=tl.load(U+d*N+c,c<N,0).to(tl.float32);w=(i+bx[None,:]).to(tl.bfloat16).to(tl.float32);w=ac(c[None,:]<N,w,-float('inf'));z=tl.exp(w-tl.max(w,1)[:,None]);V=z/tl.sum(z,1)[:,None];y=b<M;C=y&(b<BL);cf=tl.load(e+d*2*BL+b,C,-1);df=tl.load(e+d*2*BL+BL+b,C,-1);P=tl.load(X+d*BL+b,C,.0);F=(V-P[:,None]*(c[None,:]==cf[:,None])-(1-P[:,None])*(c[None,:]==df[:,None]))*(1./BL)
   if a[11]:jj=tl.load(J+d*a[13]+b-BL,y&(b>=BL),0);dm=tl.load(Y+d*a[14]+jj);bc=tl.load(Z+d*a[14]+jj);aW=(V-(c[None,:]==dm[:,None]))*(bc[:,None]/a[13]);F=ac((b<BL)[:,None],F,aW)
   tl.store(O+(d*M+b[:,None])*N+c[None,:],F,y[:,None]&(c[None,:]<N))
def N(a,b,f=None,full=z):d,m,k=a.shape;n=b.shape[-1];g=16 if n<=16 else 64;c=4 if m<=256 and n<=256 and(k>=1024)else 1;h=(c,d,m,n)if c>1 else(d,m,n);e=q(h,device=a.device,dtype=av if c>1 or full else a.dtype);s[ab.cdiv(m,64),ab.cdiv(n,g),d*c](a,b,e,a,m,n,k,a if f is None else f,a,a,a,a,(*a.stride(),*b.stride(),c,d,f is not None,0,0,False,g));return e.sum(0).to(a.dtype)if c>1 else e
def _n(P,k,l,e,a,c,d,j,g=None,n=None,m=None,f=None):
 h,y,b=_L(k,l,e,a,d,j,g,f)
 for i in range(J):h=_F.apply(h,K(P[i].mT),c,i)
 return h,y,b
def _o(P,G,A,x,k,z,t,i,j,o,h):
 c=ar*k.index_select(0,t.view(1))[0];e=c*(aB/ar);d=aa.where(t<20,2*(H(t)+1),40+1.*(H(t)-19));a=2 if BL==2048 else 1.;r=1-.9**d;s=(1-.99**d).sqrt()
 for p,g,m,v in zip(P[J:],G[J:],i,j):m.mul_(.9**a).add_(g,alpha=1-.9**a);v.mul_(.99**a).addcmul_(g,g,value=1-.99**a);p.mul_(1-e*X).sub_(e/r*m/(v.sqrt()/s+1e-8))
 b=[]
 for g,f in zip(G[:J],A):f.mul_(h).add_(g);b.append(g.add(f,alpha=h))
 q=K(o.flatten(0,1));l=H(N(K(L(b[1:])),q.mT));n,y=_B([b[0],l]);C=[n]+list(N(K(y),q,full=I).split(E))
 for p,u in zip(P[:J],C):p.mul_(1-c*X).sub_(u*(c*max(1.,p.shape[-2]/p.shape[-1])**.5))
 w=1-z.index_select(0,t.view(1))[0]
 for B,p in zip(x,P):B.lerp_(p,w)
def _p(S,a,n=False):
 P=S['P'];e,b,g,f=_O(S);c=S['t']
 for p in P:p.grad=None
 if a:d=S['w'](P,S['m'],S['n'],e,b,c,S['s'],g,S['l'],S['uy'],S['uw'],f)
 else:
  d=S['w'](P,S['m'],S['n'],e,b,c,S['s'],g)
 h,j,i=d;l,m,k=S['D'](h.detach(),P[-2].detach(),P[-1].detach(),j,i,S['uy']if a else None,S['uw']if a else None,f if a else None);h.backward(l);P[-2].grad,P[-1].grad=m,k
 with o():S['x']([p.detach()for p in P],[p.grad for p in P],S['j'],S['o'],S['h'],S['i'],S['t'],S['q'],S['r'],S['F'],u**(2 if BL==2048 else 5.6));S['t'].add_(1)
@ad
def h(A,I,V,J,l,N:'constexpr',K:'constexpr',b:'constexpr'):
 d=ae(0);i=ae(1);a=i*b[0]+af(0,b[0]);f=M(A+d*N+a,a<N,-8.).to(tl.float32)
 if b[4]:h=M(I+d*N+a,a<N,16383)
 else:
  h=a
 if b[2]:f=f-M(l+a,a<N,.0)
 value=tl.clamp((f+8)*16384,0,262143).to(tl.uint32);e=16383;c=value<<14|e-h&e
 if b[3]:c=ac(a==d,0,c)
 for j in range(K):g=tl.max(c,0);k=(d*b[1]+i)*K+j;S(V+k,(g>>14)*(1./16384)-8);S(J+k,e-(g&e));c=c*(c!=g)
def y(a,k,c=None,diag=z):n,m=a.shape;b=ab.cdiv(m,2048);v=q((n,b*k),device=a.device);i=q((n,b*k),device=a.device,dtype=aa.int32);d=q((n,k),device=a.device);e=q((n,k),device=a.device,dtype=R);h[n,b](a,a,v,i,a if c is None else c,m,k,(2048,b,c is not None,diag,z),enable_fp_fusion=False);h[n,1](v,i,d,e,v,b*k,k,(ab.next_power_of_2(b*k),1,z,z,I),enable_fp_fusion=False);return d,e
def _s(P,d,a,q,t):
 m=E
 with Z('cuda',dtype=G):u,o,_=O(P,d[None].expand(m,-1,-1));y,x,_=O(P,q[None].expand(m,-1,-1))
 p=H(o).mean(1,I);n=lambda h:(H(h)-p).sign()*(H(h)-p).abs().pow(aE);c=n(o);b=H(F.one_hot(a,C));v=b.T@c/b.sum(0)[None,:,None];g=c-v[:,a];j=g.mT@g/len(a);w=j.diagonal(0,-2,-1).mean(-1).clamp_min(1e-6);i=A(W,device=d.device)[None].expand(m,-1,-1);l=B.cholesky_ex(.5*j+.5*w[:,None,None]*i)[0];s=B.solve_triangular(l,i,upper=z);k=lambda r:Q(r@s.mT);f=K(k(c).permute(1,0,2).reshape(len(d),-1));e=K(k(n(x)).permute(1,0,2).reshape(len(q),-1));f,e=_R(f,e,a,t,u);return f,e,b
def _t(S):
 u,h,d=S['m'],S['n'],S['l'];m=E
 if'A'not in _S:_S['A']=x(_s)
 b,a,i=_S['A'](S['o'],u,h,d,S['I']);t,f=a@b.T,a@a.T;l=ap*y(b@a.T,aq)[0].mean(1);k=ap*y(f,aq,diag=I)[0].mean(1);g,e=y(t,aA,l);c=Y(len(d),C,device='cuda').scatter_add_(1,h[e],(au*g/m).softmax(1));s,r=y(f,v,k,I);o,j=L([g[:,:v],s],1).topk(v,1);n=L([e[:,:v],r+len(b)],1).gather(1,j);q=(au*o/m).softmax(1);p=c
 for _ in range(az):p=al*(q[...,None]*L([i,p])[n]).sum(1)+(1-al)*c
 return p
def _u(S,i=I):
 for h in 0,1:
  j=slice(h*E//2,(h+1)*E//2);P=[p[j]for p in S['o']]
  with Z('cuda',dtype=G):a=O(P,S['l'][None].expand(E//2,-1,-1))[0]
  p=H(a).softmax(-1).mean(0);e,b=p.max(-1);w=aQ(e).scatter_(0,e.topk(int(ax*len(b))).indices,1.);d=slice((1-h)*E//2,(2-h)*E//2);S['uy'][d]=b;S['uw'][d]=w
 with Z('cuda',dtype=G):a,_,f=O(S['o'],S['m'][None].expand(E,-1,-1))
 p=H(a).softmax(-1);c=-p.gather(2,S['n'][None,:,None].expand(E,-1,-1)).squeeze(-1).clamp_min(1e-8).log();g=(1-m)/len(S['m'])+m*c/c.sum(1,I);S['v'].copy_(g.cumsum(-1));n(S)
 if i:_P(S,f)
def _v(b,g):n,d=b.shape;i=ab.cdiv(n,512);c=g.argsort(stable=True);k=aV(g[c],T(11,device=b.device));a=b[c];l=L([H(k),a.square().sum(-1),H(c)]);e=aa.full((2,n,i*10),-float('inf'),device=b.device);s[ab.cdiv(n,16),10](a,a,e,e,n,n,d,l,a,a,a,a,(0,d,1,0,1,d,1,1,False,2,0,False,512),num_warps=8,num_stages=1);m,h=e;j,o=m.topk(10,dim=-1);f=h.gather(1,o).long().where(j.isfinite(),T(n,device=b.device)[:,None]);return (b[f]-b[:,None]).mT/10**.5,f
def _w(S):
 P,a=S['P'],S['m']
 with o():
  P[0].copy_(Q(a[S['K']].view(E,w,D)))
  for p,c in zip(P[1:J],S['L']):p.copy_(c)
  P[-1].zero_();P[-2].zero_()
  for d,p in zip(S['o'],P):d.copy_(p)
  for b in S['j']:b.zero_()
  for v in S['q']+S['r']:v.zero_()
  for q in S['F']:q.copy_(A(q.shape[-1],device=q.device)[None])
  S['t'].zero_();S['v'].copy_(aY(1/len(a),1,len(a),device='cuda')[None].expand(E,-1));n(S);S['uy'].zero_();S['uw'].zero_()
def _x(d,f):
 b,i,e=Y,aQ,R;a=dict(device='cuda');u=[(E,w,D)]+[(E,w,W)]*(J-1)+[(E,C,W),(E,C)];P=[b(s,**a,requires_grad=I)for s in u];t=T(at+1,**a,dtype=av);r=9;q=A(W,dtype=G,**a)[None,None].repeat(J-1,E,1,1);c={'F':q,'P':P,'t':b((),dtype=e,**a),'h':aa.clamp((t+1)/r,max=1)*.5*(1+aa.cos(aa.pi*t/at)),'i':aa.clamp((1+t)/(10+t),max=ay),'o':[i(p)for p in P],'j':[i(p)for p in P[:J]],'q':[i(p)for p in P[J:]],'r':[i(p)for p in P[J:]],'e':b(E,2049,device='cuda',dtype=e),'v':aY(1/d,1,d,**a)[None].repeat(E,1),'m':b(d,D,**a),'n':b(d,dtype=e,**a),'s':b(d,D,U,**a),'l':b(f,D,**a),'uy':b(E,f,dtype=e,**a),'uw':b(E,f,**a),'u':b(E,f//2,dtype=e,**a),'D':x(_H),'w':x(_n),'x':x(_o),'I':b(d,10,dtype=e,**a)};h=c['h']
 for k in range(20):h[k]=h[2*k:2*k+2].sum();c['i'][k]=c['i'][2*k:2*k+2].prod()
 for k in range(28):o,n=40+4.*k,40+4.*(k+1);m=[(j,min(n,j+1)-max(o,j))for j in range(math.floor(o),math.ceil(n))];h[20+k]=sum(l*h[j]for j,l in m);c['i'][20+k]=math.prod(((j+1)/(j+10))**l for j,l in m)
 g=aa.Generator(**a);g.manual_seed(0);c['K']=aa.randint(0,d,(E*w,),generator=g,**a);c['L']=tuple(aS(p.shape,generator=g,**a)*W**(-.5)for p in P[1:J]);aS(P[-2].shape,generator=g,**a);c['u'].copy_(aa.randperm(f,generator=g,**a).view(2,-1).repeat_interleave(E//2,0));return c
def _y(S,a,c,b):a,b=_k(a,c,b);S['m'].copy_(a);S['n'].copy_(c);S['l'].copy_(b);d,e=_v(S['m'],S['n']);S['s'].copy_(d*aJ);S['I'].copy_(e)
def p(S,b):
 global BL,BU;_y(S,*b);_w(S)
 for a in range(48):
  if a in(20,28,36,43):
   BL,BU=2048,512
   with o():_u(S,a==20)
  BL,BU=(4096,1024)if a>=20 else(2048,512);_p(S,a>=20,a%8==0)
 BL,BU=2048,512
 with o():
  for e,p in zip(S['o'],S['P']):e.lerp_(p,1-.1081748218867733)
  return _t(S).argmax(-1)
def classify(c,g,d):
 if'J'not in _S:
  S=_x(len(c),len(d));_S['S']=S;a=[v.clone()for v in(c,g,d)];f=i.Stream();f.wait_stream(zc())
  with i.stream(f):
   for _ in range(2):p(S,a)
  zc().wait_stream(f);b=i.CUDAGraph()
  with i.graph(b):e=p(S,a)
  _S['J']=b,a,e
 b,a,e=_S['J']
 for h,value in zip(a,(c,g,d)):h.copy_(value)
 b.replay();return e
def n(S):S['e'].copy_(aV(S['v'],j((T(2049,device='cuda')/2048)[None].expand(E,-1))))
def _B(b):
 a=K(b[0]).mT;h=K(b[1]);a=a/(a.norm('fro',(-2,-1),I)+1e-7);h=h/(h.norm('fro',(-2,-1),I)+1e-7);x=L([F.pad(a,(0,W-w,0,w-D)),h])
 for c in range(d):v=N(x,x.mT);x=N(v,x,N(v,v),c==d-1)
 return H(x[:E,:D,:w].mT),H(x[E:])
from triton.language.extra.cuda import libdevice
@ad
def _C(x):return x.to(tl.bfloat16).to(tl.float32)
@ad
def _D(i,t,L:'constexpr'):s=t.to(tl.uint32)*2246822519+(L*3266489917&4294967295);x=(i.to(tl.uint32)+s)*2654435761;x=(x^x>>16)*2246822507;x=(x^x>>13)*3266489909;return(x^x>>16)&2147483647>=214748364
@ad
def _E(G,U,T,O,N:'constexpr',L:'constexpr',d:'constexpr'):
 a=ae(0)*4+af(0,4);b=af(0,d[2]);e=a[:,None]*N+b[None,:];c=(a[:,None]<d[0])&(b[None,:]<N);u=M(U+e,c,.0);r=libdevice.rsqrt(tl.sum(u*u,1)/N+1e-6);p=M(G+a[:,None]*2*N+b[None,:],c,.0).to(tl.float32);n=M(G+a[:,None]*2*N+b[None,:]+N,c,.0).to(tl.float32)
 if d[1]:
  t=M(T);p=ac(_D(a[:,None]*2*N+b[None,:],t,L),p/.9,.0);n=ac(_D(a[:,None]*2*N+b[None,:]+N,t,L),n/.9,.0)
  if L==0:p=_C(p);n=_C(n)
 g=ac(u>0,p,ac(u<0,-n,.0));f=tl.sum(g*u,1);v=f*-.5*(r*r*r)*(1./N);h=g*r[:,None]+u*2*v[:,None];S(O+e,h,c)
class _F(aa.autograd.Function):
 @staticmethod
 def forward(c,a,b,t,f):e,m,k=a.shape;n=b.shape[-1];h=q((e,m,2*n),device=a.device,dtype=a.dtype);u=q((e,m,n),device=a.device);s[ab.cdiv(m,32),e](a,b,h,u,m,n,k,a if t is None else t,a,a,a,a,(*a.stride(),*b.stride(),1,e,False,1,f,t is not None,ab.next_power_of_2(n)));c.save_for_backward(a,b,u,a if t is None else t);c.l=f;c.d=t is not None;return h
 @staticmethod
 def backward(c,g):a,b,u,t=c.saved_tensors;e,m,n=u.shape;d=q(u.shape,device=u.device,dtype=g.dtype);_E[ab.cdiv(e*m,4),](j(g),u,t,d,n,c.l,(e*m,c.d,ab.next_power_of_2(n)));return N(d,b.mT)if c.needs_input_grad[0]else None,N(a.mT,d),None,None
def O(P,x,b=None):
 a=[];h=x*c
 for i in range(J):a.append(h);h=_F.apply(K(h),K(P[i].mT),b,i)
 return aa.baddbmm(P[-1][:,None],h,P[-2].mT),h,a
def _H(h,i,l,y,g,f=None,p=None,d=None,n=True):e,m,k=h.shape;c=i.shape[1];w=K(i);b=q((e,m,c),device=h.device,dtype=h.dtype);a=f is not None;o=y.shape[1]//2;j=0 if d is None else d.shape[1];s[ab.cdiv(m,32),e](h,w.mT,b,K(l),m,c,k,y,g,f if a else y,p if a else g,d if a else y,(*h.stride(),*w.mT.stride(),1,e,False,3,o,a,16,j,f.shape[1]if a else 0),enable_fp_fusion=False);return N(b,w)if n else None,H(N(h.mT,b).mT),H(b.sum(1))
def x(f):return aa.compile(f,fullgraph=I,dynamic=z)
@ad
def V(x,y):return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,y],dtype=tl.float32,is_pure=True,pack=1)
@ad
def _K(T,X,I,R,L,Y,Q,J,H,n,t,B:'constexpr',U:'constexpr',d:'constexpr',g:'constexpr'):
 p=ae(0)*g[3]+af(0,g[3]);e=p//(g[1]*d);a=p//d%g[1];c=p%d;m=p<g[2];b=m&(a<B//2);x,o,i=_M(T,X,I,R,e,a,c,b,B,d,g[0]);y,q,j=_M(T,X,I,R,e,a+B,c,b,B,d,g[0]);h=M(L+e*B+a,b,.0);s=tl.sin(h*1.5707963267948966);f=s*s;h=tl.maximum(f,1.-f);A=tl.fma(h,V(x,o),(1.-h)*V(y,q));z=tl.fma(h,V(x,-o),(1.-h)*V(y,-q));S(H+(e*(B+U)+a)*d+c,A*2.,b);S(H+(e*(B+U)+a+B//2)*d+c,z*2.,b);u=M(Y+i,b&(c==0),0);w=M(Y+j,b&(c==0),0)
 for r in tl.static_range(2):l=r*(B//2);S(n+e*2*B+a+l,u,b&(c==0));S(n+e*2*B+B+a+l,w,b&(c==0));S(t+e*B+a+l,h,b&(c==0))
 if U:k=m&(a>=B//2);C=M(J+e*U+a-B//2,k,0);v=M(Q+C*d+c,k,.0);S(H+(e*(B+U)+B+a-B//2)*d+c,v*2.,k)
def _L(c,j,a,d,g,i,b=None,f=None):u=BU if b is not None else 0;h=q((E,BL+u,D),device=c.device,dtype=G);y=aX(a);e=aX(d);n=E*(BL//2+u)*D;_K[ab.cdiv(n,128),](g,c,a,i,d,j,c if b is None else b,a if f is None else f,h,y,e,BL,u,D,(U,BL//2+u,n,128));return h,y,e
@ad
def _M(T,X,I,R,e,g,c,d,B:'constexpr',h:'constexpr',b:'constexpr'):f=af(0,16);i=M(I+e*2*B+g,d,0);a=M(T+(i[:,None]*h+c[:,None])*b+f[None,:],d[:,None]&(f[None,:]<b),.0);r=M(R+(e[:,None]*2*B+g[:,None])*b+f[None,:],d[:,None]&(f[None,:]<b),.0);return M(X+i*h+c,d,.0),tl.sum(a*r,1),i
@ad
def e(n,j,o,T,I,L,J,R,N:'constexpr',B:'constexpr',U:'constexpr',h:'constexpr',E:'constexpr'):
 a=ae(0)*128+af(0,128);t=M(T).to(tl.uint32)*(524288 if B==4096 else 262144);z=tl.randn(109,t+65536+a);S(R+a,z,a<E*2*B*10)
 if ae(0)<tl.cdiv(E*B,128):
  b=a<E*B;e=a//B;c=a%B;u=tl.rand(109,t+a);v=tl.rand(109,t+E*B+a);m=tl.rand(109,t+2*E*B+a);l=tl.minimum((u*2048).to(tl.int32),2047);d=M(j+e*2049+l,b,0).to(tl.int32);f=M(j+e*2049+l+1,b,0).to(tl.int32)
  while tl.max(f-d,0)>0:g=(d+f)//2;p=M(n+e*N+g,b&(g<N),float('inf'));i=p<u;k=d<f;d=ac(k&i,g+1,d);f=ac(k&~i,g,f)
  S(I+e*2*B+c,tl.minimum(d,N-1),b);S(I+e*2*B+B+c,tl.minimum((v*N).to(tl.int32),N-1),b);S(L+a,m,b);q=tl.minimum((tl.rand(109,t+(49152 if B==4096 else 32768)+e*U+c)*h).to(tl.int32),h-1);value=M(o+e*h+q,b&(c<U),0);S(J+e*U+c,value,b&(c<U))
def _O(S):c=q((E,2*BL),device='cuda',dtype=R);b=q((E,BL),device='cuda');a=q((E,2*BL,U),device='cuda');d=q((E,BU),device='cuda',dtype=R);e[ab.cdiv(a.numel(),128),](S['v'],S['e'],S['u'],S['t'],c,b,d,a,len(S['m']),BL,BU,S['u'].shape[1],E);return c,b,a,d
def _P(S,e):
 for h,q in zip(e[1:],S['F']):b=H(h[:,:4096]);d=b.shape[-1];a=b.mT@b/b.shape[1];g=a.diagonal(0,-2,-1).mean(-1,I).clamp_min(1e-6);c=A(d,device=h.device)[None].expand(E,-1,-1);a=.5*a/g[...,None]+.5*c;f=_Q(a,c);q.copy_(f)
def _Q(a,c):
 b=a.norm('fro',(-2,-1),I);y=a/b;z=c
 for _ in range(10):t=.5*(3*c-z@y);y=y@t;z=t@z
 return z/b.sqrt()
def _R(p,o,e,m,l):b=Q(H(p).view(-1,E,W).transpose(0,1));k=H(o).view(-1,E,W).transpose(0,1);c=(-H(l).log_softmax(-1).mean(0).gather(1,e[:,None]).flatten()).topk(256).indices;h=b[:,c];d=aa.bmm(h,b.mT).sum(0)*(80/E);d=d.scatter(1,L([c[:,None],m[c]],1),-float('inf'));n=e[c,None]==e[None];i=d.softmax(-1)-d.masked_fill(~n,-float('inf')).softmax(-1);f=L([h,b],1);j=L([i[None]@b,i.T[None]@h],1);g=f.mT@(j-(j*f).sum(-1,I)*f);a=A(W,device=e.device)[None]-g/g.norm('fro',(-2,-1),I).clamp_min(1e-12);return[K(Q(z@a).transpose(0,1).reshape(z.shape[1],-1))for z in(b,k)]
def Q(x):return F.normalize(x,dim=-1)
def H(x):return x.float()
def K(x):return x.bfloat16()
def j(x):return x.contiguous()
ae=tl.program_id;af=tl.arange;ac=tl.where;r=256
