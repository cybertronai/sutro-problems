ar=.5;ap=True;aj=.5;ai=-.5;from torch import no_grad as X;from triton.language import load as _L,store as _S,program_id as _P,arange as _A,where as _W;import math,torch,torch as k,torch.nn.functional as F,triton,triton.language as tl;I=False;G=True;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;br,cq,dg,c9,cr,dx=(.02,.1,256,512,.02,.6);cz,ca=((3000,10000),(16,16));P,bs,cs=(2000,15,3.);f6=k.float32;M,al=(1,(60,1024,512,256,10));ax,ad,LR,bt,bb=(650,1024,.011,.8,2/3);bu,di,dj,fa,dk=(.25,.6,2000.,1e-10,11);fb,dm,ct,cu=(.5,.1,.6,.9);ei=k.bfloat16;fh={'compile_threads':8,'combo_kernels':G,'triton.autotune_pointwise':I,'pattern_matcher':I,'reorder_for_peak_memory':I};aw={};D=triton.cdiv;U=k.where;T=k.stack;B=k.empty;Z=k.zeros_like;N=k.empty_like;C=k.cat;E=k.eye;t=k.Tensor;Q=t.square;K=t.reshape;J=t.float;V=t.contiguous;W=t.clamp_min;Y=t.sum;H=t.sqrt;R=t.half;O=t.copy_;a6=k.backends.cuda.matmul;a2=k.cuda;a7=k.linalg;ab=F.one_hot;a9=F.softmax;a8=k.rsqrt;aa=k.Tensor.norm;a3=k.Tensor.mean
def eq(a):
 return a7.cholesky_ex(a)[0]
def rn(x):
 return x*(x.shape[1]**.5/W(aa(x,dim=1,keepdim=G),1e-6))**dx
def fj(a,b,e=.0):
 c=Y(Q(a),1,keepdim=G);d=Y(Q(b),1,keepdim=G);A=C([a,c,k.ones_like(c)],1);B=V(C([2*br*b,k.full_like(d,-br),e-br*d],1).T);return(A,B)
def e6(a,b):
 A,B=fj(a,b);return(A@B).exp_()
def d5(z):
 A,B=fj(z,z,cs);a,b=(R(A),R(B));c,d=(R(A-J(a)),R(B-J(b)));return(C([a,a,c],1),C([b,d,b],0))
def d6(c,d,g,e,f):
 a=[]
 for i in e:
  r=min(g,c.shape[0]-i);b=k.mm(c[i:i+r],d[:,i:]);b.exp_();b[:,:r].diagonal().fill_(f);a.append(b)
 return a
class fz:
 def __init__(S,a,b,n):
  S.pans,S.pp,S.n=(a,b,n)
 def mv(S,g):
  f,n=(S.pp,S.n);y=B(n,g.shape[1],device=g.device);c=S.pans[0];b=min(c.shape[0],n);k.mm(c[:b,:n],g,out_dtype=f6,out=y[:b])
  if b<n:
   k.mm(c[:,b:n].T,g[:b],out_dtype=f6,out=y[b:n])
  for j in range(1,len(S.pans)):
   a=j*f
   if a>=n:
    break
   d=S.pans[j];e=min(a+d.shape[0],n);k.addmm(y[a:e],d[:,:n-a],g[a:n],out_dtype=f6,out=y[a:e])
   if e<n:
    k.addmm(y[e:n],d[:,e-a:n-a].T,g[a:e],out_dtype=f6,out=y[e:n])
  return y
def c3(c,d,w,r,e,v,f):
 a=e/Y(d*c,0);w=w+d*a;r=r-c*a;p=r-v@(f@r);b=Y(r*p,0);d=p+d*(b/e);return(w,r,d,b,R(d))
def e7(c,d):
 b=c.clone();b.diagonal().add_(d);a=E(len(c),device=c.device);return a7.solve_triangular(eq(b),a,upper=I).T
def fx(i,t,h,g,u):
 a=u.T@u;a.diagonal().add_(h);b=E(len(a),device=u.device);v=u@a7.solve_triangular(eq(a),b,upper=I).T;f=V(v.T);w=Z(t);r=t.clone();d=k.addmm(r,v,f@r,alpha=-1);e=a7.vecdot(r,d,dim=0);c=R(d)
 for _ in range(g):
  w,r,d,e,c=c3(i.mv(c),d,w,r,e,v,f)
 return w
def cw(a):
 c=aa(a);n=len(a);x=k.zeros(2*n,2*n,device=a.device);x[:n,n:]=a/c;x[n:,:n]=E(n,device=a.device);b=1.5*E(2*n,device=a.device)
 for _ in range(bs):
  x=x@k.addmm(b,x,x,alpha=-.5)
 return x[:n,n:]*H(c)
def e4(e,z,w,f):
 s=f.shape[0];d=C([w,K(w[:,:,None]*z[:,None,:],len(z),-1)],1);r=J(f)@d;c=w.shape[1];g=K(r[:,c:],s,c,-1)-r[:,:c,None]*z[:s,None,:];a=e.shape[0];g=K(g@e.T,-1,a);m=g.T@g;b=cw(m);m=b@cw(b);m=(m+m.T)/2;m=m*(a/W(m.trace(),1e-12));m=(1-cr)*m+cr*E(a,device=z.device);return eq(m)
@X()
def en(x,y):
 c=math.exp(cs);n,e,l=(cq*c,.0001*c,c*(1+cq));x=rn(J(x));k=J(ab(y.long(),10));d=E(x.shape[1],device=x.device)
 for m,j in zip(cz,ca):
  z=x[:m]@d;i=e6(z,z[:dg])*c;f=P if m%P==0 else m;a,b=d5(z);h=d6(a,b,f,range(0,m,f),l);w=fx(fz(h,f,m),k[:m],n,j,i@e7(i[:dg],e));g=h[0][:c9].clone();g.diagonal().fill_(c);bc=d;d=e4(d,z,w,g)
 return(d,z,w,bc)
def es(f,g,h):
 e=('teacher',tuple(f.shape),str(f.device))
 if e not in aw:
  b=[f.clone(),g.clone()];d=a2.Stream();d.wait_stream(a2.current_stream())
  with a2.stream(d):
   for _ in range(2):
    en(*b)
  a2.current_stream().wait_stream(d);c=a2.CUDAGraph()
  with a2.graph(c):
   a=en(*b)
  aw[e]=(c,b,a)
 c,b,a=aw[e];O(b[0],f);O(b[1],g);c.replay();return tuple((v.clone()for v in a))
cf,ak,am=(.12,True,8);a5,bo=(True,32);ba=(0,1,2,4,5,6,7)
def e8(z):
 b=E(z.shape[-2],device=z.device,dtype=z.dtype);a=z@z.mT;p=b
 for i in range(3):
  d=3.4445*b-4.775*a+2.0315*(a@a);p=d if i==0 else d@p;a=d@a@d.mT
 return p
@X()
def a0(p,j,i,r,m):
 w=J(p[0][0]);e,x=(J(i[0]),J(r[0]));af=e.T@e/ad-bu**2*E(al[0],device=e.device);q=T((af@w,x.T@e/ad@w,e.T@x/ad@w));q=q-w*Y(w*q,1,keepdim=G)/W(Y(Q(w),0,keepdim=G),1e-12);a4=Y(m[:,None,None]*q,0)[None];n=q.flatten(1);y=n@n.T;s=W(y.trace(),1e-12);h=y/s+.001*E(3,device=e.device);C=n@J(j[0]).flatten()/s;a,b,c,d,f,g=(h[0,0],h[0,1],h[0,2],h[1,1],h[1,2],h[2,2]);o,k,l=(d*g-f*f,c*f-b*g,b*f-c*d);I,u,M=(a*g-c*c,b*c-a*f,a*d-b*b);K=T((o,k,l,k,I,u,l,u,M)).view(3,3);L=a*o+b*k+c*l;F=(K@C/W(L,1e-12)).clamp(-2,2);return(a4,F)
@X()
def d9(f,e,l,k0,t,c,d,h,n):
 t.add_(1);D=1-bt**t;Et=H(1-.999**t);zs={}
 for i,(p,g,m,v)in enumerate(zip(h,e,l,k0)):
  g=J(g)-n if i==0 else J(g);m.lerp_(g,1-bt)
  if i in ba:
   z=g.lerp(m,bt)[0];zs[i]=(z.T if z.shape[0]>z.shape[1]else z).bfloat16()
  else:
   v.mul_(.999).addcmul_(g,g,value=.001);p.add_(-(c/D)*m/(H(v)/Et+1e-8))
 for b2 in((0,4),(1,5),(2,6),(7,)):
  if ak:
   b5=[W(aa(zs[i]),1e-7)for i in b2];b1=T([zs[i]/s for i,s in zip(b2,b5)]);b0=e8(b1)
   for j,(i,s)in enumerate(zip(b2,b5)):
    O(d[i],b0[j]/s)
  for j,i in enumerate(b2):
   p=h[i];z=d[i]@zs[i];z=J((z.T if p.shape[-2]>p.shape[-1]else z)[None]);b8=math.sqrt(max(1,p.shape[-1]/p.shape[-2]));p.add_(-(c/LR)*cf*b8*(ar if ap and i==5 else 1.)*z)
 for i in range(8):
  O(f[i],h[i])
def bn(u):
 b,a=k.var_mean(u,dim=-2,unbiased=I,keepdim=G);return(u-a)*a8(b+fa)
def cx(t):
 return F.leaky_relu(t,.1)
def av(u,a,p):
 return dv(bn(u),a,p)
def dv(v,d,p):
 a=v*d;b=cx(v*p[0]+d*p[2]+a*p[4]+p[6]);c=cx(v*p[1]+d*p[3]+a*p[5]+p[7]);return b*p[8]+c*p[9]+p[10]
@triton.jit
def bf(u,d,n,e):
 c=tl.sum(u,0)/n;a=_W(d,u-c[None,:],.0);b=tl.rsqrt(tl.sum(a*a,0)/n+e);return(a*b[None,:],b)
@triton.jit
def co(U,L,P,O,N:tl.constexpr,S:tl.constexpr,E:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
 A=tl.float32;c=_P(0)*C+_A(0,C);r=_A(0,R);g=(r<N)[:,None];d=g&(c<S)[None,:];f=r[:,None]*S+c[None,:];u=_L(U+f,d,0).to(A);e=_L(L+f,d,0).to(A);v,_=bf(u,g,N,E);p=P+c;i=dy(p,c<S,S,0);j=dy(p,c<S,S,1);k=dy(p,c<S,S,2);l=dy(p,c<S,S,3);m=dy(p,c<S,S,4);n=dy(p,c<S,S,5);o=dy(p,c<S,S,6);q=dy(p,c<S,S,7);s=dy(p,c<S,S,8);t=dy(p,c<S,S,9);h=dy(p,c<S,S,10);x=v*e;a=v*i+e*k+x*m+o;b=v*j+e*l+x*n+q;a=_W(a>0,a,.1*a);b=_W(b>0,b,.1*b);_S(O+f,a*s+b*t+h,d)
@triton.jit
def cp(U,L,P,G,O,K,M,N:tl.constexpr,S:tl.constexpr,E:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
 J=tl.float32;c=_P(0)*C+_A(0,C);r=_A(0,R);l=(r<N)[:,None];f=l&(c<S)[None,:];i=r[:,None]*S+c[None,:];u=_L(U+i,f,0).to(J);h=_L(L+i,f,0).to(J);g=_L(G+i,f,0).to(J);v,B=bf(u,l,N,E);p=P+c;n=dy(p,c<S,S,0);o=dy(p,c<S,S,1);s=dy(p,c<S,S,2);t=dy(p,c<S,S,3);w=dy(p,c<S,S,4);y=dy(p,c<S,S,5);D=dy(p,c<S,S,6);F=dy(p,c<S,S,7);H=dy(p,c<S,S,8);I=dy(p,c<S,S,9);x=v*h;a=v*n+h*s+x*w+D;b=v*o+h*t+x*y+F;d=g*H*_W(a>0,1.,.1);e=g*I*_W(b>0,1.,.1);q=M+c;_S(q,tl.sum(d*v,0),c<S);_S(q+S,tl.sum(e*v,0),c<S);_S(q+2*S,tl.sum(d*h,0),c<S);_S(q+3*S,tl.sum(e*h,0),c<S);_S(q+4*S,tl.sum(d*x,0),c<S);_S(q+5*S,tl.sum(e*x,0),c<S);_S(q+6*S,tl.sum(d,0),c<S);_S(q+7*S,tl.sum(e,0),c<S);_S(q+8*S,tl.sum(g*_W(a>0,a,.1*a),0),c<S);_S(q+9*S,tl.sum(g*_W(b>0,b,.1*b),0),c<S);_S(q+10*S,tl.sum(g,0),c<S)
 for j in range(11,17):
  _S(q+j*S,.0,c<S)
 m=d*w+e*y;z=d*s+e*t+m*v;k=_W(f,d*n+e*o+m*h,.0);A=B[None,:]*(k-tl.sum(k,0)[None,:]/N-v*tl.sum(k*v,0)[None,:]/N);_S(O+i,A,f);_S(K+i,z,f)
@triton.jit
def dy(p,m,s,j:tl.constexpr):
 return _L(p+j*s,m,0).to(tl.float32)[None,:]
class FC(k.autograd.Function):
 @staticmethod
 def forward(d,u,b,p):
  n,s=u.shape;a=N(u);c=16 if s>=1024 else 8 if s>=512 else 4 if s>=256 else 2;co[D(s,c),](u,b,p,a,n,s,1e-10,triton.next_power_of_2(n),c,num_warps=8,enable_fp_fusion=I);d.save_for_backward(u,b,p);return a
 @staticmethod
 def backward(f,g):
  u,e,p=f.saved_tensors;n,s=u.shape;d,a,b=(N(u),N(e),N(p));c=8 if s==512 else 4;cp[D(s,c),](u,e,p,V(g),d,a,b,n,s,1e-10,triton.next_power_of_2(n),c,num_warps=4 if s==1024 else 8,enable_fp_fusion=I);return(d,a,b)
def b6(u,a,p):
 if u.dtype==ei:
  return FC.apply(K(u,-1,u.shape[-1]),K(a,-1,a.shape[-1]),K(p,17,-1)).reshape_as(a)
 return av(u,a,p)
def mm(h,w,b=None):
 s=h.shape
 with k.autocast('cuda',dtype=b or k.float16,enabled=b is not None):
  a=k.bmm(K(h,s[0],-1,s[-1]),w)
 return K(a if b is not None else J(a),*s[:-1],w.shape[-1])
@triton.jit
def b4(I,L,R,C,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
 m=_P(1);j=_P(0)*B+_A(0,B);i=_L(I+m*K+j,j<K,0);d=_L(L+m*K+j,j<K,0);b=d.to(tl.uint32,bitcast=True).to(tl.uint64);a=_L(C).to(tl.uint64)+1;c=a*2048+j.to(tl.uint64)+1<<32|b;tl.atomic_max(R+m*N+i,c.to(tl.int64),j<K)
def b3(e,b,a,d,c):
 b4[D(ad,128),M](a,d,b,c,e.shape[1],ad,128)
az,au,ao,aq=(32,16,16,8)
@triton.jit
def c6(U,B,o,p,H,O,Z,T,S:tl.constexpr,n:tl.constexpr,C:tl.constexpr):
 A=tl.float32;m=256 if S==512 else S;a=_P(1);r=_A(0,1024);c=_P(0)*C+_A(0,C);b=(c<S)[None,:];h=(a*1024+r[:,None])*S+c[None,:];d=_L(U+h,b,0).to(A);e=tl.sum(d,0)/1024;f=d-e[None,:];g=tl.rsqrt(tl.sum(f*f,0)/1024+1e-10).to(U.dtype.element_ty).to(A);_S(T+a*S+c,e,c<S);_S(T+(2+a)*S+c,g,c<S);l=_L(o+a*1024+r).to(A);k=_L(p+a*m+c%m,c<S,0).to(A);i=_L(H+r[:,None]*n+(c%m)[None,:],b,0).to(A);z=(d-e.to(U.dtype.element_ty).to(A)[None,:])*g[None,:]+.25*l[:,None]*i*k[None,:];j=_L(B+c,c<S,0).to(A);_S(O+h,tl.maximum(z+j[None,:],0),b)
 if a==1:
  _S(Z+r[:,None]*S+c[None,:],z,b)
@triton.jit
def c7(U,O,T,o,p,n,m,S:tl.constexpr,C:tl.constexpr):
 A=tl.float32;b=_P(1);r=_A(0,1024);c=_P(0)*C+_A(0,C);a=(c<S)[None,:];e=(b*1024+r[:,None])*S+c[None,:];h=_L(U+e,a,0).to(A);i=_L(T+b*S+c,c<S,0);f=_L(T+(2+b)*S+c,c<S,0);j=_L(O+e,a,0)>0;g=_L(o+e,a,0).to(A)*j;d=g
 if b==1:
  d=d+_L(p+r[:,None]*S+c[None,:],a,0).to(A)
 k=tl.sum(d*(h-i.to(U.dtype.element_ty).to(A)[None,:]),0)/1024;l=f[None,:]*(d-tl.sum(d,0)[None,:]/1024-f[None,:]*f[None,:]*(h-i[None,:])*k[None,:]);_S(n+e,l,a);_S(m+b*S+c,tl.sum(g,0),c<S)
class b9(k.autograd.Function):
 @staticmethod
 def forward(g,u,b,f,e,d):
  _,n,s=u.shape;a=N(u);z=N(u[0]);c=B(4,s,device=u.device);c6[D(s,az if s==1024 else 16),2](u,b,f,e,d,a,z,c,s,d.stride(0),az if s==1024 else 16,num_warps=au if s==1024 else 8,enable_fp_fusion=I);g.save_for_backward(u,a,c,b);return(a,z)
 @staticmethod
 def backward(f,g,h):
  u,a,e,b=f.saved_tensors;_,n,s=u.shape;d=N(u);c=B(2,s,device=u.device);c7[D(s,ao if s>=512 else 8),2](u,a,e,V(g),V(h),d,c,s,ao if s>=512 else 8,num_warps=aq if s>=512 else 8,enable_fp_fusion=I);return(d,Y(c,0).to(b.dtype).reshape_as(b),None,None,None)
@triton.jit
def eb(s,U,v,t,X,Y,z,u,w,y,x,N:tl.constexpr,D:tl.constexpr,K:tl.constexpr,R:tl.constexpr):
 e=_P(0)*R+_A(0,R);d=e<K;b=_L(U+e,d,0)*_L(s+N-1);h=tl.full((R,),0,tl.int32);j=tl.full((R,),N,tl.int32)
 for _ in range(14):
  g=(h+j)//2;c=_L(s+g,d&(g<N),float('inf'));k=c<b;h=_W(k,g+1,h);j=_W(k,j,g)
 i=tl.minimum(h,N-1);_S(u+e,i,d);n=_L(t).to(tl.int32);o=_L(v+n*K+e,d,0).to(tl.int32);a=_A(0,64);f=d[:,None]&(a<D)[None,:];l=e[:,None]*D+a[None,:];q=_L(X+i[:,None]*D+a[None,:],f,0);r=_L(z+o[:,None]*D+a[None,:],f,0);_S(w+l,q,f);_S(x+l,r,f);m=d[:,None]&(a<10)[None,:];p=_L(Y+i[:,None]*10+a[None,:],m,0);_S(y+e[:,None]*10+a[None,:],p,m)
def gather(e,g,f):
 a=B(1,1024,60,device=e.e);d=B(1,1024,10,device=e.e);c=N(a);b=B(1,1024,device=e.e,dtype=k.int64);eb[D(1024,16),](g,f,e.A,e.i,e.x,e.y,e.E,b,a,d,c,e.n,60,1024,16,num_warps=4,enable_fp_fusion=I);return(b,a,d,c)
@triton.jit
def el(c,b,e,d,N:tl.constexpr,B:tl.constexpr):
 r=_A(0,B);v=_L(c+2*r,r<N,.0);a=tl.sum(v,0);m=_L(e+_L(b).to(tl.int32));p=_W(r<N,v/a*m+(1-m)/N,.0);_S(d+r,tl.associative_scan(p,0,ff),r<N)
@triton.jit
def ff(a,b):
 return a+b
class ey:
 def __init__(S,n,q,c,e=al):
  global ak,a5,ap;ap=G;a5=G;ak=G;S.w=I;S.I=[];h=k.arange(ad,device=c)[:,None]&k.arange(max(e[1:]),device=c)[None]
  for m in(16,8,4,2,1):
   h=h^h>>m
  S.H=(1-2*(h&1)).to(ei);S.n,S.P,S.e=(n,q,c);d=lambda*s:k.zeros(*s,device=c);S.c=[d(M,a,b)for a,b in zip(e[:-1],e[1:])];S.j=[d(M,b,a)for a,b in zip(e[:-1],e[1:])];S.a=[p.to(ei)for p in S.c];S.d=[p.to(k.float16 if i==0 else ei)for i,p in enumerate(S.j)];S.b=[d(M,1,b)for b in e[1:]];S.g=d(M,1,e[-1]);S.h=[d(17,M,1,s)for s in e];S.f=S.a+S.d+S.b+[S.g]+S.h;S.l=S.c+S.j+S.b+[S.g]+S.h
  for p in S.f:
   p.requires_grad_(G)
  S.v=d(n,60,4);S.C=d(n,2);S.x=d(n,e[0]);S.y=d(n,10);S.E=d(n+q,e[0]);S.p=d(1);S.fm=d(2);S.B=k.zeros(M,n,device=c,dtype=k.long);S.z=S.B.view(f6)[:,::2];S.i=k.zeros((),dtype=k.long,device=c);S.A=k.zeros(ax,M,ad,dtype=k.long,device=c);g=k.arange(ax,device=c,dtype=f6)/ax;S.k=LR*((1-g)/(1-bb)).clamp(max=1.);rt=(LR*(480-k.arange(ax,device=c,dtype=f6))/(480-256)).clamp(0,LR);S.k[256:]=rt[256:]*(S.k[256:480].sum()/rt[256:480].sum());o=((cu-g)/(cu-ct)).clamp(0,1);S.q=fb*U(g<dm,.0,U(g<ct,1.,o));S.G=k.tensor(LR,device=c);S.t=[Z(p)for p in S.l];S.u=[Z(p)for p in S.l];S.D=k.zeros((),device=c);S.o=d(3);S.r=[E(min(p.shape[-2:]),device=c,dtype=ei)if i in ba else None for i,p in enumerate(S.f)];S.m=Z(S.l[0]);S.K=k.compile(a0,fullgraph=G,dynamic=I,options=fh);S.R=k.compile(d9,fullgraph=G,dynamic=I,options=fh);S.Q=b3;S.O=k.compile(S._l,dynamic=I,options=fh);S.yd=d(n+q,10);S._z();a5=True
  for l in(G,I):
   ak=l;f=S.cap()
   if l:
    S.F=f
   else:
    S.M=f
  S.w=G;S.w=G;a5=I;S.U=S.cap(8);ap=I;ak=G;S.N=S.cap();ak=I;S.J=S.cap();S.V=S.cap(8);a5,ak=(G,G)
 def cap(S,n=1):
  global ak;ay=ak;a=a2.Stream();a.wait_stream(a2.current_stream())
  with a2.stream(a):
   for _ in range(3):
    for q in range(n):
     ak=ay if n==1 else q==0
     for p in S.f:
      p.grad=None
     S._a()
  a2.current_stream().wait_stream(a);b=a2.CUDAGraph()
  with a2.graph(b):
   for q in range(n):
    ak=ay if n==1 else q==0
    for p in S.f:
     p.grad=None
    S._a();S.I.append((S.L,[p.grad for p in S.f]))
  ak=ay;return b
 def _l(S,A,B,y,x):
  q,s=([b.to(ei)for b in S.b],S.g.to(ei));g=[p.to(ei)for p in S.h];h=T((A,y),1);c=bu*k.randn_like(h);v=S.v[x];n=S.C[x];e=c[:,0];u=k.randn(*e.shape[:-1],4,device=e.device);e=n[:,:,0,None]*e+n[:,:,1,None]*Y(v*u[:,:,None,:],-1);c=T((e,c[:,1]),1);h=h+c;d=[h[:,1]];h=h.to(ei);m=len(S.a)-1
  for i,w in enumerate(S.a):
   a=mm(h,w,ei);l=(k.rand((*a.shape[:-1],1),device=a.device)>.5).to(ei)*2-1;j=(k.rand((*a.shape[:2],1,256 if i==1 and w.shape[-1]==512 else a.shape[-1]),device=a.device)>.5).to(ei)*2-1
   if i<m:
    h,z=b9.apply(K(a,2,ad,-1),S.b[i],l,j,S.H);h=h.reshape_as(a);d.append(z[None])
   else:
    z=bn(a)+bu*l*S.H[None,None,:,:a.shape[-1]]*j;d.append(z[:,1]);h=(z+q[i][:,None])*s[:,None]
  o=J(h);f=-Y(B*F.log_softmax(o[:,0],-1),-1);u=S.A.index_select(0,S.i.view(1))[0];b7=S.yd[u];u=S.y[u.clamp(max=S.n-1)]*(u<S.n)[:,:,None];v=a9(o[:,1],-1);h=S.fm[0]*(o[:,1]-a3(o[:,1],-1,keepdim=G))/10+S.fm[1]*v;v=v+Y(u,-1,keepdim=G)*(h-h.detach());v=v+S.p*(u-v*Y(u,-1,keepdim=G)).detach();v=v+S.p*((256-S.i)/64).clamp(0,1)*(1-Y(u,-1,keepdim=G))*(b7-v).detach();r=b6(v.to(ei),d[-1],g[-1])
  for i in range(m,0,-1):
   r=b6(mm(r,S.d[i],ei),d[i],g[i])
  r=b6(J(mm(r,S.d[0],k.float16)),d[0],S.h[0]);t=Y(a3(Q(J(r)-y),(1,2)));return(Y(a3(f,-1))+dj*t,f.detach(),c[:,1])
 def _a(S):
  g=S.i.view(1);c=N(S.z);el[1,](S.z,S.i,S.q,c,S.n,16384,num_warps=8,enable_fp_fusion=I);f,l,m,b=gather(S,c,k.rand(M,ad,device=S.e));O(S.G,S.k.index_select(0,g).squeeze(0));j,i,d=S.O(l,m,b,f);j.backward();e=[p.grad for p in S.f]
  if a5:
   a,h=S.K(S.f,e,d,b,S.o)
  else:
   a=S.m
  S.L=(d,b,a);S.R(S.f,e,S.t,S.u,S.D,S.G,S.r,S.l,a)
  if a5:
   O(S.o,h)
  with X():
   S.Q(S.z,S.B,f,i,S.i);S.i.add_(1)
 @X()
 def _z(S,an=None):
  d,e=(S.a,S.d);S.a,S.d=(S.c,S.j)
  if an is None:
   g=k.Generator(device=S.e);g.manual_seed(dk)
   for w in S.a+S.d:
    O(w,k.randn(w.shape,generator=g,device=S.e)*w.shape[1]**(-.5))
   for w,v in zip(S.a,S.d):
    O(v,w.mT*(w.shape[1]/w.shape[2])**.5)
   for p in S.h:
    k.randn(12,*p.shape[1:],generator=g,device=S.e);p.zero_();p[0].fill_(.1);p[1].fill_(-.1);p[2].fill_(1.);p[3].fill_(-1.);p[8].fill_(.9/1.1);p[9].fill_(-.9/1.1)
   for p in S.b:
    p.zero_()
   S.g.fill_(1.);b=S.n+S.P;c=b//ad;j=math.ceil(ax/c);u=k.rand(M,j,b,generator=g,device=S.e).argsort(-1)[:,:,:c*ad];O(S.A,K(u,M,-1,ad)[:,:ax].transpose(0,1));a2.manual_seed(dk+1)
   if S.w:
    h=S.x[None]
    for w in S.a[:-1]:
     h=F.relu(bn(mm(h,w)))
    i=S.y.T@h[0]/W(Y(S.y,0)[:,None],1);f=(i-a3(h[0],0)).T;a=F.normalize(f,dim=0)[None];O(S.a[-1],a);O(S.d[-1],a.mT*math.sqrt(a.shape[1]/a.shape[2]))
  else:
   k._foreach_copy_(S.l,an)
  for p in S.t+S.u:
   p.zero_()
  S.D.zero_();S.o.zero_()
  for p in S.r:
   if p is not None:
    p.zero_()
  S.i.zero_();S.B.fill_(1065353216);S.a,S.d=(d,e);k._foreach_copy_(S.a+S.d,S.c+S.j)
 def act(S,i,t,b,a):
  return(t+b)*a if i==len(S.a)-1 else F.relu(t+b)
 @X()
 def _p(S,q):
  g=k.Generator(device=S.e);g.manual_seed(1);d=S.n//1000;h=S.x[k.randperm(S.n,generator=g,device=S.e)[:d*1000]];h=K(h,1,d,1000,-1).expand(M,-1,-1,-1);q=q.expand(M,-1,-1);T=[h[:,0]];L=[q]
  for i,w in enumerate(S.c):
   b=mm(h,w);c,a=k.var_mean(b,dim=2,unbiased=I,keepdim=G);z=(b-a)*a8(c+fa);T.append(z[:,0]);h=S.act(i,z,S.b[i][:,None],S.g[:,None]);e,f=(a3(a,1),a3(c,1)*1000/(1000-1));v=(mm(q,w)-e)*a8(f+fa);L.append(v);q=S.act(i,v,S.b[i],S.g)
  return(q,T,L)
@X()
def eh(x,y,q,p):
 a6.allow_tf32=I;i=p.topk(3,1);d=(i.values[:,0]-i.values[:,1]).topk(1024,largest=I).indices;l=i.indices[d];z=R(x);s=R(q[d]);D=W(Y(Q(J(s)),1)[:,None]+Y(Q(J(z)),1)[None]-2*k.mm(s,z.T,out_dtype=f6),0);C,o=D.topk(2048,1,largest=I);t=y[o][:,None,:]==l[:,:,None];f,B=C[:,None,:].masked_fill(~t,float('inf')).topk(32,2,largest=I);e=o.gather(1,B.flatten(1)).view(1024,3,32);g=k.isinf(f[:,:,0]);e=U(k.isinf(f),e[:,:,:1],e);r=x[e];j=a3(r,2);a=K(r-j[:,:,None],-1,32,x.shape[1]);v=K(q[d,None]-j,-1,x.shape[1],1);ag=K(W(U(g,k.ones_like(f[:,:,0]),f[:,:,0]/x.shape[1]),1e-5),-1,1,1);c=a@a.mT+8*ag*E(32,device=x.device);w=k.cholesky_solve(a@v,eq(c));A=v-a.mT@w;b=K(Y(Q(A),(1,2)),1024,3).masked_fill(g,float('inf'));n=(b[:,1:]-b[:,:1]).abs();u=W(k.nanmedian(n.masked_fill(~n.isfinite(),float('nan'))).nan_to_num(nan=1.),1e-6);h=k.zeros(1024,10,device=x.device).scatter_(1,l,(-b/u).softmax(1));h=U(g.all(1,keepdim=G),p[d],h);m=p.clone();m[d]=p[d].lerp(h,.35);return m
@X()
def g2(x,y):
 a6.allow_tf32=I;z=R(x);a=Y(Q(J(z)),1);b=(a[:,None]+a[None,:]-2*k.mm(z,z.T,out_dtype=f6)).clamp_min_(0);b.masked_fill_(y[:,None]!=y[None,:],float('inf'));b.fill_diagonal_(float('inf'));c,d=b.topk(4,1,largest=I);a6.allow_tf32=G;return(x[d]-x[:,None]).mT*c.isfinite()[:,None,:]
@X()
def g3(c):
 b=Y(Q(c),(1,2));a=.5*(b>1e-12);return T((H(1-a),.25*H(a*60/W(b,1e-12))),1)
@X()
def a1(S,y,q):
 e,ae,L=S._p(q);p=a3(a9(e,-1),0);b=(p.topk(2,1).values[:,0]-p.topk(2,1).values[:,1]).topk(1024,largest=I).indices;c=p[b].topk(3,1).indices;g=[z[:,b]for z in L];r=a9((ae[-1]+S.b[-1])*S.g,-1);d=[]
 for i in range(4,-1,-1):
  if i<4:
   r=mm(r,S.j[i])
  v,m=k.var_mean(r,dim=-2,unbiased=I,keepdim=G);d.append((m,a8(v+fa)));r=dv((r-m)*a8(v+fa),ae[i],S.h[i])
 E=[];ac=[]
 for j in range(3):
  r=J(ab(c[:,j],10))[None]
  for t,i in enumerate(range(4,-1,-1)):
   if i<4:
    r=mm(r,S.j[i])
   m,v=d[t];r=dv((r-m)*v,g[i],S.h[i])
  E.append(a3(Q(r[0]-q[b]),-1));ac.append(Y(r[0]*q[b],-1))
 E=T(E,1);f=(E[:,1]-E[:,0]).abs().median().clamp_min(1e-9);ac=T(ac,1);rs=(ac[:,1]-ac[:,0]).abs().median().clamp_min(1e-9);a=a9(-.5*E/f+.5*ac/rs,-1);p[b]*=1-.5;p[b[:,None],c]+=.5*a;return eh(S.x,y,q,p)
def ladder_kernel(r,o,s):
 a6.allow_tf32=I;h,tz,tw,to=es(r,o,s);a6.allow_tf32=G
 with X():
  x,q=(rn(J(r))@h,rn(J(s))@h);c=(J(r)*di).std()/x.std();x,q=(x*c,q*c);d=x.double();m=a3(d,0);d=d-m;_,v=a7.eigh(d.T@d);z=d@v;f=z[z.abs().argmax(0),k.arange(60,device=x.device)].sign();x,q=(J(z*f),J((q.double()-m)@(v*f)))
 g=('ladder',x.shape[0],q.shape[0],str(x.device))
 if g not in aw:
  aw[g]=ey(x.shape[0],q.shape[0],x.device,(60,1024,256,256,10))
 a=aw[g]
 with X():
  O(a.x,x);O(a.y,J(ab(o.long(),10)));O(a.E,C((x,q)))
 O(a.yd,C((a.y,ah(s,tz,tw,to))));O(a.v,g2(x,o));O(a.C,g3(a.v));a.p.zero_();a.fm.zero_();a._z();at=[p.detach().clone()for p in a.l];a.F.replay();i=a.o.clone();a._z(at);O(a.o,i);a.p.fill_(1.);a.fm[0]=aj;a.fm[1]=ai
 for b in range(32):
  (a.F if b%8==0 else a.M).replay()
 for b in range(12):
  a.U.replay()
 for b in range(44):
  a.V.replay()
 for b in range(480,480):
  (a.N if b%8==0 else a.J).replay()
 with X():
  p=a1(a,o,q)
 a6.allow_tf32=I;return p.argmax(1)
@X()
def ah(q,z,w,bc):
 a6.allow_tf32=I;q=R(rn(J(q))@bc);z=R(z);d=k.mm(q,z.T,out_dtype=f6);d.mul_(-2).add_(Y(Q(J(q)),1)[:,None]).add_(Y(Q(J(z)),1)[None]).clamp_min_(0);d.mul_(-br).exp_().mul_(math.exp(cs));p=a9(d@w/.15,-1);a6.allow_tf32=G;return p
