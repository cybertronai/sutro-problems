"""D5 Ladder; Rasmus,SethTS,spartanhaden,sigkillme0,islamborghini,likelytobelaura,Vincent/Bengio; HazyResearch/TK; sjbaebae."""
v=hasattr;q=zip;b=enumerate;import math as n,os,torch,torch.nn.functional as I,triton,triton.language as A;R=torch.empty_like;U=torch.arange;E=torch.cat;m=torch.backends.cuda.matmul;P=torch.zeros;c=torch.empty;r=torch.randn;W=torch.rsqrt;x=torch.zeros_like;B6=torch.ones;B7=torch.full;w=torch.tensor;S=torch.addmm;V=torch.mm;X=torch.bmm;d=torch.var_mean;L=torch.eye;y=torch.linalg.solve_triangular;Y=torch.cuda.current_stream;e=torch.library.custom_op;O=torch.ops;s=torch.cuda.Stream;C=A.constexpr;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;o,z,f,AL,A0,AM=.02,.1,256,512,.02,.6;AN,AO,AP=(3000,10000),(16,16),14;AQ,Z,AR,A1=.15,2000,15,3.;F=torch.float32;M,K=1,(60,480,480,256,10);J,T,A2,AS,AT=315,2500,.022857142857142857,.8,.5;AU,A3=.2,5;A4,AV,AW,g,AX,A5=.25,.6,2e3,1e-10,.025,11;AY,AZ,A6,A7=.5,.1,.6,.9;Aa,A8=.5,.1;N=torch.bfloat16;Ab,B8,A9=.5,10,4;h=4194304;i,Ac,Ad,Ae,Af=60,3,2048,1024,.5;Ag={'combo_kernels':True,'triton.autotune_pointwise':False,'pattern_matcher':False,'fx_graph_cache':False,'autotune_local_cache':False,'reorder_for_peak_memory':False,'size_asserts':False,'scalar_asserts':False,'alignment_asserts':False};CACHE={};G:C=A.float32;B9:C=A.bfloat16;B=torch.Tensor
def j(a):return torch.linalg.cholesky_ex(a)[0]
def k(x):return x*(x.shape[1]**.5/x.norm(dim=1,keepdim=True).clamp_min(1e-06))**AM
def p(a,b,shift=.0):A=a.square().sum(1,keepdim=True);C=b.square().sum(1,keepdim=True);D=E([a,A,torch.ones_like(A)],1);B=E([2*o*b,torch.full_like(C,-o),shift-o*C],1).T.contiguous();return D,B
def AA(a,b):A,B=p(a,b);return(A@B).exp_()
def AB(z):D,B=p(z,z,A1);A,C=D.half(),B.half();F,G=(D-A.float()).half(),(B-C.float()).half();return E([A,A,F],1),E([C,G,C],0)
def AC(A3,B3,pp,rows,diag):
	C=[]
	for A in rows:D=min(pp,A3.shape[0]-A);B=V(A3[A:A+D],B3[:,A:]);B.exp_();B[:,:D].diagonal().fill_(diag);C.append(B)
	return C
class AD:
	def __init__(o,pans,pp,n):o.pans,o.pp,o.n=pans,pp,n
	def mv(o,dh):
		D=dh;I,A=o.pp,o.n;B=c(A,D.shape[1],device=D.device);G=o.pans[0];E=min(G.shape[0],A);V(G[:E,:A],D,out_dtype=F,out=B[:E])
		if E<A:V(G[:,E:A].T,D[:E],out_dtype=F,out=B[E:A])
		for j in range(1,len(o.pans)):
			a=j*I
			if a>=A:break
			H=o.pans[j];C=min(a+H.shape[0],A);S(B[a:C],H[:,:A-a],D[a:A],out_dtype=F,out=B[a:C])
			if C<A:S(B[C:A],H[:,C-a:A-a].T,D[a:C],out_dtype=F,out=B[C:A])
		return B
	def cross(o,w,lo,hi):
		C=w.abs().amax(0,keepdim=True).clamp_min(1e-30);D=(w/C).half();A=c(hi-lo,w.shape[1],device=w.device)
		for(j,B)in b(o.pans):
			a=j*o.pp
			if a>=lo:break
			if j==0:V(B[:,lo-a:hi-a].T,D[a:a+B.shape[0]],out_dtype=F,out=A)
			else:S(A,B[:,lo-a:hi-a].T,D[a:a+B.shape[0]],out_dtype=F,out=A)
		return A*C
def Ah(ap,d,w,r,nrm,v,vt):a=nrm/(d*ap).sum(0);w=w+d*a;r=r-ap*a;p=r-v@(vt@r);A=(r*p).sum(0);d=p+d*(A/nrm);return w,r,d,A,d.half()
def AE(kmm,jit):A=kmm;B=A.clone();B.diagonal().add_(jit);C=L(len(A),device=A.device);return y(j(B),C,upper=False).T
def AF(op,t,lam,steps,u):
	C=u.T@u;C.diagonal().add_(lam);H=L(len(C),device=u.device);D=u@y(j(C),H,upper=False).T;E=D.T.contiguous();w=x(t);A=t.clone();B=S(A,D,E@A,alpha=-1);F=torch.linalg.vecdot(A,B,dim=0);G=B.half()
	for I in range(steps):w,A,B,F,G=Ah(op.mv(G),B,w,A,F,D,E)
	return w
def AG(a):
	c=a.norm();A=len(a);B=P(2*A,2*A,device=a.device);B[:A,A:]=a/c;B[A:,:A]=L(A,device=a.device);C=1.5*L(2*A,device=a.device)
	for D in range(AR):B=B@S(C,B,B,alpha=-.5)
	return B[:A,A:]*c.sqrt()
def Ai(transform,z,w,fac):
	D=transform;F=fac.shape[0];I=E([w,(w[:,:,None]*z[:,None,:]).reshape(len(z),-1)],1);G=fac.float()@I;c=w.shape[1];A=G[:,c:].reshape(F,c,-1)-G[:,:c,None]*z[:F,None,:];B=D.shape[0];A=(A@D.T).reshape(-1,B);C=AG(A.T@A)
	def H(m):m=(m+m.T)/2;m=m*(B/m.trace().clamp_min(1e-12));return j((1-A0)*m+A0*L(B,device=z.device))
	return H(C@AG(C)),H(C)
def AH(train_x,train_y,test_x):
	c=n.exp(A1);P,Q,R=z*c,.0001*c,c*(1.+z)
	with torch.no_grad():
		A,U=k(train_x.float()),k(test_x.float());F=len(A);B=I.one_hot(train_y.long(),10).float();B=E([B,B[:,:6]],1);G=L(A.shape[1],device=A.device)
		for(m,V)in q(AN,AO):H=A[:m]@G;C=AA(H,H[:f])*c;K=Z if m%Z==0 else m;M,O=AB(H);D=AC(M,O,K,range(0,m,K),R);w=AF(AD(D,K,m),B[:m],P,V,C@AE(C[:f],Q));S=D[0][:AL].clone();S.diagonal().fill_(c);del D;G,W=Ai(G,H,w[:,:10],S)
		J=E([A,U])@G;N=len(J);M,O=AB(J);D=AC(M,O,Z,range(0,F,Z),R);C=AA(J,J[:f])*c;T=AD(D,Z,F);return T.cross(AF(T,B,P,AP,C[:F]@AE(C[:f],Q)),F,N)[:,:10],W
def Aj(train_x,train_y,test_x):
	H=train_y;E=test_x;D=train_x;F='teacher',D.shape,E.shape
	if F not in CACHE:
		A=[D.clone(),H.clone(),E.clone()];G=s();G.wait_stream(Y())
		with torch.cuda.stream(G):
			for I in range(2):AH(*A)
		Y().wait_stream(G);B=torch.cuda.CUDAGraph()
		with torch.cuda.graph(B):C=AH(*A)
		CACHE[F]=B,A,C
	B,A,C=CACHE[F];A[0].copy_(D);A[1].copy_(H);A[2].copy_(E);B.replay();return C[0].clone(),C[1].clone()
def Ak(zq,zv,yv,valid,cls):B,C=cls.shape;H,D=V(*p(zq,zv)).topk(Ad,1);I=valid[D][:,None,:]&(yv[D][:,None,:]==cls[:,:,None]);J,K=H[:,None,:].masked_fill(~I,-float('inf')).topk(i,2);E=D.gather(1,K.reshape(B,-1)).reshape(B,C,i);F=torch.isinf(J);A=zv[torch.where(F,E[:,:,:1],E)].reshape(B*C,i,-1);m=A.mean(1,keepdim=True);A=A-m;G=(zq.repeat_interleave(C,0)[:,None]-m).mT;M=torch.baddbmm(L(i,device=zq.device),A,A.mT);return(G-A.mT@torch.cholesky_solve(A@G,j(M))).mT.square().sum(-1).view(B,C).masked_fill(F[:,:,0],1e30)
def Al(s,za,ty):D=len(ty);F=-s.topk(2).values.diff()[:,0];B=F>=Af;A=F.topk(Ae,largest=False).indices;G=s[A].topk(Ac).indices;C=Ak(za[D:][A],za,E([ty,s.argmax(1)]),E([B.new_ones(D),B]),G);H=torch.softmax((C.min(1,keepdim=True).values-C)/C.min(1).values.mean(),1);return s.index_add(0,A,x(s[A]).scatter_(1,G,H)*(~B[A]).float()[:,None])
@torch.no_grad()
def Am(x,y,k=10,rank=4):G=len(x);A=y.long();B=torch.bincount(A,minlength=10);K=(int(B.max())+127)//128*128;C=A.argsort(stable=True);H=B.cumsum(0)-B;I=U(K,device=x.device);L=I[None,:]<B[:,None];J=C[(H[:,None]+I[None,:]).clamp_max(G-1)];D=x[J];E=X(D,D.mT)-.5*D.square().sum(-1)[:,None,:];E.masked_fill_(~L[:,None,:],-float('inf'));E.diagonal(dim1=-2,dim2=-1).fill_(-float('inf'));M=R(C).scatter_(0,C,U(G,device=x.device));N=M-H[A];O=E[A,N].topk(k,dim=-1).indices;P=J[A[:,None],O];F=x[P]-x[:,None];Q=torch.linalg.eigh(F@F.mT)[1][...,-rank:];return F.mT@Q/k**.5
def AI(u):A,B=d(u,dim=-2,unbiased=False,keepdim=True);return(u-B)*W(A+g)
def l(t):return I.leaky_relu(t,.1)
def An(u,lat,p):return t(AI(u),lat,p)
def t(v,lat,p):A=lat;B=v*A;C=l(v*p[0]+A*p[2]+B*p[4]+p[6]);D=l(v*p[1]+A*p[3]+B*p[5]+p[7]);E=l(C*p[8]+D*p[10]+p[12]);F=l(C*p[9]+D*p[11]+p[13]);return E*p[14]+F*p[15]+p[16]
@triton.jit
def Q(p,o,m):return A.load(p+o,mask=m,other=.0)
@triton.jit
def D(w,a,f=1.):A.store(w,a*f)
@triton.jit
def H(P,j,S,c,cm):return A.load(P+j*S+c,mask=cm,other=.0).to(G)[None,:]
@triton.jit
def Ao(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39):return _0+_20,_1+_21,_2+_22,_3+_23,_4+_24,_5+_25,_6+_26,_7+_27,_8+_28,_9+_29,_10+_30,_11+_31,_12+_32,_13+_33,_14+_34,_15+_35,_16+_36,_17+_37,_18+_38,_19+_39
@triton.jit
def Ap(_0,_1,_2,_3,_4,_5):return _0+_3,_1+_4,_2+_5
@triton.jit
def Aq(GO,U,MEAN,RSTD,LAT,P,DU,DLAT,DP,WS,DV,B,S,ROWS,NRS,NCB:C,BC:C,SUB:C,SB:C,PHASE:C,NR:C,NE:C):
	A2:C=U.dtype.element_ty;Z=A.program_id(0);Ae=Z%NCB;AC=Z//NCB;c=Ae*BC+A.arange(0,BC);cm=c<S;R=AC*ROWS;L=A.minimum(R+ROWS,B);AD=A.arange(0,SUB);Af=NCB*NRS;A3=1./B.to(G);Ag=A.load(MEAN+c).to(G);AE=A.load(RSTD+c).to(G);A4=AE[None,:];AF=(-Ag*AE)[None,:];AG=H(P,0,S,c,cm);AH=H(P,1,S,c,cm);AI=H(P,2,S,c,cm);AJ=H(P,3,S,c,cm);AK=H(P,4,S,c,cm);AL=H(P,5,S,c,cm);Ah=H(P,6,S,c,cm);Ai=H(P,7,S,c,cm);AM=H(P,8,S,c,cm);AN=H(P,9,S,c,cm);AO=H(P,10,S,c,cm);AP=H(P,11,S,c,cm);Aj=H(P,12,S,c,cm);Ak=H(P,13,S,c,cm);T=A.load(P+14*S+c).to(G);V=A.load(P+15*S+c).to(G);Al=T[None,:]*AM;Am=V[None,:]*AN;An=T[None,:]*AO;Aq=V[None,:]*AP
	if PHASE==1:
		E=A.zeros([SUB,BC],G);a,b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x=E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E,E;F=R+AD;m=(F<L)[:,None];o=F[:,None]*S+c[None,:];AQ=Q(GO,o,m);W=Q(U,o,m);AR=Q(LAT,o,m)
		for A5 in range(R,L,SUB):A6=AQ.to(G);A7=W.to(G);M=AR.to(G);X,y=m,o;F=A5+SUB+AD;m=(F<L)[:,None];o=F[:,None]*S+c[None,:];AQ=Q(GO,o,m);W=Q(U,o,m);AR=Q(LAT,o,m);I=A7*A4+AF;z=I*M;AS=I*AG+M*AI+z*AK+Ah;AT=I*AH+M*AJ+z*AL+Ai;AU=A.where(AS>0,1.,.1);AV=A.where(AT>0,1.,.1);A0=AS*AU;A1=AT*AV;AW=A0*AM+A1*AO+Aj;AX=A0*AN+A1*AP+Ak;N=A6*A.where(AW>0,1.,.1);O=A6*A.where(AX>0,1.,.1);J=(N*Al+O*Am)*AU;K=(N*An+O*Aq)*AV;AY=J*AK+K*AL;Y=J*AG+K*AH+AY*M;Ar=J*AI+K*AJ+AY*I;A.store(DLAT+y,Ar.to(A2),mask=X);A.store(DV+y,Y.to(A2),mask=X);a+=J*I;b+=K*I;d+=J*M;e+=K*M;f+=J*z;g+=K*z;h+=J;i+=K;j+=N*A0;k+=O*A0;l+=N*A1;n+=O*A1;p+=N;q+=O;r+=N*AW;s+=O*AX;t+=A6;u+=Y;v+=Y*I;x+=A.where(X,I,.0)
		a,b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x=A.reduce((a,b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x),0,Ao);w=WS+AC*20*S+c;D(w,a);D(w+S,b);D(w+2*S,d);D(w+3*S,e);D(w+4*S,f);D(w+5*S,g);D(w+6*S,h);D(w+7*S,i);D(w+8*S,j,T);D(w+9*S,k,V);D(w+10*S,l,T);D(w+11*S,n,V);D(w+12*S,p,T);D(w+13*S,q,V);D(w+14*S,r);D(w+15*S,s);D(w+16*S,t);D(w+17*S,u);D(w+18*S,v);D(w+19*S,x)
	else:
		AZ=A.cdiv(17*S,Af);A8=Z*AZ+A.arange(0,NE);Aa=A8<A.minimum((Z+1)*AZ,17*S);A9=A.arange(0,NR);Ab=A9<NRS;As=A.load(WS+A9[:,None]*20*S+A8[None,:],mask=Ab[:,None]&Aa[None,:],other=.0);A.store(DP+A8,A.sum(As,0),mask=Aa);AA=A9[:,None]*20*S+c[None,:];AB=Ab[:,None]&cm[None,:];At=A.load(WS+AA+17*S,mask=AB,other=.0);Au=A.load(WS+AA+18*S,mask=AB,other=.0);Av=A.load(WS+AA+19*S,mask=AB,other=.0);Aw,Ax,Ay=A.reduce((At,Au,Av),0,Ap);Az=(Aw*A3)[None,:];A_=(Ax*A3)[None,:];B0=(Ay*A3)[None,:];Ac=A.arange(0,SB);F=R+Ac;m=(F<L)[:,None];o=F[:,None]*S+c[None,:];W=A.load(U+o,mask=m,other=.0);Ad=A.load(DV+o,mask=m,other=.0)
		for A5 in range(R,L,SB):A7=W.to(G);Y=Ad.to(G);X,y=m,o;F=A5+SB+Ac;m=(F<L)[:,None];o=F[:,None]*S+c[None,:];W=A.load(U+o,mask=m,other=.0);Ad=A.load(DV+o,mask=m,other=.0);I=A7*A4+AF;A.store(DU+y,(A4*(Y-Az-(I-B0)*A_)).to(A2),mask=X)
Ar={480:(16,16,4,3,64),60:(4,16,2,4,64),256:(16,16,2,4,64)}
def As(go:B,u:B,mean:B,rstd:B,lat:B,pb:B)->tuple[B,B,B]:
	I=rstd;H=mean;A=lat;B,S=u.shape[-2:];BC,SUB,N,J,SB=Ar[S];E=S//BC;G=-(-B//max(1,(torch.cuda.get_device_properties(u.device).multi_processor_count*J-(4 if J>1 else 0))//E));G=-(-G//SB)*SB;C=-(-B//G);go,u,H,I,pb=go.contiguous(),u.contiguous(),H.contiguous(),I.contiguous(),pb.contiguous()
	if A.stride(-1)!=1 or A.stride(-2)!=S:A=A.contiguous()
	D=lambda*A,dtype=u.dtype:c(*A,device=u.device,dtype=dtype);K,L,M,O,P=D(u.shape),D(u.shape),D(pb.shape,dtype=F),D(C,20,S,dtype=F),D(u.shape)
	for Q in(1,2):Aq[E*C,](go,u,H,I,A,pb,K,L,M,O,P,B,S,G,C,NCB=E,BC=BC,SUB=SUB,SB=SB,PHASE=Q,num_warps=N,NR=triton.next_power_of_2(C),NE=triton.next_power_of_2(triton.cdiv(17*S,E*C)))
	return K,L,M
def At():
	if not v(O.d5bottom,'comb_bwd'):A=lambda t,d:t.new_empty(t.shape,dtype=d);e('d5bottom::comb_bwd',As,mutates_args=()).register_fake(lambda go,u,mean,rstd,lat,pb:(A(u,u.dtype),A(u,u.dtype),A(pb,F)))
class AJ(torch.autograd.Function):
	@staticmethod
	def forward(ctx,u,lat,p):A=p.to(u.dtype);D,B=d(u,dim=-2,unbiased=False,keepdim=True);C=W(D+g);E=t((u-B)*C,lat,A);ctx.save_for_backward(u,B,C,lat,A);return E
	@staticmethod
	def backward(ctx,go):A,B,C,D,E=ctx.saved_tensors;F,G,H=O.d5bottom.comb_bwd(go,A,B,C,D,E);return F,G,H
def a(h,w,act=None):
	A=act
	if A is torch.bfloat16:return AK.apply(h,w,torch.bfloat16)
	if A is None and w.shape[-1]==60 and h.requires_grad:return AK.apply(h,w,F)
	B=h.shape
	with torch.autocast('cuda',dtype=A or torch.float16,enabled=A is not None):C=X(h.reshape(B[0],-1,B[-1]),w)
	return(C if A is not None else C.float()).reshape(*B[:-1],w.shape[-1])
class Au:
	def __init__(o,n,nq,device):
		B=device;o.n,o.nq,o.d=n,nq,B;A=lambda*A:P(*A,device=B);o.e=[A(M,a,B)for(a,B)in q(K[:-1],K[1:])];o.D=[A(M,B,a)for(a,B)in q(K[:-1],K[1:])];o.b=[A(M,1,B)for B in K[1:]];o.g=A(M,1,K[-1]);o.c=[A(17,M,1,B)for B in K];o.a=o.e+o.D+o.b+[o.g]+o.c
		for p in o.a:p.requires_grad_(True)
		o.x=A(n,K[0]);o.y=A(n,10);o.p=A(n+nq,K[0]);o.P=A(n+nq,10);o.T=A(n,K[0],A9);o.s=A(M,n).long();o.t=P((),dtype=torch.long,device=B);o.iu=P(J,M,T,dtype=torch.long,device=B);C=U(J,device=B,dtype=F)/J;o.l=A2*((1-C)/(1-AT)).clamp(max=1.);E=((A7-C)/(A7-A6)).clamp(0,1);o.m=AY*torch.where(C<AZ,.0,torch.where(C<A6,1.,E));o.r=w(A2,device=B);o.h=P(int(AU*nq),dtype=torch.long,device=B);o.B=A(h).to(N);o.f=A(J,len(K)).long();o.u=A(J,M,T);o.v=A(J,M,T,1,A9);o.W=torch.optim.Adam(o.a,lr=o.r,betas=(AS,.999),foreach=True,capturable=True);o.O=torch.compile(o.W.step);o.F=torch.compile(o.V,dynamic=False,options=Ag);o.U=torch.compile(B5,dynamic=False);o.z=w([.1,-.1,.9,-.9,0,0,0,0,1,0,0,1,0,0,1/1.01,-1/1.01,0],device=B)[:,None,None,None];o.R();D=s();D.wait_stream(Y())
		with torch.cuda.stream(D):
			for G in range(3):o.W.zero_grad(set_to_none=True);o.S()
		Y().wait_stream(D);o.G=torch.cuda.CUDAGraph();o.W.zero_grad(set_to_none=True)
		with torch.cuda.graph(o.G):
			for G in range(A3):
				for p in o.a:p.grad=None
				o.S()
	def L(o,xl,yl,xu,tu,tg,ofs,lb):
		F=ofs;F=list(F.unbind());K=lambda t:o.B[(F.pop()+U(t.numel(),device=t.device))%h].view(t.shape).to(t.dtype);P,Q=[A.to(N)for A in o.b],o.g.to(N);R=[p.to(N)for p in o.c];xl=xl+Ab*(tg*o.v.index_select(0,o.t.view(1)).squeeze(0)).sum(-1);A=torch.stack((xl,xu),1);A=A+A4*K(A);E=[A[:,1]];A=A.to(N);G=len(o.e)-1
		for(B,w)in b(o.e):C=Ax(a(A,w,N),2)if B<G else AI(a(A,w,N));C=C+(A4*(1-o.t.float()/J).clamp_min(.2).pow(.5)*K(C).float()).to(C.dtype);E.append(C[:,1]);L=P[B][:,None];A=(C+L)*Q[:,None]if B==G else I.relu(C+L)
		H=A.float();M=-(yl*I.log_softmax(H[:,0],-1)).sum(-1);S=(o.t<J*.5)|lb[...,0];T=-((tu*I.log_softmax(H[:,1],-1)).sum(-1)*S).mean(-1).sum();O=I.softmax(H[:,1],-1);D=An((O+(lb*(tu-O)).detach()).to(N),E[-1],R[-1])
		for B in range(G,0,-1):D=AJ.apply(a(D,o.D[B],N),E[B],o.c[B])
		D=AJ.apply(a(D.float(),o.D[0]),E[0],o.c[0]);V=(D.float()-xu).square().mean((1,2)).sum();return M.mean(-1).sum()+Aa*T+AW*V,M.detach()
	def V(o):A=o.t.view(1);m=o.m.index_select(0,A);c=O.d5e.c(o.s,m);B=torch.searchsorted(c,o.u.index_select(0,A).squeeze(0)*c[:,-1:]).clamp_max(o.n-1);C=o.iu.index_select(0,A).squeeze(0);o.r.copy_(o.l.index_select(0,A).squeeze(0));D=o.T[B];E,F=o.L(o.x[B],o.y[B],o.p[C],o.P[C],D,o.f.index_select(0,A).squeeze(0),(C<o.n)[...,None]);O.d5e.u(o.s,B,F,o.t);o.t.add_(1);return E
	def S(o):o.F().backward();o.O()
	@torch.no_grad()
	def R(o):
		A=torch.Generator(device=o.d);A.manual_seed(A5)
		for w in o.e+o.D:w.copy_(r(w.shape,generator=A,device=o.d)*w.shape[1]**-.5)
		for(a,H)in q(o.e,o.D):H.copy_(a.mT*n.sqrt(a.shape[1]/a.shape[2]))
		B=[0,1,2,3,4,5,8,9,10,11,14,15]
		for p in o.c:p.zero_();p[B]=r(len(B),*p.shape[1:],generator=A,device=o.d)*AX;p.copy_(o.z)
		for p in o.b:p.zero_()
		o.g.fill_(1.)
		for p in o.a:
			C=o.W.state.get(p)
			if C:C['exp_avg'].zero_();C['exp_avg_sq'].zero_();C['step'].zero_()
		o.t.zero_();o.s.fill_(1065353216);F=U(o.n+o.nq,device=o.d);G=[]
		for(D,B)in((J//2,F),(J-J//2,E([F,o.h]))):I=torch.rand(n.ceil(D*T/len(B)),len(B),generator=A,device=o.d).argsort(-1);G.append(B[I.view(-1)[:D*T]].view(D,1,T))
		o.iu.copy_(E(G));o.B.copy_(r(h,generator=A,device=o.d));o.f.copy_(torch.randint(h,o.f.shape,generator=A,device=o.d));o.u.uniform_(generator=A);o.v.normal_(generator=A);torch.cuda.manual_seed(A5+1)
	def A(o,i,t,b,gm):return(t+b)*gm if i==len(o.e)-1 else I.relu(t+b)
	@torch.no_grad()
	def Q(o,q):
		A=E((o.x,q))[None];C=[]
		for(B,w)in b(o.e):D=a(A,w);G,H=d(D[:,:o.n],dim=-2,unbiased=False,keepdim=True);F=(D-H)*W(G+g);C.append((F+o.b[B])[0]);A=o.A(B,F,o.b[B],o.g)
		return C+[A[0]]
Av,Aw=60,40
def guard(s,p,e):B,A=s.topk(2,1);A=A[:,0];C=B[:,0]-B[:,1]>=AQ;c=P(100,device=s.device).index_add_(0,p*10+A,C.float());c[::11]=0;return torch.where(C&((c.max()>Av)|(e>Aw)),A,p)
def ladder_kernel(train_x,train_y,test_x):
	Q=test_x;M=train_x;D=train_y;os.environ.setdefault('TRITON_CACHE_DIR','/tmp/triton-cache');os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/inductor-cache');At();encoded_register();B4();m.allow_tf32=False;G,R=Aj(M,D,Q);m.allow_tf32=True
	with torch.no_grad():B,C=k(M.float())@R,k(Q.float())@R;c=(M.float()*AV).std()/B.std();B,C=B*c,C*c
	N='ladder',len(B),len(C)
	if N not in CACHE:CACHE[N]=Au(B.shape[0],C.shape[0],B.device)
	A=CACHE[N]
	with torch.no_grad():A.x.copy_(B);A.T.copy_(Am(B,D));A.y.copy_(I.one_hot(D.long(),10).float());A.p.copy_(E((B,C)));A.P.copy_(E((A.y,I.softmax(G/A8,1))));S=G.topk(2,1).values;A.h.copy_((S[:,0]-S[:,1]).topk(len(A.h),largest=False).indices+len(B))
	A.R()
	for T in range(J//A3):A.G.replay()
	with torch.no_grad():H=A.Q(C);U,V=[A[:len(B)]for A in H],[A[len(B):]for A in H];d=(U[-1].argmax(1)!=D).sum();e=E([B,C]);K=E([U[2],V[2]]).float();K=K*e.norm()/K.norm();m.allow_tf32=False;p=(1-.2)*I.softmax(V[-1],-1)+.2*I.softmax(G/A8,1);W=[A.p]+[C-A.b[B][0]for(B,C)in b(H[:-1])];X=[w[0]for w in A.D];Y=[A[:,0]for A in A.c];T,f=A.U(I.softmax(H[-1][:len(B)],-1),[A[:len(B)]for A in W],X,Y);F=(-p.topk(2).values.diff()[:,0]).topk(1024,largest=False).indices;Z=p[F].topk(3).indices;g=[A[len(B):][F,None].expand(-1,3,-1).reshape(-1,A.shape[-1])for A in W];L,T=A.U(I.one_hot(Z.flatten(),10).float(),g,X,Y,f);L=L.reshape(1024,3,60);a=C[F,None];O=(L-a).square().mean(-1);P=(L*a).sum(-1);h=(O[:,1]-O[:,0]).abs().median().clamp_min(1e-09);i=(P[:,1]-P[:,0]).abs().median().clamp_min(1e-09);j=(-.5*O/h+.5*P/i).softmax(-1);p[F]*=1-.5;p[F[:,None],Z]+=.5*j
	return guard(G,Al(p,K,D.long()).argmax(1),d)
def Ax(u,groups):B=groups;a=u.reshape(*u.shape[:-2],B,u.shape[-2]//B,u.shape[-1]);f=a.float();A=f.mean(-2,keepdim=True);C=(f.square().mean(-2,keepdim=True)-A.square()).clamp_min(0).to(a.dtype);A=A.to(a.dtype);return((a-A)*W(C+g)).reshape_as(u)
@triton.jit
def Ay(S,P,T,N:C,K:C,B:C=512):C,D=A.program_id(0)//K,A.program_id(0)%K;E=A.arange(0,B);j=D*B+E;F=A.load(S+C*N+j,j<N,0);H=(F&4294967295).to(A.uint32).to(G,bitcast=True);p=A.cumsum(H,0);A.store(P+C*N+j,p,j<N);I=A.sum(A.where(E==A.minimum(B,N-D*B)-1,p,.0),0);A.store(T+C*K+D,I)
def Az(state:B,mix:B)->B:B=state;m,C=B.shape;A=triton.cdiv(C,512);p=R(B,dtype=F);D=c((m,A),device=B.device);E=R(B,dtype=F);Ay[m*A,](B,p,D,C,A,num_warps=4);B1[m*A,](p,D,mix,E,C,A,triton.next_power_of_2(A),num_warps=4);return E
@triton.jit
def A_(S,I,CE,T,N:C,BATCH:C,L:C,BLOCK:C=256):B=A.program_id(0)*BLOCK+A.arange(0,BLOCK);C=B<L;D=A.load(I+B,C,0);E=B//BATCH;F=A.load(CE+B,C,.0).to(G);H=F.to(A.uint32,bitcast=True).to(A.int64);J=A.load(T).to(A.int64)*BATCH+B%BATCH+1;K=J<<32|H;A.atomic_max(S+E*N+D,K,C,sem='relaxed')
def B0(state:B,indices:B,ce:B,counter:B)->None:C=indices;B=state;m,D=B.shape;A=C.shape[1];A_[triton.cdiv(m*A,256),](B,C,ce,counter,D,A,m*A)
@triton.jit
def B1(P,T,MIX,C,N:C,K:C,R:C,B:C=512):E,G=A.program_id(0)//K,A.program_id(0)%K;F=A.arange(0,R);H=A.load(T+E*K+F,F<K,.0);J=A.sum(H,0);L=A.sum(A.where(F<G,H,.0),0);D=G*B+A.arange(0,B);p=A.load(P+E*N+D,D<N,.0);I=A.load(MIX);c=I*(p+L)/J+(1-I)*(D+1)/N;A.store(C+E*N+D,c,D<N)
def encoded_register():
	if not v(O.d5e,'c'):e('d5e::c',Az,mutates_args=()).register_fake(lambda state,mix:R(state,dtype=F));e('d5e::u',B0,mutates_args={'state'}).register_fake(lambda state,indices,ce,counter:None)
u={}
def B2(device):
	B=device;A=B.index
	if A not in u:u[A]=s(device=B)
	return u[A]
def B3(go:B,h:B,w:B,need_h:bool)->tuple[B,B]:
	A=go;A=A.contiguous();h=h.contiguous();w=w.contiguous();C=Y(h.device);B=B2(h.device);B.wait_stream(C)
	with torch.cuda.stream(B):D=X(h.transpose(1,2),A);h.record_stream(B);A.record_stream(B)
	E=X(A,w.transpose(1,2))if need_h else h.new_empty((0,));C.wait_stream(B);D.record_stream(C);return E,D
def _(go,h,w,need_h):return R(h)if need_h else h.new_empty((0,)),w.new_empty(w.shape)
def B4():
	if not v(O.d5pc,'backward'):e('d5pc::backward',B3,mutates_args=()).register_fake(_)
class AK(torch.autograd.Function):
	@staticmethod
	def forward(ctx,h,w,act):A=h.shape;B=h.reshape(A[0],-1,A[-1]).to(act);C=w.to(act);ctx.save_for_backward(B,C);ctx.shape=A;return X(B,C).reshape(*A[:-1],w.shape[-1])
	@staticmethod
	def backward(ctx,go):A=ctx;B,w=A.saved_tensors;C,D=O.d5pc.backward(go.reshape(B.shape[0],-1,w.shape[-1]),B,w,A.needs_input_grad[0]);return C.reshape(A.shape)if A.needs_input_grad[0]else None,D.float(),None
def B5(r,latent,weights,params,stats=None):
	D=stats;E=[]
	for(j,A)in b(range(4,-1,-1)):
		if A<4:r=r@weights[A]
		if D is None:F,B=d(r,dim=0,unbiased=False,keepdim=True);C=W(F+1e-10)
		else:B,C=D[j]
		E.append((B,C));r=t((r-B)*C,latent[A],params[A])
	return r,E