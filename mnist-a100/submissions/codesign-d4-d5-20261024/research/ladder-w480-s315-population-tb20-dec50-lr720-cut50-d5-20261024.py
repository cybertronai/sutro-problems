"""D5 Ladder; Rasmus,SethTS,spartanhaden,sigkillme0,islamborghini,likelytobelaura,Vincent/Bengio; HazyResearch/TK; sjbaebae."""
u=hasattr;t=zip;b=enumerate;import math as v,os,torch,torch.nn.functional as I,triton,triton.language as A;R=torch.empty_like;U=torch.arange;D=torch.cat;m=torch.backends.cuda.matmul;P=torch.zeros;c=torch.empty;n=torch.randn;W=torch.rsqrt;x=torch.zeros_like;B5=torch.ones;B6=torch.full;AK=torch.tensor;S=torch.addmm;V=torch.mm;X=torch.bmm;d=torch.var_mean;L=torch.eye;y=torch.linalg.solve_triangular;Y=torch.cuda.current_stream;e=torch.library.custom_op;O=torch.ops;q=torch.cuda.Stream;E=A.constexpr;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;o,z,f,AL,A0,AM=.02,.1,256,512,.02,.6;AN,AO,AP=(3000,10000),(16,16),14;AQ,Z,AR,A1=.15,2000,15,3.;G=torch.float32;M,K=1,(60,480,480,256,10);J,T,w,AS,AT=315,2500,.022857142857142857,.8,.5;AU,A2=.2,5;A3,AV,AW,g,AX,A4=.25,.6,2e3,1e-10,.025,11;AY,AZ,A5,A6=.5,.1,.6,.9;Aa,A7=.5,.1;N=torch.bfloat16;Ab,B7,A8=.5,10,4;h=4194304;i,Ac,Ad,Ae,Af=60,3,2048,1024,.5;Ag={'combo_kernels':True,'triton.autotune_pointwise':False,'pattern_matcher':False,'fx_graph_cache':False,'autotune_local_cache':False,'reorder_for_peak_memory':False,'size_asserts':False,'scalar_asserts':False,'alignment_asserts':False};CACHE={};F:E=A.float32;B8:E=A.bfloat16;B=torch.Tensor
def j(a):return torch.linalg.cholesky_ex(a)[0]
def k(x):return x*(x.shape[1]**.5/x.norm(dim=1,keepdim=True).clamp_min(1e-06))**AM
def p(a,b,shift=.0):A=a.square().sum(1,keepdim=True);C=b.square().sum(1,keepdim=True);E=D([a,A,torch.ones_like(A)],1);B=D([2*o*b,torch.full_like(C,-o),shift-o*C],1).T.contiguous();return E,B
def A9(a,b):A,B=p(a,b);return(A@B).exp_()
def AA(z):E,B=p(z,z,A1);A,C=E.half(),B.half();F,G=(E-A.float()).half(),(B-C.float()).half();return D([A,A,F],1),D([C,G,C],0)
def AB(A3,B3,pp,rows,diag):
	C=[]
	for A in rows:D=min(pp,A3.shape[0]-A);B=V(A3[A:A+D],B3[:,A:]);B.exp_();B[:,:D].diagonal().fill_(diag);C.append(B)
	return C
class AC:
	def __init__(o,pans,pp,n):o.pans,o.pp,o.n=pans,pp,n
	def mv(o,dh):
		D=dh;I,A=o.pp,o.n;B=c(A,D.shape[1],device=D.device);F=o.pans[0];E=min(F.shape[0],A);V(F[:E,:A],D,out_dtype=G,out=B[:E])
		if E<A:V(F[:,E:A].T,D[:E],out_dtype=G,out=B[E:A])
		for j in range(1,len(o.pans)):
			a=j*I
			if a>=A:break
			H=o.pans[j];C=min(a+H.shape[0],A);S(B[a:C],H[:,:A-a],D[a:A],out_dtype=G,out=B[a:C])
			if C<A:S(B[C:A],H[:,C-a:A-a].T,D[a:C],out_dtype=G,out=B[C:A])
		return B
	def cross(o,w,lo,hi):
		C=w.abs().amax(0,keepdim=True).clamp_min(1e-30);D=(w/C).half();A=c(hi-lo,w.shape[1],device=w.device)
		for(j,B)in b(o.pans):
			a=j*o.pp
			if a>=lo:break
			if j==0:V(B[:,lo-a:hi-a].T,D[a:a+B.shape[0]],out_dtype=G,out=A)
			else:S(A,B[:,lo-a:hi-a].T,D[a:a+B.shape[0]],out_dtype=G,out=A)
		return A*C
def Ah(ap,d,w,r,nrm,v,vt):a=nrm/(d*ap).sum(0);w=w+d*a;r=r-ap*a;p=r-v@(vt@r);A=(r*p).sum(0);d=p+d*(A/nrm);return w,r,d,A,d.half()
def AD(kmm,jit):A=kmm;B=A.clone();B.diagonal().add_(jit);C=L(len(A),device=A.device);return y(j(B),C,upper=False).T
def AE(op,t,lam,steps,u):
	C=u.T@u;C.diagonal().add_(lam);H=L(len(C),device=u.device);D=u@y(j(C),H,upper=False).T;E=D.T.contiguous();w=x(t);A=t.clone();B=S(A,D,E@A,alpha=-1);F=torch.linalg.vecdot(A,B,dim=0);G=B.half()
	for I in range(steps):w,A,B,F,G=Ah(op.mv(G),B,w,A,F,D,E)
	return w
def AF(a):
	c=a.norm();A=len(a);B=P(2*A,2*A,device=a.device);B[:A,A:]=a/c;B[A:,:A]=L(A,device=a.device);C=1.5*L(2*A,device=a.device)
	for D in range(AR):B=B@S(C,B,B,alpha=-.5)
	return B[:A,A:]*c.sqrt()
def Ai(transform,z,w,fac):
	E=transform;F=fac.shape[0];I=D([w,(w[:,:,None]*z[:,None,:]).reshape(len(z),-1)],1);G=fac.float()@I;c=w.shape[1];A=G[:,c:].reshape(F,c,-1)-G[:,:c,None]*z[:F,None,:];B=E.shape[0];A=(A@E.T).reshape(-1,B);C=AF(A.T@A)
	def H(m):m=(m+m.T)/2;m=m*(B/m.trace().clamp_min(1e-12));return j((1-A0)*m+A0*L(B,device=z.device))
	return H(C@AF(C)),H(C)
def AG(train_x,train_y,test_x):
	c=v.exp(A1);P,Q,R=z*c,.0001*c,c*(1.+z)
	with torch.no_grad():
		A,U=k(train_x.float()),k(test_x.float());F=len(A);B=I.one_hot(train_y.long(),10).float();B=D([B,B[:,:6]],1);G=L(A.shape[1],device=A.device)
		for(m,V)in t(AN,AO):H=A[:m]@G;C=A9(H,H[:f])*c;K=Z if m%Z==0 else m;M,O=AA(H);E=AB(M,O,K,range(0,m,K),R);w=AE(AC(E,K,m),B[:m],P,V,C@AD(C[:f],Q));S=E[0][:AL].clone();S.diagonal().fill_(c);del E;G,W=Ai(G,H,w[:,:10],S)
		J=D([A,U])@G;N=len(J);M,O=AA(J);E=AB(M,O,Z,range(0,F,Z),R);C=A9(J,J[:f])*c;T=AC(E,Z,F);return T.cross(AE(T,B,P,AP,C[:F]@AD(C[:f],Q)),F,N)[:,:10],W
def Aj(train_x,train_y,test_x):
	H=train_y;E=test_x;D=train_x;F='teacher',D.shape,E.shape
	if F not in CACHE:
		A=[D.clone(),H.clone(),E.clone()];G=q();G.wait_stream(Y())
		with torch.cuda.stream(G):
			for I in range(2):AG(*A)
		Y().wait_stream(G);B=torch.cuda.CUDAGraph()
		with torch.cuda.graph(B):C=AG(*A)
		CACHE[F]=B,A,C
	B,A,C=CACHE[F];A[0].copy_(D);A[1].copy_(H);A[2].copy_(E);B.replay();return C[0].clone(),C[1].clone()
def Ak(zq,zv,yv,valid,cls):B,C=cls.shape;H,D=V(*p(zq,zv)).topk(Ad,1);I=valid[D][:,None,:]&(yv[D][:,None,:]==cls[:,:,None]);J,K=H[:,None,:].masked_fill(~I,-float('inf')).topk(i,2);E=D.gather(1,K.reshape(B,-1)).reshape(B,C,i);F=torch.isinf(J);A=zv[torch.where(F,E[:,:,:1],E)].reshape(B*C,i,-1);m=A.mean(1,keepdim=True);A=A-m;G=(zq.repeat_interleave(C,0)[:,None]-m).mT;M=torch.baddbmm(L(i,device=zq.device),A,A.mT);return(G-A.mT@torch.cholesky_solve(A@G,j(M))).mT.square().sum(-1).view(B,C).masked_fill(F[:,:,0],1e30)
def Al(s,za,ty):E=len(ty);F=-s.topk(2).values.diff()[:,0];B=F>=Af;A=F.topk(Ae,largest=False).indices;G=s[A].topk(Ac).indices;C=Ak(za[E:][A],za,D([ty,s.argmax(1)]),D([B.new_ones(E),B]),G);H=torch.softmax((C.min(1,keepdim=True).values-C)/C.min(1).values.mean(),1);return s.index_add(0,A,x(s[A]).scatter_(1,G,H)*(~B[A]).float()[:,None])
@torch.no_grad()
def Am(x,y,k=10,rank=4):G=len(x);A=y.long();B=torch.bincount(A,minlength=10);K=(int(B.max())+127)//128*128;C=A.argsort(stable=True);H=B.cumsum(0)-B;I=U(K,device=x.device);L=I[None,:]<B[:,None];J=C[(H[:,None]+I[None,:]).clamp_max(G-1)];D=x[J];E=X(D,D.mT)-.5*D.square().sum(-1)[:,None,:];E.masked_fill_(~L[:,None,:],-float('inf'));E.diagonal(dim1=-2,dim2=-1).fill_(-float('inf'));M=R(C).scatter_(0,C,U(G,device=x.device));N=M-H[A];O=E[A,N].topk(k,dim=-1).indices;P=J[A[:,None],O];F=x[P]-x[:,None];Q=torch.linalg.eigh(F@F.mT)[1][...,-rank:];return F.mT@Q/k**.5
def AH(u):A,B=d(u,dim=-2,unbiased=False,keepdim=True);return(u-B)*W(A+g)
def l(t):return I.leaky_relu(t,.1)
def An(u,lat,p):return s(AH(u),lat,p)
def s(v,lat,p):A=lat;B=v*A;C=l(v*p[0]+A*p[2]+B*p[4]+p[6]);D=l(v*p[1]+A*p[3]+B*p[5]+p[7]);E=l(C*p[8]+D*p[10]+p[12]);F=l(C*p[9]+D*p[11]+p[13]);return E*p[14]+F*p[15]+p[16]
@triton.jit
def Q(p,o,m):return A.load(p+o,mask=m,other=.0)
@triton.jit
def C(w,a,f=1.):A.store(w,a*f)
@triton.jit
def H(P,j,S,c,cm):return A.load(P+j*S+c,mask=cm,other=.0).to(F)[None,:]
@triton.jit
def Ao(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39):return _0+_20,_1+_21,_2+_22,_3+_23,_4+_24,_5+_25,_6+_26,_7+_27,_8+_28,_9+_29,_10+_30,_11+_31,_12+_32,_13+_33,_14+_34,_15+_35,_16+_36,_17+_37,_18+_38,_19+_39
@triton.jit
def Ap(GO,U,MEAN,RSTD,LAT,P,DU,DLAT,DP,WS,DV,B,S,ROWS,NRS,NCB:E,BC:E,SUB:E,SB:E,PHASE:E):
	A3:E=U.dtype.element_ty;a=A.program_id(0);Ag=a%NCB;AA=a//NCB;c=Ag*BC+A.arange(0,BC);cm=c<S;R=AA*ROWS;L=A.minimum(R+ROWS,B);AB=A.arange(0,SUB);Ah=NCB*NRS;A4=1./B.to(F);Ai=A.load(MEAN+c).to(F);AC=A.load(RSTD+c).to(F);A5=AC[None,:];AD=(-Ai*AC)[None,:];AE=H(P,0,S,c,cm);AF=H(P,1,S,c,cm);AG=H(P,2,S,c,cm);AH=H(P,3,S,c,cm);AI=H(P,4,S,c,cm);AJ=H(P,5,S,c,cm);Aj=H(P,6,S,c,cm);Ak=H(P,7,S,c,cm);AK=H(P,8,S,c,cm);AL=H(P,9,S,c,cm);AM=H(P,10,S,c,cm);AN=H(P,11,S,c,cm);Al=H(P,12,S,c,cm);Am=H(P,13,S,c,cm);T=A.load(P+14*S+c).to(F);V=A.load(P+15*S+c).to(F);An=T[None,:]*AK;Ap=V[None,:]*AL;Aq=T[None,:]*AM;Ar=V[None,:]*AN
	if PHASE==1:
		D=A.zeros([SUB,BC],F);b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x,y=D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D;G=R+AB;m=(G<L)[:,None];o=G[:,None]*S+c[None,:];AO=Q(GO,o,m);W=Q(U,o,m);AP=Q(LAT,o,m)
		for A6 in range(R,L,SUB):A7=AO.to(F);A8=W.to(F);M=AP.to(F);X,z=m,o;G=A6+SUB+AB;m=(G<L)[:,None];o=G[:,None]*S+c[None,:];AO=Q(GO,o,m);W=Q(U,o,m);AP=Q(LAT,o,m);I=A8*A5+AD;A0=I*M;AQ=I*AE+M*AG+A0*AI+Aj;AR=I*AF+M*AH+A0*AJ+Ak;AS=A.where(AQ>0,1.,.1);AT=A.where(AR>0,1.,.1);A1=AQ*AS;A2=AR*AT;AU=A1*AK+A2*AM+Al;AV=A1*AL+A2*AN+Am;N=A7*A.where(AU>0,1.,.1);O=A7*A.where(AV>0,1.,.1);J=(N*An+O*Ap)*AS;K=(N*Aq+O*Ar)*AT;AW=J*AI+K*AJ;Y=J*AE+K*AF+AW*M;As=J*AG+K*AH+AW*I;A.store(DLAT+z,As.to(A3),mask=X);A.store(DV+z,Y.to(A3),mask=X);b+=J*I;d+=K*I;e+=J*M;f+=K*M;g+=J*A0;h+=K*A0;i+=J;j+=K;k+=N*A1;l+=O*A1;n+=N*A2;p+=O*A2;q+=N;r+=O;s+=N*AU;t+=O*AV;u+=A7;v+=Y;x+=Y*I;y+=A.where(X,I,.0)
		b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x,y=A.reduce((b,d,e,f,g,h,i,j,k,l,n,p,q,r,s,t,u,v,x,y),0,Ao);w=WS+AA*20*S+c;C(w,b);C(w+S,d);C(w+2*S,e);C(w+3*S,f);C(w+4*S,g);C(w+5*S,h);C(w+6*S,i);C(w+7*S,j);C(w+8*S,k,T);C(w+9*S,l,V);C(w+10*S,n,T);C(w+11*S,p,V);C(w+12*S,q,T);C(w+13*S,r,V);C(w+14*S,s);C(w+15*S,t);C(w+16*S,u);C(w+17*S,v);C(w+18*S,x);C(w+19*S,y)
	else:
		AX=A.cdiv(17*S,Ah);AY=A.minimum((a+1)*AX,17*S)
		for At in range(a*AX,AY,256):
			A9=At+A.arange(0,256);AZ=A9<AY;Aa=A.zeros([256],F)
			for Z in range(NRS):Aa+=A.load(WS+Z*20*S+A9,mask=AZ,other=.0,cache_modifier='.cg')
			A.store(DP+A9,Aa,mask=AZ)
		Ab=A.zeros([BC],F);Ac=A.zeros([BC],F);Ad=A.zeros([BC],F)
		for Z in range(NRS):Ab+=A.load(WS+(Z*20+17)*S+c,cache_modifier='.cg');Ac+=A.load(WS+(Z*20+18)*S+c,cache_modifier='.cg');Ad+=A.load(WS+(Z*20+19)*S+c,cache_modifier='.cg')
		Au=(Ab*A4)[None,:];Av=(Ac*A4)[None,:];Aw=(Ad*A4)[None,:];Ae=A.arange(0,SB);G=R+Ae;m=(G<L)[:,None];o=G[:,None]*S+c[None,:];W=A.load(U+o,mask=m,other=.0);Af=A.load(DV+o,mask=m,other=.0)
		for A6 in range(R,L,SB):A8=W.to(F);Y=Af.to(F);X,z=m,o;G=A6+SB+Ae;m=(G<L)[:,None];o=G[:,None]*S+c[None,:];W=A.load(U+o,mask=m,other=.0);Af=A.load(DV+o,mask=m,other=.0);I=A8*A5+AD;A.store(DU+z,(A5*(Y-Au-(I-Aw)*Av)).to(A3),mask=X)
Aq={480:(16,16,4,3,64),60:(4,16,2,4,64),256:(16,16,2,4,64)}
def Ar(go:B,u:B,mean:B,rstd:B,lat:B,pb:B)->tuple[B,B,B]:
	F=rstd;E=mean;A=lat;B,S=u.shape[-2:];BC,SUB,N,J,SB=Aq[S];H=S//BC;D=-(-B//max(1,(torch.cuda.get_device_properties(u.device).multi_processor_count*J-(4 if J>1 else 0))//H));D=-(-D//SB)*SB;I=-(-B//D);go,u,E,F,pb=go.contiguous(),u.contiguous(),E.contiguous(),F.contiguous(),pb.contiguous()
	if A.stride(-1)!=1 or A.stride(-2)!=S:A=A.contiguous()
	C=lambda*A,dtype=u.dtype:c(*A,device=u.device,dtype=dtype);K,L,M,O,P=C(u.shape),C(u.shape),C(pb.shape,dtype=G),C(I,20,S,dtype=G),C(u.shape)
	for Q in(1,2):Ap[H*I,](go,u,E,F,A,pb,K,L,M,O,P,B,S,D,I,NCB=H,BC=BC,SUB=SUB,SB=SB,PHASE=Q,num_warps=N)
	return K,L,M
def As():
	if not u(O.d5bottom,'comb_bwd'):A=lambda t,d:t.new_empty(t.shape,dtype=d);e('d5bottom::comb_bwd',Ar,mutates_args=()).register_fake(lambda go,u,mean,rstd,lat,pb:(A(u,u.dtype),A(u,u.dtype),A(pb,G)))
class AI(torch.autograd.Function):
	@staticmethod
	def forward(ctx,u,lat,p):A=p.to(u.dtype);D,B=d(u,dim=-2,unbiased=False,keepdim=True);C=W(D+g);E=s((u-B)*C,lat,A);ctx.save_for_backward(u,B,C,lat,A);return E
	@staticmethod
	def backward(ctx,go):A,B,C,D,E=ctx.saved_tensors;F,G,H=O.d5bottom.comb_bwd(go,A,B,C,D,E);return F,G,H
def a(h,w,act=None):
	A=act
	if A is torch.bfloat16:return AJ.apply(h,w,torch.bfloat16)
	if A is None and w.shape[-1]==60 and h.requires_grad:return AJ.apply(h,w,G)
	B=h.shape
	with torch.autocast('cuda',dtype=A or torch.float16,enabled=A is not None):C=X(h.reshape(B[0],-1,B[-1]),w)
	return(C if A is not None else C.float()).reshape(*B[:-1],w.shape[-1])
class At:
	def __init__(o,n,nq,device):
		B=device;o.n,o.nq,o.d=n,nq,B;A=lambda*A:P(*A,device=B);o.e=[A(M,a,B)for(a,B)in t(K[:-1],K[1:])];o.D=[A(M,B,a)for(a,B)in t(K[:-1],K[1:])];o.b=[A(M,1,B)for B in K[1:]];o.g=A(M,1,K[-1]);o.c=[A(17,M,1,B)for B in K];o.a=o.e+o.D+o.b+[o.g]+o.c
		for p in o.a:p.requires_grad_(True)
		o.x=A(n,K[0]);o.y=A(n,10);o.p=A(n+nq,K[0]);o.P=A(n+nq,10);o.T=A(n,K[0],A8);o.s=A(M,n).long();o.t=P((),dtype=torch.long,device=B);o.iu=P(J,M,T,dtype=torch.long,device=B);C=U(J,device=B,dtype=G)/J;o.l=w*((1-C)/(1-AT)).clamp(max=1.);E=((A6-C)/(A6-A5)).clamp(0,1);o.m=AY*torch.where(C<AZ,.0,torch.where(C<A5,1.,E));o.r=AK(w,device=B);o.h=P(int(AU*nq),dtype=torch.long,device=B);o.B=A(h).to(N);o.f=A(J,len(K)).long();o.u=A(J,M,T);o.v=A(J,M,T,1,A8);o.W=torch.optim.Adam(o.a,lr=o.r,betas=(AS,.999),foreach=True,capturable=True);o.O=torch.compile(o.W.step);o.F=torch.compile(o.V,dynamic=False,options=Ag);o.U=torch.compile(B4,dynamic=False);o.R();D=q();D.wait_stream(Y())
		with torch.cuda.stream(D):
			for F in range(3):o.W.zero_grad(set_to_none=True);o.S()
		Y().wait_stream(D);o.G=torch.cuda.CUDAGraph();o.W.zero_grad(set_to_none=True)
		with torch.cuda.graph(o.G):
			for F in range(A2):
				for p in o.a:p.grad=None
				o.S()
	def L(o,xl,yl,xu,tu,tg,ofs,lb):
		F=ofs;F=list(F.unbind());K=lambda t:o.B[(F.pop()+U(t.numel(),device=t.device))%h].view(t.shape).to(t.dtype);P,Q=[A.to(N)for A in o.b],o.g.to(N);R=[p.to(N)for p in o.c];xl=xl+Ab*(tg*o.v.index_select(0,o.t.view(1)).squeeze(0)).sum(-1);A=torch.stack((xl,xu),1);A=A+A3*K(A);E=[A[:,1]];A=A.to(N);G=len(o.e)-1
		for(B,w)in b(o.e):C=Aw(a(A,w,N),2)if B<G else AH(a(A,w,N));C=C+(A3*(1-o.t.float()/J).clamp_min(.2).pow(.5)*K(C).float()).to(C.dtype);E.append(C[:,1]);L=P[B][:,None];A=(C+L)*Q[:,None]if B==G else I.relu(C+L)
		H=A.float();M=-(yl*I.log_softmax(H[:,0],-1)).sum(-1);S=(o.t<J*.5)|lb[...,0];T=-((tu*I.log_softmax(H[:,1],-1)).sum(-1)*S).mean(-1).sum();O=I.softmax(H[:,1],-1);D=An((O+(lb*(tu-O)).detach()).to(N),E[-1],R[-1])
		for B in range(G,0,-1):D=AI.apply(a(D,o.D[B],N),E[B],o.c[B])
		D=AI.apply(a(D.float(),o.D[0]),E[0],o.c[0]);V=(D.float()-xu).square().mean((1,2)).sum();return M.mean(-1).sum()+Aa*T+AW*V,M.detach()
	def V(o):A=o.t.view(1);m=o.m.index_select(0,A);c=O.d5e.c(o.s,m);B=torch.searchsorted(c,o.u.index_select(0,A).squeeze(0)*c[:,-1:]).clamp_max(o.n-1);C=o.iu.index_select(0,A).squeeze(0);o.r.copy_(o.l.index_select(0,A).squeeze(0));D=o.T[B];E,F=o.L(o.x[B],o.y[B],o.p[C],o.P[C],D,o.f.index_select(0,A).squeeze(0),(C<o.n)[...,None]);O.d5e.u(o.s,B,F,o.t);o.t.add_(1);return E
	def S(o):o.F().backward();o.O()
	@torch.no_grad()
	def R(o):
		A=torch.Generator(device=o.d);A.manual_seed(A4)
		for w in o.e+o.D:w.copy_(n(w.shape,generator=A,device=o.d)*w.shape[1]**-.5)
		B=[0,1,2,3,4,5,8,9,10,11,14,15]
		for p in o.c:p.zero_();p[B]=n(len(B),*p.shape[1:],generator=A,device=o.d)*AX
		for p in o.b:p.zero_()
		o.g.fill_(1.)
		for p in o.a:
			C=o.W.state.get(p)
			if C:C['exp_avg'].zero_();C['exp_avg_sq'].zero_();C['step'].zero_()
		o.t.zero_();o.s.fill_(1065353216);F=U(o.n+o.nq,device=o.d);G=[]
		for(E,B)in((J//2,F),(J-J//2,D([F,o.h]))):H=torch.rand(v.ceil(E*T/len(B)),len(B),generator=A,device=o.d).argsort(-1);G.append(B[H.view(-1)[:E*T]].view(E,1,T))
		o.iu.copy_(D(G));o.B.copy_(n(h,generator=A,device=o.d));o.f.copy_(torch.randint(h,o.f.shape,generator=A,device=o.d));o.u.uniform_(generator=A);o.v.normal_(generator=A);torch.cuda.manual_seed(A4+1)
	def A(o,i,t,b,gm):return(t+b)*gm if i==len(o.e)-1 else I.relu(t+b)
	@torch.no_grad()
	def Q(o,q):
		A=D((o.x,q))[None];C=[]
		for(B,w)in b(o.e):E=a(A,w);G,H=d(E[:,:o.n],dim=-2,unbiased=False,keepdim=True);F=(E-H)*W(G+g);C.append((F+o.b[B])[0]);A=o.A(B,F,o.b[B],o.g)
		return C+[A[0]]
Au,Av=60,40
def guard(s,p,e):B,A=s.topk(2,1);A=A[:,0];C=B[:,0]-B[:,1]>=AQ;c=P(100,device=s.device).index_add_(0,p*10+A,C.float());c[::11]=0;return torch.where(C&((c.max()>Au)|(e>Av)),A,p)
def ladder_kernel(train_x,train_y,test_x):
	Q=test_x;M=train_x;E=train_y;os.environ.setdefault('TRITON_CACHE_DIR','/tmp/triton-cache');os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/inductor-cache');As();encoded_register();B3();m.allow_tf32=False;G,R=Aj(M,E,Q);m.allow_tf32=True
	with torch.no_grad():B,C=k(M.float())@R,k(Q.float())@R;c=(M.float()*AV).std()/B.std();B,C=B*c,C*c
	N='ladder',len(B),len(C)
	if N not in CACHE:CACHE[N]=At(B.shape[0],C.shape[0],B.device)
	A=CACHE[N]
	with torch.no_grad():A.x.copy_(B);A.T.copy_(Am(B,E));A.y.copy_(I.one_hot(E.long(),10).float());A.p.copy_(D((B,C)));A.P.copy_(D((A.y,I.softmax(G/A7,1))));S=G.topk(2,1).values;A.h.copy_((S[:,0]-S[:,1]).topk(len(A.h),largest=False).indices+len(B))
	A.R()
	for T in range(J//A2):A.G.replay()
	with torch.no_grad():H=A.Q(C);U,V=[A[:len(B)]for A in H],[A[len(B):]for A in H];d=(U[-1].argmax(1)!=E).sum();e=D([B,C]);K=D([U[2],V[2]]).float();K=K*e.norm()/K.norm();m.allow_tf32=False;p=(1-.2)*I.softmax(V[-1],-1)+.2*I.softmax(G/A7,1);W=[A.p]+[C-A.b[B][0]for(B,C)in b(H[:-1])];X=[w[0]for w in A.D];Y=[A[:,0]for A in A.c];T,f=A.U(I.softmax(H[-1][:len(B)],-1),[A[:len(B)]for A in W],X,Y);F=(-p.topk(2).values.diff()[:,0]).topk(1024,largest=False).indices;Z=p[F].topk(3).indices;g=[A[len(B):][F,None].expand(-1,3,-1).reshape(-1,A.shape[-1])for A in W];L,T=A.U(I.one_hot(Z.flatten(),10).float(),g,X,Y,f);L=L.reshape(1024,3,60);a=C[F,None];O=(L-a).square().mean(-1);P=(L*a).sum(-1);h=(O[:,1]-O[:,0]).abs().median().clamp_min(1e-09);i=(P[:,1]-P[:,0]).abs().median().clamp_min(1e-09);j=(-.5*O/h+.5*P/i).softmax(-1);p[F]*=1-.5;p[F[:,None],Z]+=.5*j
	return guard(G,Al(p,K,E.long()).argmax(1),d)
def Aw(u,groups):B=groups;a=u.reshape(*u.shape[:-2],B,u.shape[-2]//B,u.shape[-1]);f=a.float();A=f.mean(-2,keepdim=True);C=(f.square().mean(-2,keepdim=True)-A.square()).clamp_min(0).to(a.dtype);A=A.to(a.dtype);return((a-A)*W(C+g)).reshape_as(u)
@triton.jit
def Ax(S,P,T,N:E,K:E,B:E=512):C,D=A.program_id(0)//K,A.program_id(0)%K;E=A.arange(0,B);j=D*B+E;G=A.load(S+C*N+j,j<N,0);H=(G&4294967295).to(A.uint32).to(F,bitcast=True);p=A.cumsum(H,0);A.store(P+C*N+j,p,j<N);I=A.sum(A.where(E==A.minimum(B,N-D*B)-1,p,.0),0);A.store(T+C*K+D,I)
def Ay(state:B,mix:B)->B:B=state;m,C=B.shape;A=triton.cdiv(C,512);p=R(B,dtype=G);D=c((m,A),device=B.device);E=R(B,dtype=G);Ax[m*A,](B,p,D,C,A,num_warps=4);B0[m*A,](p,D,mix,E,C,A,triton.next_power_of_2(A),num_warps=4);return E
@triton.jit
def Az(S,I,CE,T,N:E,BATCH:E,L:E,BLOCK:E=256):B=A.program_id(0)*BLOCK+A.arange(0,BLOCK);C=B<L;D=A.load(I+B,C,0);E=B//BATCH;G=A.load(CE+B,C,.0).to(F);H=G.to(A.uint32,bitcast=True).to(A.int64);J=A.load(T).to(A.int64)*BATCH+B%BATCH+1;K=J<<32|H;A.atomic_max(S+E*N+D,K,C,sem='relaxed')
def A_(state:B,indices:B,ce:B,counter:B)->None:C=indices;B=state;m,D=B.shape;A=C.shape[1];Az[triton.cdiv(m*A,256),](B,C,ce,counter,D,A,m*A)
@triton.jit
def B0(P,T,MIX,C,N:E,K:E,R:E,B:E=512):E,G=A.program_id(0)//K,A.program_id(0)%K;F=A.arange(0,R);H=A.load(T+E*K+F,F<K,.0);J=A.sum(H,0);L=A.sum(A.where(F<G,H,.0),0);D=G*B+A.arange(0,B);p=A.load(P+E*N+D,D<N,.0);I=A.load(MIX);c=I*(p+L)/J+(1-I)*(D+1)/N;A.store(C+E*N+D,c,D<N)
def encoded_register():
	if not u(O.d5e,'c'):e('d5e::c',Ay,mutates_args=()).register_fake(lambda state,mix:R(state,dtype=G));e('d5e::u',A_,mutates_args={'state'}).register_fake(lambda state,indices,ce,counter:None)
r={}
def B1(device):
	B=device;A=B.index
	if A not in r:r[A]=q(device=B)
	return r[A]
def B2(go:B,h:B,w:B,need_h:bool)->tuple[B,B]:
	A=go;A=A.contiguous();h=h.contiguous();w=w.contiguous();C=Y(h.device);B=B1(h.device);B.wait_stream(C)
	with torch.cuda.stream(B):D=X(h.transpose(1,2),A);h.record_stream(B);A.record_stream(B)
	E=X(A,w.transpose(1,2))if need_h else h.new_empty((0,));C.wait_stream(B);D.record_stream(C);return E,D
def _(go,h,w,need_h):return R(h)if need_h else h.new_empty((0,)),w.new_empty(w.shape)
def B3():
	if not u(O.d5pc,'backward'):e('d5pc::backward',B2,mutates_args=()).register_fake(_)
class AJ(torch.autograd.Function):
	@staticmethod
	def forward(ctx,h,w,act):A=h.shape;B=h.reshape(A[0],-1,A[-1]).to(act);C=w.to(act);ctx.save_for_backward(B,C);ctx.shape=A;return X(B,C).reshape(*A[:-1],w.shape[-1])
	@staticmethod
	def backward(ctx,go):A=ctx;B,w=A.saved_tensors;C,D=O.d5pc.backward(go.reshape(B.shape[0],-1,w.shape[-1]),B,w,A.needs_input_grad[0]);return C.reshape(A.shape)if A.needs_input_grad[0]else None,D.float(),None
def B4(r,latent,weights,params,stats=None):
	D=stats;E=[]
	for(j,A)in b(range(4,-1,-1)):
		if A<4:r=r@weights[A]
		if D is None:F,B=d(r,dim=0,unbiased=False,keepdim=True);C=W(F+1e-10)
		else:B,C=D[j]
		E.append((B,C));r=s((r-B)*C,latent[A],params[A])
	return r,E