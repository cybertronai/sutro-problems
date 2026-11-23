"""Ordered FP32 CNN primitives. No tensor cores, atomics, or fused multiply-add.

NCHW tensors; convolution weight layout Cout,Cin,3,3. Forward uses input
channel, kernel row, kernel column order. Input-gradient uses output channel,
flipped-kernel row, flipped-kernel column order. Weight-gradient uses sample,
output row, output column order. All starting accumulators are positive zero.
"""
import triton
import triton.language as tl
import hashlib

KW = {'num_warps':4, 'enable_fp_fusion':False}

@triton.jit
def _pad(X,P,N:tl.constexpr,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    col=i%(W+2)-1; row=i//(W+2)%(H+2)-1; nc=i//((H+2)*(W+2))
    valid=(i<N*C*(H+2)*(W+2))&(row>=0)&(row<H)&(col>=0)&(col<W)
    value=tl.load(X+nc*H*W+row*W+col,valid,other=0.)
    tl.store(P+i,value,i<N*C*(H+2)*(W+2))

@triton.jit
def _forward(XP,K,O,N:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=i%W; y=i//W%H; co=i//(H*W)%CO; n=i//(CO*H*W)
    valid=i<N*CO*H*W
    acc=tl.full((BLOCK,),0.,tl.float32)
    for c in range(CI):
        for ky in tl.static_range(3):
            for kx in tl.static_range(3):
                a=tl.load(XP+((n*CI+c)*(H+2)+y+ky)*(W+2)+x+kx,valid,other=0.)
                b=tl.load(K+((co*CI+c)*3+ky)*3+kx,valid,other=0.)
                product=a*b
                acc=acc+product
    tl.store(O+i,acc,valid)

@triton.jit
def _dinput(DP,K,DX,N:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=i%W; y=i//W%H; ci=i//(H*W)%CI; n=i//(CI*H*W)
    valid=i<N*CI*H*W
    acc=tl.full((BLOCK,),0.,tl.float32)
    for c in range(CO):
        for ky in tl.static_range(3):
            for kx in tl.static_range(3):
                a=tl.load(DP+((n*CO+c)*(H+2)+y+ky)*(W+2)+x+kx,valid,other=0.)
                b=tl.load(K+((c*CI+ci)*3+2-ky)*3+2-kx,valid,other=0.)
                product=a*b
                acc=acc+product
    tl.store(DX+i,acc,valid)

@triton.jit
def _dweight(XP,DY,DK,N:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    kx=i%3; ky=i//3%3; ci=i//9%CI; co=i//(9*CI)
    valid=i<CO*CI*9
    acc=tl.full((BLOCK,),0.,tl.float32)
    for pos in range(N*H*W):
        x=pos%W; y=pos//W%H; n=pos//(H*W)
        a=tl.load(XP+((n*CI+ci)*(H+2)+y+ky)*(W+2)+x+kx,valid,other=0.)
        b=tl.load(DY+((n*CO+co)*H+y)*W+x,valid,other=0.)
        product=a*b
        acc=acc+product
    tl.store(DK+i,acc,valid)

@triton.jit
def _mm(A,B,O,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,AT:tl.constexpr,BT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//N; col=i%N; valid=i<M*N
    acc=tl.full((BLOCK,),0.,tl.float32)
    for k in range(K):
        aa=k*M+row if AT else row*K+k
        bb=col*K+k if BT else k*N+col
        a=tl.load(A+aa,valid,other=0.)
        b=tl.load(B+bb,valid,other=0.)
        product=a*b
        acc=acc+product
    tl.store(O+i,acc,valid)

@triton.jit
def _rows(A,O,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    col=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0.,tl.float32)
    for row in range(M):
        a=tl.load(A+row*N+col,col<N,other=0.)
        acc=acc+a
    tl.store(O+col,acc,col<N)

@triton.jit
def _relu(X,O,DY,DX,SIZE:tl.constexpr,BACKWARD:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0.)
    active=x>0.
    if BACKWARD:
        d=tl.load(DY+i,i<SIZE,other=0.)
        tl.store(DX+i,tl.where(active,d,0.),i<SIZE)
    else:
        tl.store(O+i,tl.where(active,x,0.),i<SIZE)

@triton.jit
def _momentum(W,G,V,SIZE:tl.constexpr,LR:tl.constexpr,MU:tl.constexpr,DECAY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    w=tl.load(W+i,i<SIZE,other=0.)
    g=tl.load(G+i,i<SIZE,other=0.)
    v=tl.load(V+i,i<SIZE,other=0.)
    penalty=DECAY*w
    d=g+penalty
    carried=MU*v
    next_v=carried+d
    update=LR*next_v
    next_w=w-update
    tl.store(V+i,next_v,i<SIZE)
    tl.store(W+i,next_w,i<SIZE)

@triton.jit
def _normalize(X,O,SIZE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0.)
    y=x*4.
    y=y-0.5
    tl.store(O+i,y,i<SIZE)

@triton.jit
def _bias(X,B,O,SIZE:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0.)
    b=tl.load(B+i%C,i<SIZE,other=0.)
    tl.store(O+i,x+b,i<SIZE)

@triton.jit
def _delta(X,Y,D,SIZE:tl.constexpr,INV_BATCH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0.)
    y=tl.load(Y+i//10,i<SIZE,other=0)
    target=(i%10==y).to(tl.float32)
    error=x-target
    tl.store(D+i,error*INV_BATCH,i<SIZE)

class Backend:
    def __init__(self): self.compiled={}
    def save(self,name,k):
        key=name+'__'+hashlib.sha256(k.asm['ptx'].encode()).hexdigest()[:16]
        self.compiled[key]=k
    def pad(self,x,out):
        n,c,h,w=x.shape
        self.save('pad',_pad[(triton.cdiv(out.numel(),128),)](x,out,n,c,h,w,128,**KW))
    def forward(self,xp,k,out):
        n,co,h,w=out.shape;ci=k.shape[1]
        self.save('conv_forward',_forward[(triton.cdiv(out.numel(),128),)](xp,k,out,n,ci,co,h,w,128,**KW))
    def dinput(self,dp,k,dx):
        n,ci,h,w=dx.shape;co=k.shape[0]
        self.save('conv_dinput',_dinput[(triton.cdiv(dx.numel(),128),)](dp,k,dx,n,ci,co,h,w,128,**KW))
    def dweight(self,xp,dy,dk):
        n,co,h,w=dy.shape;ci=dk.shape[1]
        self.save('conv_dweight',_dweight[(triton.cdiv(dk.numel(),128),)](xp,dy,dk,n,ci,co,h,w,128,**KW))
    def mm(self,a,b,out,at=False,bt=False):
        m,n=out.shape;k=a.shape[0] if at else a.shape[1]
        self.save('mm',_mm[(triton.cdiv(out.numel(),128),)](a,b,out,m,k,n,at,bt,128,**KW))
    def rows(self,a,out):
        m,n=a.shape
        self.save('rows',_rows[(triton.cdiv(n,128),)](a,out,m,n,128,**KW))
    def relu(self,x,out):
        self.save('relu',_relu[(triton.cdiv(x.numel(),128),)](x,out,x,out,x.numel(),False,128,**KW))
    def drelu(self,x,dy,dx):
        self.save('drelu',_relu[(triton.cdiv(x.numel(),128),)](x,dx,dy,dx,x.numel(),True,128,**KW))
    def momentum(self,w,g,v,lr,mu=.9,decay=.0001):
        self.save('momentum',_momentum[(triton.cdiv(w.numel(),128),)](w,g,v,w.numel(),lr,mu,decay,128,**KW))
    def normalize(self,x,out):
        self.save('normalize',_normalize[(triton.cdiv(x.numel(),128),)](x,out,x.numel(),128,**KW))
    def bias(self,x,b,out):
        self.save('bias',_bias[(triton.cdiv(x.numel(),128),)](x,b,out,x.numel(),len(b),128,**KW))
    def delta(self,x,y,out,inv_batch):
        self.save('delta',_delta[(triton.cdiv(x.numel(),128),)](x,y,out,x.numel(),inv_batch,128,**KW))

@triton.jit
def _augment(RAW,LABELS,INDICES,COEFFICIENTS,ORDER,OFFSET,X,Y,B:tl.constexpr,PIXELS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    base=tl.load(OFFSET)
    valid=i<B*PIXELS
    position=(base*PIXELS+i)*4
    ix=tl.load(INDICES+position,valid,other=0)
    cf=tl.load(COEFFICIENTS+position,valid,other=0.)
    source=tl.load(RAW+ix,valid,other=0.)
    acc=source*cf
    for neighbor in tl.static_range(1,4):
        ix=tl.load(INDICES+position+neighbor,valid,other=0)
        cf=tl.load(COEFFICIENTS+position+neighbor,valid,other=0.)
        source=tl.load(RAW+ix,valid,other=0.)
        term=source*cf
        acc=acc+term
    normalized=acc*4.
    normalized=normalized-0.5
    tl.store(X+i,normalized,valid)
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    source_row=tl.load(ORDER+base+row,row<B,other=0)
    label=tl.load(LABELS+source_row,row<B,other=0)
    tl.store(Y+row,label,row<B)

@triton.jit
def _advance(OFFSET,COUNT:tl.constexpr):
    old=tl.load(OFFSET)
    tl.store(OFFSET,old+COUNT)

@triton.jit
def _softmax_delta(X,Y,D,B:tl.constexpr,INV_BATCH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=row<B
    largest=tl.load(X+row*10,valid,other=0.)
    for c in tl.static_range(1,10):
        value=tl.load(X+row*10+c,valid,other=0.)
        largest=tl.where(value>largest,value,largest)
    total=tl.full((BLOCK,),0.,tl.float32)
    for c in tl.static_range(10):
        value=tl.load(X+row*10+c,valid,other=0.)
        shifted=value-largest
        clamped=tl.minimum(tl.maximum(shifted,-16.),0.)
        scaled=clamped*0.0009765625
        term=1.+scaled
        for power in tl.static_range(10):term=term*term
        total=total+term
        tl.store(D+row*10+c,term,valid)
    truth=tl.load(Y+row,valid,other=0)
    for c in tl.static_range(10):
        term=tl.load(D+row*10+c,valid,other=0.)
        # Triton 3.1 tl.div_rn emits div.rn.ftz.f32. Pin the declared IEEE
        # operation explicitly; no approximate reciprocal or subnormal flush.
        probability=tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',
            constraints='=f,f,f',args=[term,total],dtype=tl.float32,is_pure=True,pack=1)
        target=(truth==c).to(tl.float32)
        error=probability-target
        tl.store(D+row*10+c,error*INV_BATCH,valid)

def _augment_method(self,raw,labels,indices,coefficients,order,offset,x,y):
    pixels=x.shape[-1]*x.shape[-2];b=len(x)
    self.save('augment',_augment[(triton.cdiv(x.numel(),128),)](raw,labels,indices,coefficients,order,offset,x,y,b,pixels,128,**KW))
def _advance_method(self,offset,count):
    self.save('advance',_advance[(1,)](offset,count,**KW))
def _softmax_method(self,x,y,out,inv_batch):
    self.save('softmax_delta',_softmax_delta[(triton.cdiv(len(x),128),)](x,y,out,len(x),inv_batch,128,**KW))
Backend.augment=_augment_method
Backend.advance=_advance_method
Backend.softmax_delta=_softmax_method

@triton.jit
def _argmax(X,O,B:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    best=tl.load(X+row*10,row<B,other=0.)
    index=tl.full((BLOCK,),0,tl.int32)
    for c in tl.static_range(1,10):
        value=tl.load(X+row*10+c,row<B,other=0.)
        greater=value>best
        index=tl.where(greater,c,index)
        best=tl.where(greater,value,best)
    tl.store(O+row,index,row<B)

def _argmax_method(self,x,out):
    self.save('argmax',_argmax[(triton.cdiv(len(x),128),)](x,out,len(x),128,**KW))
Backend.argmax=_argmax_method

@triton.jit
def _head_mask(MASKS,OFFSET,CACHE,SIZE:tl.constexpr,HEAD:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    base=tl.load(OFFSET)
    value=tl.load(MASKS+base*HEAD+i,i<SIZE,other=0.)
    tl.store(CACHE+i,value,i<SIZE)

@triton.jit
def _multiply(X,Y,O,SIZE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0.);y=tl.load(Y+i,i<SIZE,other=0.)
    tl.store(O+i,x*y,i<SIZE)

def _head_mask_method(self,masks,offset,cache):
    self.save('head_mask',_head_mask[(triton.cdiv(cache.numel(),128),)](masks,offset,cache,cache.numel(),cache.shape[1],128,**KW))
def _multiply_method(self,x,y,out):
    self.save('multiply',_multiply[(triton.cdiv(x.numel(),128),)](x,y,out,x.numel(),128,**KW))
Backend.head_mask=_head_mask_method
Backend.multiply=_multiply_method
