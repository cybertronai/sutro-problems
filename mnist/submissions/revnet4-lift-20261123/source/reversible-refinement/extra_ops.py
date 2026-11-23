"""Small ordered FP32 primitives for explicit residual/coupling networks.

Each multiply and add is a separate operation (enable_fp_fusion=False); no
autograd, tensor cores, atomics, or reductions are used by these primitives.
"""
import triton
import triton.language as tl


@triton.jit
def _axpby(X, Y, O, SIZE: tl.constexpr, A: tl.constexpr, B: tl.constexpr,
           BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+i, i<SIZE, 0)
    y = tl.load(Y+i, i<SIZE, 0)
    a = x*A
    b = y*B
    tl.store(O+i, a+b, i<SIZE)


@triton.jit
def _lift(X, O, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = i//(C*S)
    channel = (i//S)%C
    pixel = i%S
    value = tl.load(X+batch*S+pixel, (i<N*C*S)&(channel==0), 0)
    tl.store(O+i, value, i<N*C*S)


@triton.jit
def _slice(X, O, N: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
           START: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src = (i//(CO*S))*CI*S + START*S + i%(CO*S)
    value = tl.load(X+src, i<N*CO*S, 0)
    tl.store(O+i, value, i<N*CO*S)


@triton.jit
def _scatter(X, O, N: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
             START: tl.constexpr, S: tl.constexpr, ADD: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = (i//S)%CO
    valid_channel = (channel>=START)&(channel<START+CI)
    src = (i//(CO*S))*CI*S+(channel-START)*S+i%S
    value = tl.load(X+src, (i<N*CO*S)&valid_channel, 0)
    if ADD:
        value = tl.load(O+i, i<N*CO*S, 0)+value
    tl.store(O+i, value, i<N*CO*S)


@triton.jit
def _concat(X, Y, O, N: tl.constexpr, CA: tl.constexpr, CB: tl.constexpr,
            S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = (i//S)%(CA+CB)
    batch = i//((CA+CB)*S)
    va = tl.load(X+batch*CA*S+channel*S+i%S,
                 (i<N*(CA+CB)*S)&(channel<CA), 0)
    vb = tl.load(Y+batch*CB*S+(channel-CA)*S+i%S,
                 (i<N*(CA+CB)*S)&(channel>=CA), 0)
    tl.store(O+i, va+vb, i<N*(CA+CB)*S)


class ExtraOps:
    def __init__(self, backend):
        self.backend = backend

    def _save(self, name, kernel):
        self.backend.save('extra_'+name, kernel)

    def axpby(self, x, y, out, a=1., b=1.):
        assert x.shape == y.shape == out.shape
        k = _axpby[(triton.cdiv(out.numel(),256),)](
            x,y,out,out.numel(),float(a),float(b),256,enable_fp_fusion=False)
        self._save('axpby',k)

    def lift(self, x, out):
        n,c,h,w=out.shape
        k=_lift[(triton.cdiv(out.numel(),256),)](
            x,out,n,c,h*w,256,enable_fp_fusion=False)
        self._save('lift',k)

    def channel_slice(self, x, out, start):
        n,ci,h,w=x.shape; co=out.shape[1]
        k=_slice[(triton.cdiv(out.numel(),256),)](
            x,out,n,ci,co,start,h*w,256,enable_fp_fusion=False)
        self._save('slice',k)

    def scatter(self, x, out, start, add=False):
        n,ci,h,w=x.shape; co=out.shape[1]
        k=_scatter[(triton.cdiv(out.numel(),256),)](
            x,out,n,ci,co,start,h*w,add,256,enable_fp_fusion=False)
        self._save('scatter',k)

    def concat(self, x, y, out):
        n,ca,h,w=x.shape; cb=y.shape[1]
        k=_concat[(triton.cdiv(out.numel(),256),)](
            x,y,out,n,ca,cb,h*w,256,enable_fp_fusion=False)
        self._save('concat',k)
