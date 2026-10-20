"""D3 Fisher-PCG plus local affine-hull correction.
Fisher-PCG D2 by @spartanhaden, using @sigkillme0 and @islamborghini methods;
HKNN correction from the published hybrid-tangent D4 by @spartanhaden and contributors.
Retains the parent metric and FITC solver with 6144 hard rows, 8 PCG steps.
Corrects 384 lowest-margin queries with 70-neighbor class hulls (2048 prescreen).
Native FP32 hull-CG48 replaces the Cholesky solve. All fitting is per call; no test labels.
"""
import gc
import math
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
F32, F64, F16 = (torch.float32, torch.float64, torch.float16)
P = {'NPOW': 0.6, 'LDA_S': 3.0, 'SHR': 0.7, 'QT': 0.15, 'QROWS': 2048, 'UW0': 0.8, 'UW1': 0.196, 'NS_ITERS': 4, 'BLEND': 0.02, 'KPC': 24, 'LLOYD': 1, 'NHARD': 6144, 'GAMMA': 0.02, 'RIDGE': 0.1, 'JIT': 0.0001, 'STEPS': 8, 'VSC': 256.0}
WARM = 64
_GRAPHS = {}
_ST = {}
_C = {}

def const(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]

def eye(n, s=1.0, dt=F32):
    return const(('eye', n, s, dt), lambda: torch.eye(n, device='cuda', dtype=dt) * s)

def _stream(k):
    if k not in _ST:
        _ST[k] = torch.cuda.Stream(priority=-1)
    return _ST[k]

def chol(a):
    return torch.linalg.cholesky_ex(a)[0]

def sqrtm(a, iters):
    n = len(a)
    c = a.norm()
    x = torch.cat([torch.cat([const(('z', n), lambda: torch.zeros(n, n, device='cuda')), a / c], 1), const(('iz', n), lambda: torch.cat([torch.eye(n, device='cuda'), torch.zeros(n, n, device='cuda')], 1))], 0)
    i15 = eye(2 * n, 1.5)
    for _ in range(iters):
        x = x @ torch.addmm(i15, x, x, alpha=-0.5)
    return x[:n, n:] * c.sqrt()

def pow34(m, P):
    d = len(m)
    sq = sqrtm(m, P['NS_ITERS'])
    m = sq @ sqrtm(sq, P['NS_ITERS'])
    m = m + m.T
    return torch.addcmul(eye(d, P['BLEND']), m, (1 - P['BLEND']) * d / m.trace())

def lda(raw, y1h, cnt, s):
    n, d = raw.shape
    A = (y1h.T @ raw).mul_((cnt * n).rsqrt()[:, None]).T
    C = 1 - 1 / n + 0.001
    H = torch.addmm(eye(10), A.T, A, alpha=-1 / C)
    B = torch.addmm(eye(d, 1 / C), A, torch.cholesky_solve(A.T, chol(H)), alpha=s / C ** 2)
    return B.mul_(d / B.trace())

def qda_fisher(x, y1h, cnt, Bm, P):
    n, d = x.shape
    Nm = torch.cholesky_inverse(chol(torch.add(eye(d, 0.5), Bm, alpha=0.5)))
    xa = torch.cat([x, const(('ones', n), lambda: torch.ones(n, 1, device='cuda'))], 1)
    S = (xa.T @ (y1h[:, :, None] * x[:, None, :]).view(n, 10 * d)).view(d + 1, 10, d)
    rc = 1.0 / cnt
    mu = S[d] * rc[:, None]
    sig = torch.baddbmm(S[:d].transpose(0, 1) * rc[:, None, None], mu[:, :, None], mu[:, None, :], beta=P['SHR'], alpha=-P['SHR'])
    L = chol(sig.add_(Nm, alpha=1 - P['SHR']))
    Pc = torch.cholesky_inverse(L)
    R = P['QROWS']
    Dl = x[:R][None] - mu[:, None]
    J = torch.bmm(Dl, Pc)
    lg = (Dl * J).sum(2).add_(L.diagonal(dim1=1, dim2=2).log().sum(1)[:, None], alpha=2.0).mul_(-0.5 * P['QT'])
    p = torch.softmax(lg.T, 1)
    G = J.transpose(0, 1)
    K = torch.bmm(p[:, None, :], G).squeeze(1)
    return torch.addmm((J * p.T[:, :, None]).view(-1, d).T @ J.view(-1, d), K.T, K, alpha=-1)

def features(xx, n, T):
    A = xx @ T
    Q = A.norm(dim=1, keepdim=True)
    Pm = Q[:n].square().mean()
    lq = (Q.clamp_min(1e-06) / Pm.sqrt()).log_()
    s = (1 + 0.25 * lq[:n].square().mean()).rsqrt()
    out = torch.empty(len(A) + 1, A.shape[1] + 1, device=A.device)
    out[-1] = 0
    torch.mul(A, Pm.sqrt() * s / Q.clamp_min(1e-06), out=out[:-1, :-1])
    torch.mul(lq, 0.5 * Pm.sqrt() * s, out=out[:-1, -1:])
    return out

def class_slots(lab, valid, cap, ncls=10):
    n = len(lab)
    oh = F.one_hot(lab, ncls).T.float()
    if valid is not None:
        oh = oh * valid
    rank = (oh.cumsum(1) * oh).sum(0)
    slot = torch.where((rank > 0) & (rank <= cap), lab * cap + rank.long() - 1, ncls * cap)
    tbl = torch.full((ncls * cap + 1,), n, dtype=torch.long, device=lab.device)
    tbl.scatter_(0, slot, const(('ar', n), lambda: torch.arange(n, device='cuda')))
    return tbl[:ncls * cap]

def compress(z, zhl, lab, P, hook):
    n, d = z.shape
    kpc = P['KPC']
    m = 10 * kpc
    C = z[class_slots(lab, None, kpc)]
    zh = zhl[:, :d]
    own = lab[:, None, None].expand(n, 1, kpc)
    ar = const(('ar', m), lambda: torch.arange(m, device='cuda'))
    base = lab * kpc

    def scores(C):
        return torch.addmm(C.square().sum(1), zh, C.half().T, beta=-0.5, out_dtype=F32).view(n, 10, kpc)
    hist = [C]
    for _ in range(P['LLOYD']):
        a = scores(C).gather(1, own).squeeze(1).argmax(1).add_(base)
        s = torch.mm((a[:, None] == ar).half().T, zhl, out_dtype=F32)
        cnt = s[:, 2 * d:]
        C = torch.where(cnt > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1), C)
        hist.append(C)
        if len(hist) == 2:
            hook(C)
    S = scores(C)
    jo, ja = S.gather(1, own).squeeze(1).max(1)
    oth = S.masked_fill(F.one_hot(lab, 10).bool()[:, :, None], float('-inf')).amax((1, 2))
    marg = (jo - oth) / (z.square().sum(1) - jo - oth).clamp_min(1e-06)
    hard = marg.argsort()[:P['NHARD']]
    oh = (ja.add_(base)[:, None] == ar).half().index_fill_(0, hard, 0.0)
    s = torch.mm(oh.T, zhl, out_dtype=F32)
    cnt = s[:, 2 * d]
    Q = torch.where(cnt[:, None] > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1)[:, None], C)
    pts = torch.cat([z[hard], Q])
    plab = torch.cat([lab[hard], const(('cl', kpc), lambda: torch.arange(10, device='cuda').repeat_interleave(kpc))])
    w = torch.cat([const(('one', P['NHARD']), lambda: torch.ones(P['NHARD'], device='cuda')), cnt])
    return (pts, plab, w, hist)

def rbf32(a, b, g, bn=None):
    an = a.square().sum(1, keepdim=True)
    bn = an.T if bn is None else bn
    return torch.addmm(bn, a, b.T, beta=-g, alpha=2 * g).sub_(an, alpha=g).exp_()

def kaug(a, g, la, left):
    al = torch.addcmul(la, a.square().sum(1), const(('mg', g), lambda: torch.tensor(-g, device='cuda')))
    ah = al.half()
    n, d = a.shape
    out = torch.zeros(n, -(-(d + 4) // 8) * 8, device=a.device, dtype=F16)
    if left:
        out[:, :d] = a
        out[:, d:d + 2] = 1
        out[:, d + 2] = ah
        torch.sub(al, ah.float(), out=out[:, d + 3])
    else:
        torch.mul(a, 2 * g, out=out[:, :d])
        out[:, d] = ah
        torch.sub(al, ah.float(), out=out[:, d + 1])
        out[:, d + 2:d + 4] = 1
    return out

def flash_scores(zq, zk, val, g, vsc):
    nq, dim = zq.shape
    nk = len(zk)
    hd = -(-(dim + 2) // 8) * 8
    m = val.shape[1]
    q = torch.zeros(nq, hd, device=zq.device, dtype=F16)
    torch.mul(zq, math.sqrt(2 * g), out=q[:, :dim])
    q[:, dim:dim + 2] = 1
    qn = q[:, :dim].float().square().sum(1).mul_(0.5)
    k = torch.zeros(nk, hd, device=zq.device, dtype=F16)
    torch.mul(zk, math.sqrt(2 * g), out=k[:, :dim])
    kn = k[:, :dim].float().square().sum(1)
    torch.mul(kn, -0.5, out=k[:, dim])
    torch.addcmul(k[:, dim].float(), kn, const(('h', 0), lambda: torch.tensor(0.5, device='cuda')), out=k[:, dim + 1]).neg_()
    sv = val.abs().amax(0, keepdim=True).clamp_min_(1e-30)
    v = torch.zeros(nk, hd, device=zq.device, dtype=F16)
    torch.mul(val, vsc / sv, out=v[:, :m])
    o, lse = torch.ops.aten._scaled_dot_product_flash_attention(q[None, None], k[None, None], v[None, None], scale=1.0)[:2]
    return o[0, 0, :, :m] * (lse[0, 0] - qn).exp_().mul_(1 / vsc)[:, None] * sv

def fitc_a(lm, g, jit):
    Kmm = rbf32(lm, lm, g)
    Kmm.diagonal().add_(jit)
    L = chol(Kmm)
    return torch.linalg.solve_triangular(L, eye(len(lm)), upper=False)

def fitc_k(pts, w, sw, lm, g, ridge, jit):
    Kc = rbf32(pts, lm, g, lm.square().sum(1))
    dinv = torch.addcmul(const(('r', ridge), lambda: torch.tensor(ridge, device='cuda')), w, (1 - Kc.amax(1).square_() / (1 + jit)).clamp_min_(0)).reciprocal_()
    return (Kc.mul_(sw[:, None]), dinv)

def fitc_b(Kw, dinv, Li):
    PhiT = Li @ Kw.T
    PdT = PhiT * dinv
    return torch.linalg.solve_triangular(chol(torch.addmm(eye(len(Li)), PhiT, PdT.T)), PdT, upper=False)

def pcg_t(mvt, prect, t, steps):
    m, n = t.shape
    D = t.new_empty(steps + 1, m, n)
    G = t.new_empty(steps + 1, m, 1)
    r = t.clone()
    z = prect(r)
    rho = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
    d = torch.div(z, rho, out=D[0])
    for k in range(steps):
        ap = mvt(d)
        g = torch.bmm(d[:, None, :], ap[:, :, None], out=G[k][:, :, None])[:, 0]
        r.addcdiv_(ap, g, value=-1)
        z = prect(r)
        rn = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
        d = torch.addcdiv(d, z, rn, out=D[k + 1])
        if k == steps - 1:
            torch.mul(g, rho / rn, out=G[steps])
        rho = rn
    return (D / G).sum(0)
HK_K, HK_LAM, HK_T, HK_C, HK_M, NU = (70, 8.0, 2.0, 3, 2048, 384)

def hknn(zq, zv, yv, valid, cls):
    nq, C = cls.shape
    dim = zq.shape[1]
    qa = torch.cat([zq, zq.square().sum(1, keepdim=True), torch.ones_like(zq[:, :1])], 1)
    va = torch.cat([-2 * zv, torch.ones_like(zv[:, :1]), zv.square().sum(1, keepdim=True)], 1)
    D = (qa @ va.T).half().clamp_min_(0)
    cd, ci = D.topk(HK_M, 1, largest=False)
    ok = valid[ci][:, None, :] & (yv[ci][:, None, :] == cls[:, :, None])
    sv, si = cd[:, None, :].masked_fill(~ok, float('inf')).topk(HK_K, 2, largest=False)
    nb = ci.gather(1, si.reshape(nq, -1)).reshape(nq, C, HK_K)
    bad = torch.isinf(sv)
    nb = torch.where(bad, nb[:, :, :1], nb)
    nbz = zv[nb].reshape(nq * C, HK_K, dim)
    m = nbz.mean(1, keepdim=True)
    v = nbz - m
    r = (zq[:, None, :].expand(nq, C, dim).reshape(nq * C, 1, dim) - m).mT
    eye = torch.eye(dim, device=zq.device) * HK_LAM
    vs = v * 1024.0
    vh = vs.half()
    vl = (vs - vh.float()).half()
    g = torch.bmm(vh.mT, vh, out_dtype=F32)
    cc = torch.bmm(vh.mT, vl, out_dtype=F32)
    g.add_(cc).add_(cc.mT)
    solution = _hull_cg_solve(torch.add(eye[None], g, alpha=2.0 ** (-20)), r)
    out = (solution.square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)
    return out.masked_fill(bad[:, :, 0], 1e+30)

def hk_fix(s, za, ty, conf, w):
    n = len(ty)
    top = s.topk(2, 1).values
    marg = top[:, 0] - top[:, 1]
    keep = marg >= conf
    uidx = marg.topk(NU, largest=False).indices
    cls = s[uidx].topk(HK_C, 1).indices
    valid = torch.cat([torch.ones(n, dtype=torch.bool, device=s.device), keep])
    d = hknn(za[n:][uidx], za, torch.cat([ty, s.argmax(1)]), valid, cls)
    dn = d - d.min(1, keepdim=True).values
    pr = torch.softmax(-dn / (HK_T * d.min(1).values.mean().clamp_min(1e-12)), 1)
    add = torch.zeros_like(s[uidx]).scatter_(1, cls, pr) * (w * (~keep[uidx]).float())[:, None]
    return s.index_add(0, uidx, add)

import ctypes as ct
import ctypes.util
import glob
import time

def _library(name, candidates):
    found = ctypes.util.find_library(name)
    paths = ([found] if found else []) + candidates
    errors = []
    for path in paths:
        try:
            return ct.CDLL(path)
        except OSError as error:
            errors.append(str(error))
    raise RuntimeError(f'Cannot load {name}: ' + '; '.join(errors))

class Module:

    def __init__(self, source, name='project_kernel.cu', architecture='sm_80', *, line_info=False):
        import torch
        torch.cuda.init()
        self.cuda = _library('cuda', ['libcuda.so.1'])
        candidates = glob.glob('/usr/local/cuda*/targets/x86_64-linux/lib/libnvrtc.so*')
        candidates += glob.glob('/usr/local/lib/python*/site-packages/nvidia/*/lib/libnvrtc.so*')
        self.nvrtc = _library('nvrtc', candidates)
        nv = self.nvrtc
        nv.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        nv.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
        for suffix in ('ProgramLog', 'CUBIN'):
            getattr(nv, 'nvrtcGet' + suffix + 'Size').argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
            getattr(nv, 'nvrtcGet' + suffix).argtypes = [ct.c_void_p, ct.c_void_p]
        nv.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
        program = ct.c_void_p()
        self._nvcheck(nv.nvrtcCreateProgram(ct.byref(program), source.encode(), name.encode(), 0, None, None), 'create')
        options = [('--gpu-architecture=' + architecture).encode(), b'--std=c++17', b'--fmad=false', b'--include-path=/usr/local/cuda/include']
        if line_info:
            options.append(b'--generate-line-info')
        args = (ct.c_char_p * len(options))(*options)
        start = time.perf_counter()
        code = nv.nvrtcCompileProgram(program, len(options), args)
        self.compile_ms = (time.perf_counter() - start) * 1000
        size = ct.c_size_t()
        nv.nvrtcGetProgramLogSize(program, ct.byref(size))
        log = ct.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(program, log)
        self.log = log.value.decode(errors='replace')
        try:
            self._nvcheck(code, 'compile: ' + self.log[-4000:])
            self._nvcheck(nv.nvrtcGetCUBINSize(program, ct.byref(size)), 'cubin size')
            binary = ct.create_string_buffer(size.value)
            self._nvcheck(nv.nvrtcGetCUBIN(program, binary), 'get cubin')
            self.binary = binary.raw
        finally:
            nv.nvrtcDestroyProgram(ct.byref(program))
        cu = self.cuda
        cu.cuGetErrorString.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
        cu.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        cu.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t]
        common = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p)]
        cu.cuLaunchKernel.argtypes = common + [ct.c_void_p]
        cu.cuLaunchCooperativeKernel.argtypes = common
        self.handle = ct.c_void_p()
        self._binary = ct.create_string_buffer(self.binary)
        start = time.perf_counter()
        self._check(cu.cuModuleLoadData(ct.byref(self.handle), self._binary), 'load cubin')
        self.load_ms = (time.perf_counter() - start) * 1000
        self.functions = {}

    def _nvcheck(self, code, operation):
        if code:
            raise RuntimeError(f'NVRTC {operation}: status {code}')

    def _check(self, code, operation):
        if code:
            message = ct.c_char_p()
            self.cuda.cuGetErrorString(code, ct.byref(message))
            raise RuntimeError(f'CUDA {operation}: {code} {message.value!r}')

    def function(self, name):
        if name not in self.functions:
            handle = ct.c_void_p()
            self._check(self.cuda.cuModuleGetFunction(ct.byref(handle), self.handle, name.encode()), 'get ' + name)
            self.functions[name] = handle
        return self.functions[name]

    def launch(self, name, grid, block, arguments, *, shared=0, cooperative=False):
        import torch
        holders = [ct.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in arguments]
        parameters = (ct.c_void_p * len(holders))(*(ct.cast(ct.byref(a), ct.c_void_p) for a in holders))
        args = (self.function(name), grid, 1, 1, block, 1, 1, shared, ct.c_void_p(torch.cuda.current_stream().cuda_stream), parameters)
        if cooperative:
            code = self.cuda.cuLaunchCooperativeKernel(*args)
        else:
            code = self.cuda.cuLaunchKernel(*args, None)
        self._check(code, 'launch ' + name)
_HULL_CUDA='__device__ __forceinline__ float dot64(float v,float*s){int lane=threadIdx.x%32;for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)s[threadIdx.x/32]=v;__syncthreads();if(threadIdx.x==0)s[2]=s[0]+s[1];__syncthreads();return s[2];}\nextern "C" __global__ void hull_cg(const float*A,const float*B,float*X,int d,int iterations){\n __shared__ float mat[64*65],p[64],s[3];int t=threadIdx.x,b=blockIdx.x;\n for(int i=t;i<64*64;i+=64){int row=i/64,col=i%64;mat[row*65+col]=(row<d&&col<d)?A[(b*d+row)*d+col]:0.f;}\n float x=0.f,r=t<d?B[b*d+t]:0.f;p[t]=r;__syncthreads();\n float rho=dot64(r*r,s),initial=rho;\n for(int it=0;it<iterations;++it){\n  if(rho<=initial*1.e-12f)break;\n  float ap=0.f;for(int j=0;j<64;++j)ap=fmaf(mat[t*65+j],p[j],ap);\n  float den=dot64(p[t]*ap,s);float alpha=rho/fmaxf(den,1.e-30f);\n  x=fmaf(alpha,p[t],x);r=fmaf(-alpha,ap,r);\n  float next=dot64(r*r,s),beta=next/fmaxf(rho,1.e-30f);\n  p[t]=fmaf(beta,p[t],r);__syncthreads();rho=next;\n }\n if(t<d)X[b*d+t]=x;\n}\n'

_HC={}
def _hull_cg_solve(a,b):
 a=a.contiguous();b=b.contiguous()
 batch,d,_=a.shape
 assert d<=64 and b.shape==(batch,d,1)
 dev=a.device.index
 if dev not in _HC:_HC[dev]=Module(_HULL_CUDA,'hull_cg.cu')
 out=torch.empty_like(b)
 _HC[dev].launch('hull_cg',batch,64,[a,b,out,ct.c_int(d),ct.c_int(48)])
 return out

def run(train_x, train_y, test_x, P):
    g = P['GAMMA']
    with torch.no_grad():
        lab = train_y.long()
        n, dim = train_x.shape
        raw = torch.cat([train_x, test_x]).float()
        xx = raw * raw.norm(dim=1, keepdim=True).clamp_min_(1e-06).pow_(-P['NPOW']).mul_(dim ** (P['NPOW'] / 2))
        x = xx[:n]
        y1h = F.one_hot(lab, 10).float()
        cnt = y1h.sum(0)
        Bm = lda(raw[:n], y1h, cnt, P['LDA_S'])
        Fm = qda_fisher(x, y1h, cnt, Bm, P)
        U = torch.add(eye(dim, 1 - P['UW0'] - P['UW1']), pow34(Fm, P), alpha=P['UW0']).add_(Bm, alpha=P['UW1'])
        za = features(xx, n, chol(U))
        zt = za[:n]
        zh = zt.half()
        zhl = torch.cat([zh, (zt - zh.float()).half(), const(('ones16', n), lambda: torch.ones(n, 1, device='cuda', dtype=F16))], 1)
        main = torch.cuda.current_stream()
        side = _stream('side')
        fa = {}

        def hook(lm):
            side.wait_stream(main)
            with torch.cuda.stream(side):
                fa['L'] = fitc_a(lm, g, P['JIT'])
        pts, plab, w, hist = compress(zt, zhl, lab, P, hook)
        sw = w.sqrt()
        lw = w.clamp_min(1e-30).log_().mul_(0.5)
        Kw, dinv = fitc_k(pts, w, sw, hist[1], g, P['RIDGE'], P['JIT'])
        side.wait_stream(main)
        with torch.cuda.stream(side):
            QT = fitc_b(Kw, dinv, fa['L'])
        Km = torch.mm(kaug(pts, g, lw, True), kaug(pts, g, lw, False).T, out_dtype=F32)
        Km = torch.exp(Km, out=torch.empty_like(Km, dtype=F16))
        Km.diagonal().add_(P['RIDGE'])
        main.wait_stream(side)
        dr = dinv[None]

        def prect(r):
            return torch.addmm(r * dr, r @ QT.T, QT, alpha=-1)

        def mvt(d):
            return torch.mm(d.half(), Km, out_dtype=F32)
        t = (plab == const(('c10', 0), lambda: torch.arange(10, device='cuda')[:, None])) * sw
        alpha = pcg_t(mvt, prect, t, P['STEPS']).mul_(sw)
        scores = flash_scores(za[n:2 * n], pts, alpha.T, g, P['VSC'])
        return hk_fix(scores, za[:n + scores.shape[0]], lab, 0.2, 3.0).argmax(1)

def kc_d3(train_x, train_y, test_x):
    key = (tuple(train_x.shape), tuple(test_x.shape), train_x.dtype, train_y.dtype, str(train_x.device))
    if key not in _GRAPHS:
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run(*static, P)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(*static, P)
        _GRAPHS[key] = (graph, static, out)
        gc.collect()
        gc.freeze()
        for _ in range(WARM):
            graph.replay()
        torch.cuda.synchronize()
    graph, static, out = _GRAPHS[key]
    static[0].copy_(train_x)
    static[1].copy_(train_y)
    static[2].copy_(test_x)
    graph.replay()
    return out.clone()
if __name__ == '__main__':
    import mnist
    print(mnist.score(kc_d3, difficulty=2), 'ms per call')
