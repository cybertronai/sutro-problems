#!/usr/bin/env python3
"""Benchmark the frozen MNIST-small QDA learner using two handwritten PTX kernels.

The statistics kernel uses 8 blocks of 512 threads to accumulate sufficient
statistics for 1,000 training examples. The scoring kernel uses 32 blocks of
320 threads; each block reduces those partials, independently fits all ten
9-dimensional class covariances, and scores up to 32 queries. Covariance
inversion uses warp-lane Gauss-Jordan elimination without pivoting.

Inputs and arithmetic are float32. Both kernels run on every CUDA graph replay,
so the measured task includes fitting and all 1,000 predictions. Data transfers,
allocation, PTX compilation, module loading, graph capture, and host preprocessing
are outside the timing and energy scope. Energy uses signed NVML total-energy
counter deltas with paired idle-power subtraction, without clipping negatives.

    python gpu_benchmark_ptx.py generated/payloads results/gpu_verification_benchmark.json 1000000 7

A directory input verifies every payload in its manifest before timing draw 0.
A single .npz input remains supported for reproducing historical measurements.
"""
import ctypes, hashlib, json, platform, statistics, time
from pathlib import Path
import numpy as np
import torch

C, D = 10, 9
TRI = [(i, j) for i in range(D) for j in range(i, D)]
TIDX = {ij: t for t, ij in enumerate(TRI)}
NT = len(TRI)
BLOCK, WARPS = 512, 16
SLOTS = 56                       # per class: count, 9 sums, 45 moments, 1 pad
NPARAM = 64                      # per class: 9 mu, 45 packed P', kappa, pad
LN2 = 0.6931471805599453
N_TRAIN = N_TEST = 1000
NB1 = 8                         # frozen statistics-block count
SPB = (N_TRAIN + NB1 - 1) // NB1
QPB = 32                        # one query per warp lane
NB2 = (N_TEST + QPB - 1) // QPB
BLOCK2 = QPB * C                     # 320 threads: warp c scores class c for the block's 32 queries


def build_kernels():
    """Emit the frozen float32 statistics and redundant-fit scoring kernels."""
    from pyptx import kernel, reg, ptx, smem, Tile
    from pyptx.types import f32, u32, s32, b32, u64, pred

    def stat_ij(t):
        if t == 0: return (0, 0)
        if t < 10: return (t - 1, t - 1)
        return TRI[t - 10]

    def emit_gauss_jordan(warp, lane, sbase, pglobal, shared_prm=None):
        """warp c (< 10): mean, covariance, inverse and kappa of class c from stats at sbase;
        parameters to global pglobal, or to shared memory at register address shared_prm."""
        zero = reg.scalar(f32, init=0.0); one = reg.scalar(f32, init=1.0); half = reg.scalar(f32, init=0.5)
        cb = reg.scalar(u32); ptx.inst.mad.lo.u32(cb, warp, SLOTS * 4, sbase)
        cnt = reg.scalar(f32); ptx.inst.ld.shared.f32(cnt, ptx.addr(cb))
        rcnt = reg.scalar(f32); ptx.inst.rcp.rn.f32(rcnt, cnt)
        mu = reg.array(f32, D)
        for j in range(D):
            a = reg.scalar(u32); ptx.inst.add.u32(a, cb, (1 + j) * 4)
            sv = reg.scalar(f32); ptx.inst.ld.shared.f32(sv, ptx.addr(a)); ptx.inst.mul.f32(mu[j], sv, rcnt)
        li = reg.scalar(u32); ptx.inst.min.u32(li, lane, D - 1)
        mui = reg.scalar(f32); ptx.inst.mov.f32(mui, mu[0])
        for j in range(1, D):
            pj = reg.scalar(pred); ptx.inst.setp.eq.u32(pj, li, j); ptx.inst.selp.f32(mui, mu[j], mui, pj)
        A = reg.array(f32, D); Pm = reg.array(f32, D)
        for j in range(D):
            am = reg.scalar(u32); ptx.inst.min.u32(am, li, j)
            bm = reg.scalar(u32); ptx.inst.max.u32(bm, li, j)
            t9 = reg.scalar(u32); ptx.inst.mul.lo.u32(t9, am, D)
            am1 = reg.scalar(u32); ptx.inst.sub.u32(am1, am, 1)
            tri = reg.scalar(u32); ptx.inst.mul.lo.u32(tri, am, am1); ptx.inst.shr.u32(tri, tri, 1)
            tt = reg.scalar(u32); ptx.inst.sub.u32(tt, t9, tri); ptx.inst.add.u32(tt, tt, bm); ptx.inst.sub.u32(tt, tt, am)
            ma = reg.scalar(u32); ptx.inst.mad.lo.u32(ma, tt, 4, cb); ptx.inst.add.u32(ma, ma, 10 * 4)
            mo = reg.scalar(f32); ptx.inst.ld.shared.f32(mo, ptx.addr(ma)); ptx.inst.mul.f32(mo, mo, rcnt)
            mm = reg.scalar(f32); ptx.inst.mul.f32(mm, mui, mu[j]); ptx.inst.sub.f32(A[j], mo, mm)
            isd = reg.scalar(pred); ptx.inst.setp.eq.u32(isd, li, j); ptx.inst.selp.f32(Pm[j], one, zero, isd)
        logdet = reg.scalar(f32, init=0.0)
        for pv_ in range(D):
            pivb = reg.scalar(b32); ptx.inst.mov.b32(pivb, A[pv_])
            sh = reg.scalar(b32); ptx.inst.shfl.sync.idx.b32(sh, pivb, pv_, 31, -1)
            piv = reg.scalar(f32); ptx.inst.mov.b32(piv, sh)
            lg = reg.scalar(f32); ptx.inst.lg2.approx.f32(lg, piv); ptx.inst.add.f32(logdet, logdet, lg)
            rpiv = reg.scalar(f32); ptx.inst.rcp.rn.f32(rpiv, piv)
            isp = reg.scalar(pred); ptx.inst.setp.eq.u32(isp, lane, pv_)
            with ptx.if_(isp):
                for j in range(D):
                    ptx.inst.mul.f32(A[j], A[j], rpiv); ptx.inst.mul.f32(Pm[j], Pm[j], rpiv)
            arow = reg.array(f32, D); prow = reg.array(f32, D)
            for j in range(D):
                b1 = reg.scalar(b32); ptx.inst.mov.b32(b1, A[j]); s1 = reg.scalar(b32); ptx.inst.shfl.sync.idx.b32(s1, b1, pv_, 31, -1); ptx.inst.mov.b32(arow[j], s1)
                b2 = reg.scalar(b32); ptx.inst.mov.b32(b2, Pm[j]); s2 = reg.scalar(b32); ptx.inst.shfl.sync.idx.b32(s2, b2, pv_, 31, -1); ptx.inst.mov.b32(prow[j], s2)
            nf = reg.scalar(f32); ptx.inst.neg.f32(nf, A[pv_])
            with ptx.if_(~isp):
                for j in range(D):
                    ptx.inst.fma.rn.f32(A[j], nf, arow[j], A[j]); ptx.inst.fma.rn.f32(Pm[j], nf, prow[j], Pm[j])
        pbo = reg.scalar(u32); ptx.inst.mul.lo.u32(pbo, warp, NPARAM * 4)
        if shared_prm is None:
            pb = reg.scalar(u64); ptx.inst.cvt.u64.u32(pb, pbo); ptx.inst.add.u64(pb, pb, pglobal)
        else:
            pbs = reg.scalar(u32); ptx.inst.add.u32(pbs, pbo, shared_prm)
        def store_param(word_off_reg, value):
            """word_off_reg: u32 register with the byte offset inside the class's parameter block"""
            if shared_prm is None:
                a64 = reg.scalar(u64); ptx.inst.cvt.u64.u32(a64, word_off_reg); ptx.inst.add.u64(a64, a64, pb); ptx.inst.st.global_.f32(ptx.addr(a64), value)
            else:
                a32 = reg.scalar(u32); ptx.inst.add.u32(a32, word_off_reg, pbs); ptx.inst.st.shared.f32(ptx.addr(a32), value)
        isrow = reg.scalar(pred); ptx.inst.setp.lt.u32(isrow, lane, D)
        with ptx.if_(isrow):
            lo = reg.scalar(u32); ptx.inst.shl.b32(lo, lane, 2); store_param(lo, mui)
            for j in range(D):
                okj = reg.scalar(pred); ptx.inst.setp.le.u32(okj, lane, j)
                with ptx.if_(okj):
                    t9 = reg.scalar(u32); ptx.inst.mul.lo.u32(t9, lane, D)
                    lm1 = reg.scalar(u32); ptx.inst.sub.u32(lm1, lane, 1)
                    tri = reg.scalar(u32); ptx.inst.mul.lo.u32(tri, lane, lm1); ptx.inst.shr.u32(tri, tri, 1)
                    tt = reg.scalar(u32); ptx.inst.sub.u32(tt, t9, tri); ptx.inst.add.u32(tt, tt, j); ptx.inst.sub.u32(tt, tt, lane); ptx.inst.add.u32(tt, tt, D)
                    po32 = reg.scalar(u32); ptx.inst.shl.b32(po32, tt, 2)
                    isd = reg.scalar(pred); ptx.inst.setp.eq.u32(isd, lane, j)
                    dbl = reg.scalar(f32); ptx.inst.add.f32(dbl, Pm[j], Pm[j])
                    val = reg.scalar(f32); ptx.inst.selp.f32(val, Pm[j], dbl, isd)
                    store_param(po32, val)
        is0 = reg.scalar(pred); ptx.inst.setp.eq.u32(is0, lane, 0)
        with ptx.if_(is0):
            ratio = reg.scalar(f32); ptx.inst.mul.f32(ratio, cnt, 1.0 / N_TRAIN)
            lp = reg.scalar(f32); ptx.inst.lg2.approx.f32(lp, ratio)
            hl = reg.scalar(f32); ptx.inst.mul.f32(hl, logdet, half)
            kk2 = reg.scalar(f32); ptx.inst.sub.f32(kk2, lp, hl); ptx.inst.mul.f32(kk2, kk2, LN2)
            ko = reg.scalar(u32, init=(D + NT) * 4); store_param(ko, kk2)

    @kernel(in_specs=(Tile(N_TRAIN, D, f32), Tile(N_TRAIN, 1, s32), Tile(N_TEST, D, f32)), out_specs=(Tile(NB1 * C * SLOTS, 1, f32), Tile(C * NPARAM, 1, f32), Tile(2, 1, s32), Tile(N_TEST, 1, s32)),
            grid=(NB1, 1, 1), block=(BLOCK, 1, 1), arch="sm_80")
    def qda_stats(X, Y, Q1, PART, PRM, COUNTER, OUT1):
        buf = smem.alloc(f32, (SPB * D + SPB + WARPS * C * SLOTS, 1))
        flag = smem.alloc(u32, (1, 1))
        px, py, pq1, pp, pprm, pcnt, po1 = ptx.global_ptrs(X, Y, Q1, PART, PRM, COUNTER, OUT1)
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        bid = reg.scalar(u32); ptx.inst.mov.u32(bid, ptx.special.ctaid.x())
        lane = reg.scalar(u32); ptx.inst.and_.b32(lane, tid, 31)
        warp = reg.scalar(u32); ptx.inst.shr.u32(warp, tid, 5)
        bbase = reg.scalar(u32); ptx.inst.mov.u32(bbase, buf)
        fbase = reg.scalar(u32); ptx.inst.mov.u32(fbase, flag)
        one = reg.scalar(f32, init=1.0)
        s_first = reg.scalar(u32); ptx.inst.mul.lo.u32(s_first, bid, SPB)
        for k in range((SPB * D + BLOCK - 1) // BLOCK):
            idx = reg.scalar(u32); ptx.inst.add.u32(idx, tid, k * BLOCK)
            gidx = reg.scalar(u32); ptx.inst.mad.lo.u32(gidx, s_first, D, idx)
            okc = reg.scalar(pred); ptx.inst.setp.lt.u32(okc, idx, SPB * D)
            okg = reg.scalar(pred); ptx.inst.setp.lt.u32(okg, gidx, N_TRAIN * D); ptx.inst.and_.pred(okc, okc, okg)
            with ptx.if_(okc):
                sa = reg.scalar(u32); ptx.inst.mad.lo.u32(sa, idx, 4, bbase)
                go = reg.scalar(u32); ptx.inst.shl.b32(go, gidx, 2)
                ga = reg.scalar(u64); ptx.inst.cvt.u64.u32(ga, go); ptx.inst.add.u64(ga, ga, px)
                w = reg.scalar(b32); ptx.inst.ld.global_.b32(w, ptx.addr(ga)); ptx.inst.st.shared.b32(ptx.addr(sa), w)
        ybase = reg.scalar(u32); ptx.inst.add.u32(ybase, bbase, SPB * D * 4)
        for k in range((SPB + BLOCK - 1) // BLOCK):
            idx = reg.scalar(u32); ptx.inst.add.u32(idx, tid, k * BLOCK)
            gidx = reg.scalar(u32); ptx.inst.add.u32(gidx, s_first, idx)
            okc = reg.scalar(pred); ptx.inst.setp.lt.u32(okc, idx, SPB)
            okg = reg.scalar(pred); ptx.inst.setp.lt.u32(okg, gidx, N_TRAIN); ptx.inst.and_.pred(okc, okc, okg)
            with ptx.if_(okc):
                go = reg.scalar(u32); ptx.inst.shl.b32(go, gidx, 2)
                ga = reg.scalar(u64); ptx.inst.cvt.u64.u32(ga, go); ptx.inst.add.u64(ga, ga, py)
                w = reg.scalar(b32); ptx.inst.ld.global_.b32(w, ptx.addr(ga))
                sa = reg.scalar(u32); ptx.inst.mad.lo.u32(sa, idx, 4, ybase); ptx.inst.st.shared.b32(ptx.addr(sa), w)
        ptx.bar.sync(0)
        iA = reg.scalar(u32, init=0); jA = reg.scalar(u32, init=0); iB = reg.scalar(u32, init=0); jB = reg.scalar(u32, init=0)
        for l in range(32):
            isl = reg.scalar(pred); ptx.inst.setp.eq.u32(isl, lane, l)
            ia, ja = stat_ij(l); ptx.inst.selp.u32(iA, ia, iA, isl); ptx.inst.selp.u32(jA, ja, jA, isl)
            if l + 32 < NT + 10:
                ib, jb = stat_ij(l + 32); ptx.inst.selp.u32(iB, ib, iB, isl); ptx.inst.selp.u32(jB, jb, jB, isl)
        isCountA = reg.scalar(pred); ptx.inst.setp.eq.u32(isCountA, lane, 0)
        isSumA = reg.scalar(pred); ptx.inst.setp.lt.u32(isSumA, lane, 10)
        hasB = reg.scalar(pred); ptx.inst.setp.lt.u32(hasB, lane, NT + 10 - 32)
        offA = reg.scalar(u32); ptx.inst.shl.b32(offA, iA, 2); offAj = reg.scalar(u32); ptx.inst.shl.b32(offAj, jA, 2)
        offB = reg.scalar(u32); ptx.inst.shl.b32(offB, iB, 2); offBj = reg.scalar(u32); ptx.inst.shl.b32(offBj, jB, 2)
        SPW = (SPB + WARPS - 1) // WARPS
        accA = reg.array(f32, C); accB = reg.array(f32, C)
        for c in range(C):
            ptx.inst.mov.f32(accA[c], 0.0); ptx.inst.mov.f32(accB[c], 0.0)
        s0 = reg.scalar(u32); ptx.inst.mul.lo.u32(s0, warp, SPW)
        for kk in range(SPW):
            s = reg.scalar(u32); ptx.inst.add.u32(s, s0, kk)
            gs = reg.scalar(u32); ptx.inst.add.u32(gs, s, s_first)
            ok = reg.scalar(pred); ptx.inst.setp.lt.u32(ok, s, SPB)
            okg = reg.scalar(pred); ptx.inst.setp.lt.u32(okg, gs, N_TRAIN); ptx.inst.and_.pred(ok, ok, okg)
            with ptx.if_(ok):
                ya = reg.scalar(u32); ptx.inst.mad.lo.u32(ya, s, 4, ybase)
                yv = reg.scalar(u32); ptx.inst.ld.shared.u32(yv, ptx.addr(ya))
                xa = reg.scalar(u32); ptx.inst.mad.lo.u32(xa, s, D * 4, bbase)
                aA = reg.scalar(u32); ptx.inst.add.u32(aA, xa, offA); xiA = reg.scalar(f32); ptx.inst.ld.shared.f32(xiA, ptx.addr(aA))
                aAj = reg.scalar(u32); ptx.inst.add.u32(aAj, xa, offAj); xjA = reg.scalar(f32); ptx.inst.ld.shared.f32(xjA, ptx.addr(aAj))
                aB = reg.scalar(u32); ptx.inst.add.u32(aB, xa, offB); xiB = reg.scalar(f32); ptx.inst.ld.shared.f32(xiB, ptx.addr(aB))
                aBj = reg.scalar(u32); ptx.inst.add.u32(aBj, xa, offBj); xjB = reg.scalar(f32); ptx.inst.ld.shared.f32(xjB, ptx.addr(aBj))
                prA = reg.scalar(f32); ptx.inst.mul.f32(prA, xiA, xjA); ptx.inst.selp.f32(prA, xiA, prA, isSumA); ptx.inst.selp.f32(prA, one, prA, isCountA)
                prB = reg.scalar(f32); ptx.inst.mul.f32(prB, xiB, xjB)
                for c in range(C):
                    mc = reg.scalar(pred); ptx.inst.setp.eq.u32(mc, yv, c)
                    ptx.inst.add.f32(accA[c], accA[c], prA, pred=mc)
                    ptx.inst.add.f32(accB[c], accB[c], prB, pred=mc)
        pbase = reg.scalar(u32); ptx.inst.add.u32(pbase, ybase, SPB * 4)
        wbase = reg.scalar(u32); ptx.inst.mad.lo.u32(wbase, warp, C * SLOTS * 4, pbase)
        for c in range(C):
            aa = reg.scalar(u32); ptx.inst.mad.lo.u32(aa, lane, 4, wbase); ptx.inst.add.u32(aa, aa, c * SLOTS * 4)
            ptx.inst.st.shared.f32(ptx.addr(aa), accA[c])
            ab = reg.scalar(u32); ptx.inst.add.u32(ab, aa, 32 * 4)
            ptx.inst.st.shared.f32(ptx.addr(ab), accB[c], pred=hasB)
        ptx.bar.sync(0)
        for kk in range(2):
            slot = reg.scalar(u32); ptx.inst.add.u32(slot, tid, kk * BLOCK)
            ok2 = reg.scalar(pred); ptx.inst.setp.lt.u32(ok2, slot, C * SLOTS)
            with ptx.if_(ok2):
                tot = reg.scalar(f32, init=0.0)
                for w in range(WARPS):
                    a = reg.scalar(u32); ptx.inst.mad.lo.u32(a, slot, 4, pbase); ptx.inst.add.u32(a, a, w * C * SLOTS * 4)
                    v = reg.scalar(f32); ptx.inst.ld.shared.f32(v, ptx.addr(a)); ptx.inst.add.f32(tot, tot, v)
                gidx = reg.scalar(u32); ptx.inst.mad.lo.u32(gidx, bid, C * SLOTS, slot)
                go = reg.scalar(u32); ptx.inst.shl.b32(go, gidx, 2)
                ga = reg.scalar(u64); ptx.inst.cvt.u64.u32(ga, go); ptx.inst.add.u64(ga, ga, pp)
                ptx.inst.st.global_.f32(ptx.addr(ga), tot)
        ptx.ret()

    @kernel(in_specs=(Tile(C * NPARAM, 1, f32), Tile(N_TEST, D, f32)), out_specs=(Tile(N_TEST, 1, s32),),
            grid=(NB2, 1, 1), block=(BLOCK2, 1, 1), arch="sm_80")
    def qda_score(PRM, Q, OUT):
        scores = smem.alloc(f32, (QPB * C, 1))
        stats = smem.alloc(f32, (C * SLOTS, 1))
        prm_s = smem.alloc(f32, (C * NPARAM, 1))
        pprm, pq, po = ptx.global_ptrs(PRM, Q, OUT)
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        bid = reg.scalar(u32); ptx.inst.mov.u32(bid, ptx.special.ctaid.x())
        lane = reg.scalar(u32); ptx.inst.and_.b32(lane, tid, 31)
        warp = reg.scalar(u32); ptx.inst.shr.u32(warp, tid, 5)
        scb = reg.scalar(u32); ptx.inst.mov.u32(scb, scores)
        muc = reg.array(f32, D); pk = reg.array(f32, NT); kap = reg.scalar(f32)
        # every block: reduce the NB1 partials (PRM here is the PART buffer), Gauss-Jordan into shared prm_s
        sbase = reg.scalar(u32); ptx.inst.mov.u32(sbase, stats); prb = reg.scalar(u32); ptx.inst.mov.u32(prb, prm_s)
        for kk in range((C * SLOTS + BLOCK2 - 1) // BLOCK2):
            slot = reg.scalar(u32); ptx.inst.add.u32(slot, tid, kk * BLOCK2)
            ok2 = reg.scalar(pred); ptx.inst.setp.lt.u32(ok2, slot, C * SLOTS)
            with ptx.if_(ok2):
                tot = reg.scalar(f32, init=0.0)
                for b_ in range(NB1):
                    go = reg.scalar(u32); ptx.inst.add.u32(go, slot, b_ * C * SLOTS); ptx.inst.shl.b32(go, go, 2)
                    ga = reg.scalar(u64); ptx.inst.cvt.u64.u32(ga, go); ptx.inst.add.u64(ga, ga, pprm)
                    v = reg.scalar(f32); ptx.inst.ld.global_.f32(v, ptx.addr(ga)); ptx.inst.add.f32(tot, tot, v)
                a0 = reg.scalar(u32); ptx.inst.mad.lo.u32(a0, slot, 4, sbase); ptx.inst.st.shared.f32(ptx.addr(a0), tot)
        ptx.bar.sync(0)
        emit_gauss_jordan(warp, lane, sbase, None, shared_prm=prb)
        ptx.bar.sync(0)
        pbs = reg.scalar(u32); ptx.inst.mad.lo.u32(pbs, warp, NPARAM * 4, prb)
        for j in range(D):
            a = reg.scalar(u32); ptx.inst.add.u32(a, pbs, j * 4); ptx.inst.ld.shared.f32(muc[j], ptx.addr(a))
        for t in range(NT):
            a = reg.scalar(u32); ptx.inst.add.u32(a, pbs, (D + t) * 4); ptx.inst.ld.shared.f32(pk[t], ptx.addr(a))
        ka = reg.scalar(u32); ptx.inst.add.u32(ka, pbs, (D + NT) * 4); ptx.inst.ld.shared.f32(kap, ptx.addr(ka))
        qi = reg.scalar(u32); ptx.inst.mad.lo.u32(qi, bid, QPB, lane)
        okq = reg.scalar(pred); ptx.inst.setp.lt.u32(okq, qi, N_TEST)
        sc = reg.scalar(f32, init=-3.0e38)
        with ptx.if_(okq):
            qoff = reg.scalar(u32); ptx.inst.mul.lo.u32(qoff, qi, D * 4)
            qaddr = reg.scalar(u64); ptx.inst.cvt.u64.u32(qaddr, qoff); ptx.inst.add.u64(qaddr, qaddr, pq)
            d = reg.array(f32, D)
            for j in range(D):
                ai = reg.scalar(u64); ptx.inst.add.u64(ai, qaddr, j * 4)
                qv = reg.scalar(f32); ptx.inst.ld.global_.f32(qv, ptx.addr(ai)); ptx.inst.sub.f32(d[j], qv, muc[j])
            accq = reg.scalar(f32, init=0.0)
            for i in range(D):
                rowacc = reg.scalar(f32, init=0.0)
                for j in range(i, D):
                    ptx.inst.fma.rn.f32(rowacc, pk[TIDX[(i, j)]], d[j], rowacc)
                ptx.inst.fma.rn.f32(accq, d[i], rowacc, accq)
            ptx.inst.fma.rn.f32(sc, accq, -0.5, kap)
        sa = reg.scalar(u32); ptx.inst.mad.lo.u32(sa, lane, C * 4, scb); ptx.inst.mad.lo.u32(sa, warp, 4, sa)
        ptx.inst.st.shared.f32(ptx.addr(sa), sc)
        ptx.bar.sync(0)
        isw0 = reg.scalar(pred); ptx.inst.setp.eq.u32(isw0, warp, 0)
        ptx.inst.and_.pred(isw0, isw0, okq)
        with ptx.if_(isw0):
            best = reg.scalar(f32, init=-3.0e38); label = reg.scalar(u32, init=0)
            for c in range(C):
                a = reg.scalar(u32); ptx.inst.mad.lo.u32(a, lane, C * 4, scb); ptx.inst.add.u32(a, a, c * 4)
                v = reg.scalar(f32); ptx.inst.ld.shared.f32(v, ptx.addr(a))
                better = reg.scalar(pred); ptx.inst.setp.lt.f32(better, best, v)
                ptx.inst.selp.f32(best, v, best, better); ptx.inst.selp.u32(label, c, label, better)
            ooff = reg.scalar(u32); ptx.inst.shl.b32(ooff, qi, 2)
            oaddr = reg.scalar(u64); ptx.inst.cvt.u64.u32(oaddr, ooff); ptx.inst.add.u64(oaddr, oaddr, po)
            ptx.inst.st.global_.u32(ptx.addr(oaddr), label)
        ptx.ret()

    return qda_stats, qda_score


class DriverLauncher:
    """cuModuleLoadData + cuLaunchKernel on PyTorch's current stream (graph-capturable)."""
    def __init__(self, ptx_text, entry):
        self.cu = ctypes.CDLL('libcuda.so.1')
        self.mod = ctypes.c_void_p(); self.fn = ctypes.c_void_p()
        torch.cuda.synchronize()
        err = self.cu.cuModuleLoadData(ctypes.byref(self.mod), ptx_text.encode())
        if err:
            raise RuntimeError(f'cuModuleLoadData failed with CUDA error {err}')
        err = self.cu.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, entry.encode())
        if err:
            raise RuntimeError(f'cuModuleGetFunction failed with CUDA error {err}')
    def launch(self, tensors, block, grid=1):
        ptrs = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
        params = (ctypes.c_void_p * len(ptrs))(*[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in ptrs])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        err = self.cu.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, 0, stream, params, None)
        if err:
            raise RuntimeError(f'cuLaunchKernel failed with CUDA error {err}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payloads(path):
    """Check the exported manifest before loading any GPU inputs."""
    if path.is_file():
        records = [{'path': path.name, 'sha256': sha256(path)}]
        base, provenance = path.parent, {}
    else:
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        records, base = manifest['draws'], path
        if len(records) != 11 or [r['draw'] for r in records] != list(range(11)):
            raise ValueError('the frozen submission requires exactly draws 0 through 10')
        seeds = manifest.get('seeds')
        if (not isinstance(seeds, list) or len(seeds) != 11
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != 11):
            raise ValueError('the manifest requires eleven distinct nonnegative integer seeds')
        if any(type(record.get('dataset_seed')) is not int
               or record['dataset_seed'] != seeds[record['draw']] for record in records):
            raise ValueError('record seed differs from the manifest seed for its draw')
        provenance = {'payload_manifest_sha256': sha256(manifest_path),
                      'payload_manifest': manifest}
    payloads = []
    for record in records:
        payload_path = base / record['path']
        if not payload_path.resolve().is_relative_to(base.resolve()):
            raise ValueError('payload path leaves manifest directory')
        if sha256(payload_path) != record['sha256']:
            raise ValueError(f'payload hash mismatch: {payload_path.name}')
        with np.load(payload_path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        for name, shape in [('x', (1000, 9)), ('q', (1000, 9)),
                            ('labels', (1000,)), ('test_labels', (1000,)),
                            ('frozen_predictions', (1000,)), ('dataset_seed', ())]:
            value = payload[name]
            if value.shape != shape:
                raise ValueError(f'{payload_path.name}: wrong {name} shape')
            if name in ('x', 'q'):
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'{payload_path.name}: {name} must be finite float32')
            elif value.dtype.kind not in 'iu':
                raise ValueError(f'{payload_path.name}: {name} must be integral')
            elif name != 'dataset_seed' and not ((value >= 0) & (value < C)).all():
                raise ValueError(f'{payload_path.name}: invalid {name}')
        if 'dataset_seed' in record and int(payload['dataset_seed']) != record['dataset_seed']:
            raise ValueError('payload seed differs from manifest')
        for name, expected in record.get('arrays', {}).items():
            value = payload[name]
            canonical = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<'), copy=False))
            actual = hashlib.sha256(canonical.tobytes(order='C')).hexdigest()
            if (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']
                    or actual != expected['sha256']):
                raise ValueError(f'{payload_path.name}: {name} differs from its array manifest')
        payloads.append((record, payload))
    return payloads, provenance


def main():
    import argparse
    import importlib.metadata
    import os
    import pynvml as nv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('payload', type=Path, help='payload .npz or all-draw payload directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('replays', type=int, nargs='?', default=1000000)
    parser.add_argument('rounds', type=int, nargs='?', default=7)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.replays <= 0 or args.rounds <= 0:
        parser.error('replays and rounds must be positive')
    for name, value in {'QDA_NB1': '8', 'QDA_QPB': '32',
                        'QDA_PTX_MODE': 'redundant', 'QDA_FP16': '0'}.items():
        if os.environ.get(name, value) != value:
            parser.error(f'{name} cannot override the frozen submission configuration ({value})')
    payloads, provenance = load_payloads(args.payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ks, kq = build_kernels()
    ptx_stats, ptx_score = ks.ptx(), kq.ptx()
    ptx_text = ptx_stats + ptx_score
    # Generated PTX is reproducible from this source; do not overwrite historic PTX.
    ptx_dir = Path(__file__).resolve().parent / 'generated/verification-ptx'
    ptx_dir.mkdir(parents=True, exist_ok=True)
    (ptx_dir / 'qda_stats.ptx').write_text(ptx_stats)
    (ptx_dir / 'qda_score.ptx').write_text(ptx_score)

    def entry_of(text):
        return next(line for line in text.splitlines() if '.entry' in line).split('.entry')[1].split('(')[0].strip()

    x = torch.empty((N_TRAIN, D), device='cuda', dtype=torch.float32)
    y = torch.empty(N_TRAIN, device='cuda', dtype=torch.int32)
    q = torch.empty((N_TEST, D), device='cuda', dtype=torch.float32)
    out = torch.empty(N_TEST, device='cuda', dtype=torch.int32)
    prm = torch.empty(C * NPARAM, device='cuda', dtype=torch.float32)
    part = torch.empty(NB1 * C * SLOTS, device='cuda', dtype=torch.float32)
    counter = torch.zeros(2, device='cuda', dtype=torch.int32)
    stats_launcher = DriverLauncher(ptx_stats, entry_of(ptx_stats))
    score_launcher = DriverLauncher(ptx_score, entry_of(ptx_score))

    def task():
        stats_launcher.launch([x, y, q, part, prm, counter, out], BLOCK, NB1)
        score_launcher.launch([part, q, out], BLOCK2, NB2)

    def load_device(payload):
        x.copy_(torch.from_numpy(payload['x']))
        y.copy_(torch.from_numpy(payload['labels'].astype(np.int32)))
        q.copy_(torch.from_numpy(payload['q']))

    def poison_scratch():
        out.fill_(-1)
        part.fill_(float('nan'))
        prm.fill_(float('nan'))

    load_device(payloads[0][1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            task()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        task()
    torch.cuda.synchronize()
    draws = []
    for index, (record, payload) in enumerate(payloads):
        load_device(payload)
        poison_scratch()
        task()
        eager = out.cpu().numpy().copy()
        graph_matches = []
        # Reuse the same captured graph after changing every draw's training and
        # query inputs. Poisoning scratch prevents cached parameters/labels from
        # satisfying the check without retraining and rewriting all outputs.
        for _ in range(3):
            poison_scratch()
            graph.replay()
            replay = out.cpu().numpy().copy()
            graph_matches.append(bool(np.array_equal(replay, eager)))
        frozen = payload['frozen_predictions']
        mismatch = np.flatnonzero(eager != frozen)
        draw = {'draw': record.get('draw', index), 'dataset_seed': int(payload['dataset_seed']),
                'payload_sha256': record['sha256'],
                'gpu_agrees_with_frozen_cpu_predictions': int((eager == frozen).sum()),
                'total': N_TEST, 'gpu_correct': int((eager == payload['test_labels']).sum()),
                'cpu_correct': int((frozen == payload['test_labels']).sum()),
                'gpu_predictions_int32_sha256': hashlib.sha256(eager.astype('<i4').tobytes()).hexdigest(),
                'mismatching_prediction_indices': mismatch.tolist(),
                'graph_replays_equal_eager': graph_matches,
                'scratch_poisoned_before_eager_and_each_replay': True}
        draws.append(draw)
        print(json.dumps({'verification': draw}), flush=True)
    validation = {'draws': draws, 'total': sum(r['total'] for r in draws),
                  'gpu_agrees_with_frozen_cpu_predictions': sum(r['gpu_agrees_with_frozen_cpu_predictions'] for r in draws),
                  'gpu_correct': sum(r['gpu_correct'] for r in draws),
                  'cpu_correct': sum(r['cpu_correct'] for r in draws),
                  'all_graph_replays_equal_eager': all(all(r['graph_replays_equal_eager']) for r in draws),
                  'intermediate_float_values_bitwise_equal_to_reference': False,
                  'all_eleven_frozen_draws_checked': len(draws) == 11,
                  'one_captured_graph_reused_across_changed_inputs': True}
    validation['passed'] = (validation['gpu_agrees_with_frozen_cpu_predictions'] == validation['total']
                            and validation['all_graph_replays_equal_eager'])
    nv.nvmlInit()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    # Resolve NVML by CUDA device UUID, so CUDA_VISIBLE_DEVICES cannot silently
    # select a different physical card for the energy measurement.
    device_uuid = str(properties.uuid)
    if not device_uuid.startswith(('GPU-', 'MIG-')):
        device_uuid = 'GPU-' + device_uuid
    handle = nv.nvmlDeviceGetHandleByUUID(device_uuid)

    def nvtext(value):
        return value.decode() if isinstance(value, bytes) else value

    provenance.update({'runner_sha256': sha256(__file__),
                       'ptx_sha256': hashlib.sha256(ptx_text.encode()).hexdigest(),
                       'benchmark_dataset_seed': int(payloads[0][1]['dataset_seed']),
                       'benchmark_payload_sha256': payloads[0][0]['sha256']})
    doc = {'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'hardware': {'name': nvtext(nv.nvmlDeviceGetName(handle)),
                        'uuid': nvtext(nv.nvmlDeviceGetUUID(handle)),
                        'memory_total_bytes': int(properties.total_memory),
                        'multiprocessors': properties.multi_processor_count,
                        'compute_capability': [properties.major, properties.minor]},
           'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                        'pyptx': importlib.metadata.version('pyptx'),
                        'nvidia-ml-py': importlib.metadata.version('nvidia-ml-py'),
                        'cuda': torch.version.cuda,
                        'driver': nvtext(nv.nvmlSystemGetDriverVersion()),
                        'python': platform.python_version(), 'platform': platform.platform()},
           'provenance': provenance, 'validation': validation,
           'kernel': {'launches': 2, 'mode': 'redundant', 'fp16_inputs': False,
                      'stats': {'grid': NB1, 'block': BLOCK, 'samples_per_block': SPB},
                      'score': {'grid': NB2, 'block': BLOCK2, 'queries_per_block': QPB},
                      'ptx_lines': len(ptx_text.splitlines())},
           'scope': 'CUDA-graph replays of complete training and 1000 predictions on device-resident normalized float32 inputs; includes host-dispatch gaps between graph replays; excludes transfers, preprocessing, allocation, JIT, module load and capture. Signed NVML total-energy deltas minus mean paired idle power times measured wall duration; negative estimates are retained.'}
    verification_path = args.output if args.verify_only else args.output.parent / 'gpu_verification.json'
    verification_path.write_text(json.dumps(doc, indent=2) + '\n')
    if not validation['passed']:
        nv.nvmlShutdown()
        raise SystemExit(f'GPU prediction verification failed; see {verification_path}')
    if args.verify_only:
        nv.nvmlShutdown()
        return

    load_device(payloads[0][1])
    poison_scratch()
    graph.replay()
    torch.cuda.synchronize()

    def stamp():
        before = time.perf_counter()
        energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
        after = time.perf_counter()
        return {'energy_mj': energy, 'time_s': (before + after) / 2,
                'read_duration_s': after - before}

    def interval(start, end):
        seconds = end['time_s'] - start['time_s']
        energy_mj = end['energy_mj'] - start['energy_mj']
        if seconds <= 0 or energy_mj < 0:
            raise RuntimeError('non-monotonic time or NVML energy counter')
        return seconds, energy_mj

    def idle():
        torch.cuda.synchronize()
        start = stamp()
        time.sleep(5.0)
        end = stamp()
        seconds, energy_mj = interval(start, end)
        return {'start': start, 'end': end, 'power_w': energy_mj / 1000 / seconds}

    rounds = []
    try:
        for index in range(args.rounds):
            before = idle()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start = stamp()
            start_event.record()
            for _ in range(args.replays):
                graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            end = stamp()
            after = idle()
            seconds, gross_mj = interval(start, end)
            idle_w = (before['power_w'] + after['power_w']) / 2
            row = {'round': index, 'repeats': args.replays,
                   'cuda_ms': start_event.elapsed_time(end_event) / args.replays,
                   'wall_ms': seconds * 1000 / args.replays,
                   'gross_mj': gross_mj / args.replays,
                   'adjusted_mj': (gross_mj - idle_w * seconds * 1000) / args.replays,
                   'idle_before_w': before['power_w'], 'idle_after_w': after['power_w'],
                   'average_power_w': gross_mj / 1000 / seconds,
                   'measurement_start': start, 'measurement_end': end,
                   'idle_before': before, 'idle_after': after}
            rounds.append(row)
            print(json.dumps(row), flush=True)
    finally:
        nv.nvmlShutdown()
    adjusted = [r['adjusted_mj'] for r in rounds]
    summary = {'cuda_ms_median': statistics.median(r['cuda_ms'] for r in rounds),
               'wall_ms_median': statistics.median(r['wall_ms'] for r in rounds),
               'gross_mj_median': statistics.median(r['gross_mj'] for r in rounds),
               'adjusted_mj_median': statistics.median(adjusted),
               'adjusted_mj_min': min(adjusted), 'adjusted_mj_max': max(adjusted),
               'adjusted_mj_sample_sd': statistics.stdev(adjusted) if len(adjusted) > 1 else None,
               'negative_adjusted_rounds': sum(value < 0 for value in adjusted),
               'energy_interpretation': 'Idle subtraction measures a small difference between much larger totals. Round spread and idle drift limit precision; this is not a statistical confidence interval.'}
    doc.update({'rounds': rounds, 'summary': summary,
                'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    args.output.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'validation': validation, 'hardware': doc['hardware']}, indent=2))


if __name__ == '__main__':
    main()
