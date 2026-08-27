"""Profile the read-distribution of the current best (V1, TI=8, TJ=4)."""
from __future__ import annotations
import os, sys, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

# Reuse gen_v1 inline
N = 16
TI_LIST = [(8, 4), (4, 4), (16, 16), (4, 8)]


def gen_v1(TI, TJ):
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def cost_of(addr):
    return math.isqrt(addr - 1) + 1


def profile(ir, TI, TJ):
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    output_addrs = [int(x) for x in lines[-1].split(",")]
    reads = Counter()
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        ops = [int(x) for x in rest.split(",")]
        if head == "copy":
            dest, src = ops
            reads[src] += 1
        else:
            if len(ops) == 3:
                dest, s1, s2 = ops
            else:
                dest, s2 = ops
                s1 = dest
            reads[s1] += 1
            reads[s2] += 1
    for a in output_addrs:
        reads[a] += 1

    sB_lo = 3
    sB_hi = 2 + TJ
    sC_lo = 3 + TJ
    sC_hi = 2 + TJ + TI*TJ
    A_lo = sC_hi + 1
    A_hi = A_lo + N*N - 1
    B_lo = A_hi + 1
    B_hi = B_lo + N*N - 1
    C_lo = B_hi + 1

    sA, tmp, sB, sC, A_bulk, B_bulk, C_bulk = 0, 0, 0, 0, 0, 0, 0
    sA_count, tmp_count, sB_count, sC_count, A_count, B_count, C_count = 0,0,0,0,0,0,0
    for addr, cnt in sorted(reads.items()):
        c = cost_of(addr) * cnt
        if addr == 1:
            sA += c; sA_count += cnt
        elif addr == 2:
            tmp += c; tmp_count += cnt
        elif sB_lo <= addr <= sB_hi:
            sB += c; sB_count += cnt
        elif sC_lo <= addr <= sC_hi:
            sC += c; sC_count += cnt
        elif A_lo <= addr <= A_hi:
            A_bulk += c; A_count += cnt
        elif B_lo <= addr <= B_hi:
            B_bulk += c; B_count += cnt
        else:
            C_bulk += c; C_count += cnt
    print(f"  sA   addr 1            {sA_count:>6} reads cost={sA:>6}")
    print(f"  tmp  addr 2            {tmp_count:>6} reads cost={tmp:>6}")
    print(f"  sB   addr {sB_lo}-{sB_hi:<3}        {sB_count:>6} reads cost={sB:>6}")
    print(f"  sC   addr {sC_lo}-{sC_hi:<3}        {sC_count:>6} reads cost={sC:>6}")
    print(f"  Abulk {A_lo}-{A_hi}      {A_count:>6} reads cost={A_bulk:>6}")
    print(f"  Bbulk {B_lo}-{B_hi}      {B_count:>6} reads cost={B_bulk:>6}")
    print(f"  Cbulk {C_lo}+           {C_count:>6} reads cost={C_bulk:>6}")
    total = sA + tmp + sB + sC + A_bulk + B_bulk + C_bulk
    print(f"  TOTAL                              cost={total:>6}")
    return total


def main():
    for TI, TJ in TI_LIST:
        ir = gen_v1(TI, TJ)
        cost = score_16x16(ir)
        print(f"\nV1(TI={TI}, TJ={TJ}) cost={cost}")
        profile(ir, TI, TJ)


if __name__ == "__main__":
    main()
