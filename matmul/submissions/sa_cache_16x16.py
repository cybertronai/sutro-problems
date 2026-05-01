"""16×16 sA-cache + sB scratchpad matmul submission.

Loop order ``(bi, bj, bk)`` with tile sizes Ti=8, Tj=4. The key trick is
caching a single A element in addr 1 before the inner ``jj`` loop, so every
multiply reads sA at cost 1 regardless of the bulk address. sB holds Tj=4
B elements at addrs 3–6.

Layout (read-count-descending → ascending address):

  addr 1        : sA_cache     (read 4,096×: every mul)
  addr 2        : tmp          (read 3,840×: every accumulating mul)
  addrs 3..6    : sB           (1,024 reads each, Tj=4 cells)
  addrs 7..38   : sC           (128 reads each, Ti×Tj=32 cells)
  addrs 39..294 : A bulk       (4 reads each)
  addrs 295..550: B bulk       (2 reads each)
  addrs 551..806: C bulk       (1 read at exit)

Run as a script to (re)write ``new_record_sa_cache_73602.ir`` and verify cost.
"""
from __future__ import annotations

N = 16
TI, TJ = 8, 4


def generate_sa_cache_16x16() -> str:
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj
    A_base = 7 + TI * TJ          # 39
    B_base = A_base + N * N        # 295
    C_base = B_base + N * N        # 551
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


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_sa_cache_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sa_cache_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"new_record_sa_cache_73602.ir  cost={cost:,}")
    assert cost == 73_602, cost
