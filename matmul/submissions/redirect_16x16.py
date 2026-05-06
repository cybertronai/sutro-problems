"""16×16 matmul with 2-level (4×8) super-block + sB-pin redirect trick.

Builds on the closed-form winner schedule_2level_singleB(Tio=4, Tjo=8) =
73,602 by adding the asymmetric address-reuse optimization: in each
(bi_o, bj_o, k>0, jb_in) cycle, after the inner ii loop's last mul reads
sB at addr 1, sB is dead. So we redirect the last mul's product to addr 1
itself (overwriting the dying sB), and the following add reads it from
addr 1 (cost 1) instead of TMP@addr 2 (cost 2). One cost saved per
redirect; 4 × 2 × 15 × 8 = 960 redirects → 73,602 − 960 = 72,642.

Layout:
  addr 1            : sB pin (single B cell, also TMP-on-last-mul)
  addr 2            : TMP    (used for non-last muls only)
  addrs 3..6        : sA cache (4 cells = Tii)
  addrs 7..38       : sC      (32 cells, super-block 4×8)
  addrs 39..294     : B bulk
  addrs 295..550    : A bulk
  addrs 551..806    : C bulk (output)
"""
from __future__ import annotations


def generate_redirect_16x16() -> str:
    n = 16
    Tio, Tjo = 4, 8
    Tii, Tji = 4, 1
    n_ibo, n_jbo = n // Tio, n // Tjo
    n_jbi = Tjo // Tji

    SB = 1
    TMP = 2
    sA = lambda ii: 3 + ii          # 3..6
    sC = lambda jb_in, ii: 7 + jb_in * Tii + ii  # 7..38

    B_base = 39
    A_base = B_base + n * n          # 295
    C_base = A_base + n * n          # 551
    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: B_base + k * n + j
    C = lambda i, j: C_base + i * n + j

    inputs = ([A(i, k) for i in range(n) for k in range(n)] +
              [B(k, j) for k in range(n) for j in range(n)])
    outputs = [C(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]

    for bi_o in range(n_ibo):
        for bj_o in range(n_jbo):
            for k in range(n):
                for ii in range(Tii):
                    lines.append(f"copy {sA(ii)},{A(bi_o * Tio + ii, k)}")
                for jb_in in range(n_jbi):
                    j = bj_o * Tjo + jb_in
                    lines.append(f"copy {SB},{B(k, j)}")
                    for ii in range(Tii):
                        if k == 0:
                            lines.append(f"mul {sC(jb_in, ii)},{sA(ii)},{SB}")
                        elif ii < Tii - 1:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{TMP}")
                        else:
                            # Redirect: last mul of jb_in writes product
                            # to addr 1 (overwriting the now-dead sB),
                            # so the following add pays cost 1 not 2.
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{SB}")
            for jb_in in range(n_jbi):
                j = bj_o * Tjo + jb_in
                for ii in range(Tii):
                    i = bi_o * Tio + ii
                    lines.append(f"copy {C(i, j)},{sC(jb_in, ii)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_redirect_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "redirect_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"redirect_16x16.ir  cost={cost:,}")
    assert cost == 72_642, f"cost mismatch: got {cost}, expected 72,642"
