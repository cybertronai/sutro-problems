"""16×16 hierarchical block outer-product matmul submission.

Loop order ``(ib, k, jb)``. Outer loop iterates over 4-row i-blocks; the
middle loop sweeps ``k``; the inner loop iterates 4-column j-blocks. This
keeps a 64-cell accumulator sC alive in cheap scratch for an entire
i-block's k-sweep, then copies it out to C bulk once per ib.

The asymmetry is the key trick: under this loop order each A cell is
loaded **1×** while each B cell is loaded **4×** (once per i-block).
Putting B at cheaper bulk addresses than A turns the extra reloads into
the cheaper of the two, so 4·SumB_cheap + 1·SumA_expensive ≪ 4·(SumA +
SumB) from a symmetric scheme.

Layout (read-count-descending → ascending address):

  addr 1            : tmp           (read 3,840×)
  addrs 2..5        : sA            (1,024 reads each, 4 cells)
  addrs 6..9        : sB            (1,024 reads each, 4 cells)
  addrs 10..73      : sC accumulator (64 reads each, 64 cells)
  addrs 74..329     : B bulk        (4 reads each — reloaded per ib)
  addrs 330..585    : A bulk        (1 read each)
  addrs 586..841    : C bulk        (1 read at exit)

Run as a script to (re)write ``hierarchical_16x16.ir`` and verify cost.
"""
from __future__ import annotations


def generate_hierarchical_16x16() -> str:
    n = 16
    block = 4
    nb = n // block

    TMP = 1
    sA = lambda ii: 2 + ii
    sB = lambda jj: 6 + jj
    sC = lambda jb, ii, jj: 10 + jb * 16 + ii * 4 + jj
    B_base = 74
    A_base = 330
    C_base = 586
    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: B_base + k * n + j
    C = lambda i, j: C_base + i * n + j

    inputs = ([A(i, k) for i in range(n) for k in range(n)] +
              [B(k, j) for k in range(n) for j in range(n)])
    outputs = [C(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    for ib in range(nb):
        for k in range(n):
            for ii in range(block):
                lines.append(f"copy {sA(ii)},{A(ib * block + ii, k)}")
            for jb in range(nb):
                for jj in range(block):
                    lines.append(f"copy {sB(jj)},{B(k, jb * block + jj)}")
                for ii in range(block):
                    for jj in range(block):
                        if k == 0:
                            lines.append(
                                f"mul {sC(jb, ii, jj)},{sA(ii)},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{sB(jj)}")
                            lines.append(
                                f"add {sC(jb, ii, jj)},{TMP}")
        for jb in range(nb):
            for ii in range(block):
                for jj in range(block):
                    lines.append(
                        f"copy {C(ib * block + ii, jb * block + jj)},"
                        f"{sC(jb, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_hierarchical_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hierarchical_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"hierarchical_16x16.ir  cost={cost}")
    assert cost == 80_217, cost
