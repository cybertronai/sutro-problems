"""4×4 outer-product matmul submission.

Loop order ``(k, i, j)`` with C = Σ_k A[:,k] ⊗ B[k,:]. During the inner
``j`` loop only ``A[i,k]`` is alive, so sA is a single cell (addr 1)
instead of caching the whole A column. That frees three cheap addresses,
which lets C/A/B all shift closer to origin. Read-count-sorted address
allocation:

  addr 1     : sA  (read 64×: every mul)
  addr 2     : tmp (read 48×: every accumulating add)
  addrs 3..6 : sB  (4 cells, each read 16×)
  addrs 7..22: C   (each read 4×: 3 adds + 1 output)
  addrs 23..38: A bulk (each loaded once)
  addrs 39..54: B bulk (each loaded once)

Run as a script to (re)write ``outer_product_4x4.ir`` and verify cost.
"""
from __future__ import annotations


def generate_outer_product_4x4() -> str:
    n = 4
    SA = 1
    TMP = 2
    sB = lambda j: 3 + j
    C  = lambda i, j: 7 + i * n + j
    A_base = 23
    B_base = A_base + n * n
    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: B_base + k * n + j

    inputs = ([A(i, k) for i in range(n) for k in range(n)] +
              [B(k, j) for k in range(n) for j in range(n)])
    outputs = [C(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    for k in range(n):
        for j in range(n):
            lines.append(f"copy {sB(j)},{B(k, j)}")
        for i in range(n):
            lines.append(f"copy {SA},{A(i, k)}")
            for j in range(n):
                if k == 0:
                    lines.append(f"mul {C(i,j)},{SA},{sB(j)}")
                else:
                    lines.append(f"mul {TMP},{SA},{sB(j)}")
                    lines.append(f"add {C(i,j)},{TMP}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_4x4  # noqa: E402

    ir = generate_outer_product_4x4()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outer_product_4x4.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_4x4(ir)
    print(f"outer_product_4x4.ir  cost={cost}")
    assert cost == 800, cost
