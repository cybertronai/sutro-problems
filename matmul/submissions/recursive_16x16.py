"""16×16 standard recursive (divide-and-conquer) matmul submission.

Splits each 16×16 matmul into eight half-size sub-products

    C[i:i+h, j:j+h] += A[i:i+h, k:k+h] · B[k:k+h, j:j+h]

with ``h`` halving at each level until a 1×1 leaf does a single
``mul``. The eight ``(di, dj, dk) ∈ {0, h}³`` children of every
internal node are emitted in lexicographic order, which yields a
Z-order (Morton) traversal of the 4,096 leaf multiplications — the
same access count as the naive triple loop, but reordered.

Layout (read-count-descending → ascending address):

  addr 1            : tmp           (read 3,840×, once per non-first leaf)
  addrs 2..257      : A bulk        (16 reads each, one per paired j)
  addrs 258..513    : B bulk        (16 reads each, one per paired i)
  addrs 514..769    : C bulk        (1 read at exit)

No scratchpad caching: this submission illustrates the recursive
structure cleanly. The cost it lands on is the floor for a 1×1-leaf
recursive scheme without any tiling — a useful reference point next
to the tiled / hierarchical variants below it.

Run as a script to (re)write ``recursive_16x16.ir`` and verify cost.
"""
from __future__ import annotations


def generate_recursive_16x16() -> str:
    n = 16

    TMP = 1
    A_base = 2
    B_base = A_base + n * n
    C_base = B_base + n * n
    A = lambda i, j: A_base + i * n + j
    B = lambda i, j: B_base + i * n + j
    C = lambda i, j: C_base + i * n + j

    inputs = ([A(i, j) for i in range(n) for j in range(n)] +
              [B(i, j) for i in range(n) for j in range(n)])
    outputs = [C(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    initialized: set[tuple[int, int]] = set()

    def recurse(i: int, j: int, k: int, size: int) -> None:
        if size == 1:
            if (i, j) in initialized:
                lines.append(f"mul {TMP},{A(i,k)},{B(k,j)}")
                lines.append(f"add {C(i,j)},{TMP}")
            else:
                lines.append(f"mul {C(i,j)},{A(i,k)},{B(k,j)}")
                initialized.add((i, j))
            return
        h = size // 2
        for di in (0, h):
            for dj in (0, h):
                for dk in (0, h):
                    recurse(i + di, j + dj, k + dk, h)

    recurse(0, 0, 0, n)
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_recursive_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "recursive_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"recursive_16x16.ir  cost={cost}")
