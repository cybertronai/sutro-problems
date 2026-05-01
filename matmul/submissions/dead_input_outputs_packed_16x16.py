"""16x16 matmul: dead-input outputs with B output cells packed lower.

Compared with ``dead_input_outputs_16x16``, this keeps the same arithmetic
schedule and liveness-safe output aliasing, but gives the B cells that later
serve as output storage the cheapest B-bulk addresses.  Those cells have five
reads each (four B reloads plus one exit read), while ordinary B cells have
four, so this is the greedy packing order within the B region.
"""
from __future__ import annotations


def generate_dead_input_outputs_packed_16x16() -> str:
    n = 16
    Tio, Tjo = 4, 8
    Tii, Tji = 4, 1
    n_ibo, n_jbo = n // Tio, n // Tjo
    n_jbi = Tjo // Tji

    SB = 1
    TMP = 2
    sA = lambda ii: 3 + ii
    sC = lambda jb_in, ii: 7 + jb_in * Tii + ii

    B_base = 39
    A_base = B_base + n * n
    C_base = A_base + n * n

    # The allocator will use the cheapest 32 dead B cells after block (3, 0).
    # Make those cells contiguous instead of interleaving them with j=8..15.
    b_hot = [(k, j) for k in range(4) for j in range(8)]
    b_rest = [(k, j) for k in range(n) for j in range(n)
              if (k, j) not in set(b_hot)]
    b_addr = {kj: B_base + idx for idx, kj in enumerate(b_hot + b_rest)}

    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: b_addr[(k, j)]

    last_bi_o, last_bj_o = n_ibo - 1, n_jbo - 1

    dead: list[int] = []
    used: set[int] = set()
    c_spill = [C_base + x for x in range(Tio * Tjo)]
    c_spill_pos = 0
    out_addr: dict[tuple[int, int], int] = {}

    for bi_o in range(n_ibo):
        for bj_o in range(n_jbo):
            block_cells = [
                (bi_o * Tio + ii, bj_o * Tjo + jb_in)
                for jb_in in range(n_jbi)
                for ii in range(Tii)
            ]

            if (bi_o, bj_o) == (last_bi_o, last_bj_o):
                for i, j in block_cells:
                    out_addr[(i, j)] = sC(j - last_bj_o * Tjo,
                                          i - last_bi_o * Tio)
                continue

            if bj_o == n_jbo - 1:
                for ii in range(Tio):
                    i = bi_o * Tio + ii
                    for k in range(n):
                        dead.append(A(i, k))

            if bi_o == n_ibo - 1:
                for k in range(n):
                    for jb_in in range(n_jbi):
                        j = bj_o * Tjo + jb_in
                        dead.append(B(k, j))

            dead.sort()
            for i, j in block_cells:
                while dead and dead[0] in used:
                    dead.pop(0)
                if dead:
                    addr = dead.pop(0)
                else:
                    addr = c_spill[c_spill_pos]
                    c_spill_pos += 1
                used.add(addr)
                out_addr[(i, j)] = addr

    inputs = ([A(i, k) for i in range(n) for k in range(n)] +
              [B(k, j) for k in range(n) for j in range(n)])
    outputs = [out_addr[(i, j)] for i in range(n) for j in range(n)]

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
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{SB}")

            if (bi_o, bj_o) == (last_bi_o, last_bj_o):
                continue

            for jb_in in range(n_jbi):
                j = bj_o * Tjo + jb_in
                for ii in range(Tii):
                    i = bi_o * Tio + ii
                    lines.append(f"copy {out_addr[(i, j)]},{sC(jb_in, ii)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_dead_input_outputs_packed_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dead_input_outputs_packed_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"dead_input_outputs_packed_16x16.ir  cost={cost:,}")
