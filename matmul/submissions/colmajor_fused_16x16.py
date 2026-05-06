"""16x16 matmul: column-major super-block order with fused last-add + copy-out.

Builds on ``colmajor_dead_input_outputs_packed_16x16`` and additionally fuses
each non-last super-block's k=15 final accumulate-update with the copy-out
to its output cell:

    standard:  mul TMP, sA, SB
               add sC[jb,ii], TMP            # k=15 update
               ... (later)
               copy OUT, sC[jb,ii]            # extra sC read

    fused:     mul TMP, sA, SB
               add OUT, sC[jb,ii], TMP       # 3-op: writes OUT, leaves sC stale

The k=14 partial sum stays in sC; the next super-block re-inits sC at k=0
via mul-into-sC, so the stale value is never read. We save one sC read per
output cell in each non-last super-block.
"""
from __future__ import annotations


def generate_colmajor_fused_16x16() -> str:
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

    b_hot = [(k, j) for k in range(n) for j in range(Tjo)]
    b_hot_set = set(b_hot)
    b_rest = [(k, j) for k in range(n) for j in range(n) if (k, j) not in b_hot_set]
    b_addr = {kj: B_base + idx for idx, kj in enumerate(b_hot + b_rest)}

    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: b_addr[(k, j)]

    super_block_order = [(bi_o, bj_o)
                         for bj_o in range(n_jbo)
                         for bi_o in range(n_ibo)]
    last_bi_o, last_bj_o = super_block_order[-1]

    dead: list[int] = []
    used: set[int] = set()
    c_spill: list[int] = [C_base + x for x in range(96)]
    c_spill_pos = 0
    out_addr: dict[tuple[int, int], int] = {}

    for bi_o, bj_o in super_block_order:
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

    for bi_o, bj_o in super_block_order:
        is_last_super = (bi_o, bj_o) == (last_bi_o, last_bj_o)
        for k in range(n):
            for ii in range(Tii):
                lines.append(f"copy {sA(ii)},{A(bi_o * Tio + ii, k)}")
            for jb_in in range(n_jbi):
                j = bj_o * Tjo + jb_in
                lines.append(f"copy {SB},{B(k, j)}")
                for ii in range(Tii):
                    if k == 0:
                        # Init via mul writing directly to sC.
                        lines.append(f"mul {sC(jb_in, ii)},{sA(ii)},{SB}")
                    elif k == n - 1 and not is_last_super:
                        # Final k: fuse update with copy-out.
                        i = bi_o * Tio + ii
                        out = out_addr[(i, j)]
                        if ii < Tii - 1:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {out},{sC(jb_in, ii)},{TMP}")
                        else:
                            # ii=3 redirect through sB.
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {out},{sC(jb_in, ii)},{SB}")
                    else:
                        if ii < Tii - 1:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{TMP}")
                        else:
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{SB}")

        # No explicit copy-out loop: the fused add at k=15 already wrote
        # output addresses for non-last super-blocks. The last super-block
        # uses sC directly as outputs.

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_colmajor_fused_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "colmajor_fused_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"colmajor_fused_16x16.ir  cost={cost:,}")
