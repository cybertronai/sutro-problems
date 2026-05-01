"""16×16 matmul: redirect (72,642) + last super-block outputs in sC.

The last super-block (bi_o=3, bj_o=1) computes 32 final C values that
sit in sC@7..38 right before the copy-out loop. Pointing those 32
output addresses *at sC* directly skips the 32 copy-outs and replaces
their high-cost C_bulk EXIT reads with low-cost sC EXIT reads.

Layout (vs redirect_16x16):
  addrs 7..38      : sC scratchpad (32 cells); ALSO output addrs for the
                     last super-block's 32 C cells (i in 12..15, j in 8..15).
  addrs 551..774   : C_bulk holding the other 224 output cells.
  Top 32 of C_bulk's old range (775..806) sit unused.

Predicted save: 162 (skipped copy-outs) + 756 (EXIT cost diff) = 918.
72,642 → 71,724.
"""
from __future__ import annotations


def generate_sc_outputs_16x16() -> str:
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
    A_base = B_base + n * n          # 295
    C_base = A_base + n * n          # 551
    A = lambda i, k: A_base + i * n + k
    B = lambda k, j: B_base + k * n + j

    # Last super-block: bi_o = n_ibo - 1, bj_o = n_jbo - 1 = (3, 1).
    last_bi_o, last_bj_o = n_ibo - 1, n_jbo - 1

    def in_last_block(i, j):
        return (i // Tio == last_bi_o) and (j // Tjo == last_bj_o)

    # Build the C output addressing:
    #   - cells in last super-block: pointed at sC scratchpad addr.
    #   - other 224 cells: contiguous in C_bulk@551..774, row-major over
    #     the non-last cells.
    other_cells = [(i, j) for i in range(n) for j in range(n)
                   if not in_last_block(i, j)]
    other_addr = {ij: C_base + idx for idx, ij in enumerate(other_cells)}

    def C_addr(i, j):
        if in_last_block(i, j):
            jb_in = j - last_bj_o * Tjo
            ii = i - last_bi_o * Tio
            return sC(jb_in, ii)
        return other_addr[(i, j)]

    inputs = ([A(i, k) for i in range(n) for k in range(n)] +
              [B(k, j) for k in range(n) for j in range(n)])
    outputs = [C_addr(i, j) for i in range(n) for j in range(n)]

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
                            # redirect last mul to sB addr (overwriting dying sB)
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{SB}")

            # Skip copy-out for the last super-block — its sC values are
            # the outputs themselves (referenced by C_addr above).
            if (bi_o, bj_o) == (last_bi_o, last_bj_o):
                continue

            for jb_in in range(n_jbi):
                j = bj_o * Tjo + jb_in
                for ii in range(Tii):
                    i = bi_o * Tio + ii
                    lines.append(f"copy {C_addr(i, j)},{sC(jb_in, ii)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_sc_outputs_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sc_outputs_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"sc_outputs_16x16.ir  cost={cost:,}")
