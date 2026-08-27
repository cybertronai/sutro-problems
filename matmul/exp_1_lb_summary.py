"""Lane 1 — final summary and LB-equivalence catalog.

Synthesizes findings across exp_1_schedule_search and exp_1_hilbert_climb:

1. The structure-preserving lower bound for the canonical sa-cache template
   (TI=8, TJ=4) is 73602, achieved exactly by the current submission.

2. Adjacent-instruction swaps preserve every cell's read count, so they
   cannot change the LB (verified empirically with 200-sample random walk
   in exp_1_schedule_search — all 29 valid swaps left LB unchanged).

3. Block-level perturbations (Hilbert tile order, tile pair fusion) also
   preserve read counts when the geometry is invariant.

4. Reducing sC to 16 cells (e.g. TI=4, TJ=4) actually concentrates more
   sC reads into hotter addresses, but is offset by the b_r jump from 2
   to 4 — the higher-volume B-bulk spreads to addresses 519..774 (cost
   23..28 each), losing more than sC saves.

5. The "skip-spill" idea (preserve sC values in their original addresses
   so they double as C output cells) was disproven: skipping spills means
   sC slots cannot be reused across (bi,bj) tiles, forcing sC to grow to
   nbi*nbj*TI*TJ = 256 cells, all reaching addrs ~7..262 with 16 reads
   each. LB rises to 103026.

6. The "TMP-distribution" perturbation (split TMP into k cells round-
   robin) cannot help: total TMP reads = 3840 is invariant, and splitting
   spreads to higher addresses while keeping each cell hot enough that
   it doesn't get cheap shells.

7. Several distinct schedules tie at 73602:
   - sa_cache TI=8 TJ=4 (canonical)
   - dual_bj_8x2 (effectively TI=8, sub-pairs of TJ=2 → equivalent geom)

This experiment writes a summary to events.jsonl and prints a final
report. No new record is produced; we strengthen the evidence that
73602 is the structure-preserving optimum for this template family.
"""
from __future__ import annotations
import os, sys, time, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_1")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 1,
                            "exp": filename, "tokens": tokens}) + "\n")


def lb_distrib(distrib):
    """Compute LB given (count, n_cells) tuples."""
    flat = []
    for c, n in distrib:
        flat += [c] * n
    flat.sort(reverse=True)
    total = 0
    for i, c in enumerate(flat):
        addr = i + 1
        total += c * (math.isqrt(addr - 1) + 1)
    return total


def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": "exp_1_lb_summary.py"})

    BEST = 73602
    N = 16

    # --- Catalog: all (TI, TJ) sa-cache shapes ---
    print("=" * 66)
    print("Lane 1 — LB catalog of sa-cache (TI, TJ) shapes")
    print("=" * 66)
    print(f"{'TI':>3} {'TJ':>3} {'sB':>5} {'sC':>6} {'a_r':>3} {'b_r':>3}  {'LB':>7}")
    for TI in (1, 2, 4, 8, 16):
        for TJ in (1, 2, 4, 8, 16):
            if N % TI or N % TJ: continue
            nbi, nbj = N // TI, N // TJ
            sB_cells = TJ
            sB_reads = N**3 // TJ
            sC_cells = TI * TJ
            sC_reads = N**3 // (TI * TJ)
            a_reads = nbj
            b_reads = nbi
            distrib = [(4096, 1), (3840, 1), (sB_reads, sB_cells),
                       (sC_reads, sC_cells), (a_reads, 256),
                       (b_reads, 256), (1, 256)]
            cost = lb_distrib(distrib)
            mark = " *" if cost == BEST else ""
            print(f"{TI:>3} {TJ:>3} {sB_cells:>5} {sC_cells:>6} "
                  f"{a_reads:>3} {b_reads:>3}  {cost:>7}{mark}")

    # --- Final summary log ---
    print()
    print("=" * 66)
    print("Conclusion")
    print("=" * 66)
    print(f"  Structure-preserving LB for sa-cache family: {BEST}")
    print(f"  Current submission achieves: {BEST} (LB-tight)")
    print(f"  No swap, block, fusion, or split perturbation improves LB.")
    print(f"  Beating 73602 requires changing the algorithm class")
    print(f"  (e.g. fewer fundamental reads, not just rearranging).")

    log_event({"ts": int(time.time()), "type": "lane1_summary",
               "lane": 1,
               "best_lb": BEST,
               "conclusion": "sa-cache family LB-tight at 73602; no "
                              "schedule perturbation (swap, block-reorder, "
                              "tile-fusion, TMP-split) reduces LB."})

    log_token("exp_1_lb_summary.py", 4500)


if __name__ == "__main__":
    main()
