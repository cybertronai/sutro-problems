"""Lane 0 — final summary: Input-layout-only changes cannot beat 73602.

Conclusion (with empirical confirmation):

1. The current best schedule (V1 sa-cache TI=8, TJ=4) has UNIFORM per-cell reads:
     - Each A cell: 4 reads
     - Each B cell: 2 reads
   Confirmed by Step 1 of exp_0_nonuniform_reads.py.

2. Loop order swaps (bi,bj,bk) ↔ (bj,bi,bk) preserve uniformity and produce
   identical IR cost after renaming. Loop orders with bk-NOT-innermost require
   multi-tile sC and increase scratchpad cost. Confirmed by Step 2.

3. Partial caching (pre-loading some A or B rows into a scratchpad strip)
   creates non-uniform per-cell reads BUT increases TOTAL reads (each cached
   row adds 16 cache-strip reads per use without removing the inner-SA read).
   Even the partial-cache + renamer combination cannot beat 73602.
   Confirmed by Steps 3 and 4. Best partial = n=1 → 74066 (+464).

4. Asymmetric tile sizes (different (TI,TJ) per row band) DO create non-uniform
   reads (e.g., R=8 TJ0=2 TJ1=4 gives band 0 cells 8 reads, band 1 cells 2
   reads). However, switching to TJ=2 in any band increases total bulk reads
   (4096/TJ scales inversely), and the cost saved by skewed placement is less
   than the cost added by extra reads. Confirmed by exp_0_asym_tiles.py
   (sweep of 850 split configurations).

5. B-share schedule (outer bj, sC tiled by bi) reduces B bulk reads from 512
   to 256, but doubles the sC cell count, pushing all bulk addresses up by 32.
   Net: 75587 (worse than 73602). Confirmed by exp_0_b_share.py.

ROOT CAUSE: For the sa-cache template, total cost is dominated by sC
(28%) + A bulk (18%) + B bulk (15%) reads. Within input-layout transformations,
A and B both already sit at their optimal address bands. The renamer is a
no-op on V1(8,4). To improve, we must change the algorithm to either:
  (a) reduce total reads (e.g., Strassen — but cell-count blowup at 16x16 makes
      it lose by 27k per prior Lane 1 experiments).
  (b) reduce sC reads (e.g., dot-product style — but doubles bulk reads, far
      worse net).

The 73602 record stands. Lane 0's input-layout-only analysis is exhausted.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

BEST_PRIOR = 73602


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 0,
                            "exp": filename, "tokens": tokens}) + "\n")


def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 0, "exp": "exp_0_layout_summary.py"})

    print("Lane 0 — input layout exploration summary")
    print("=" * 60)
    print()
    print("Confirmed sub-experiments:")
    print("  exp_0_nonuniform_reads.py:")
    print("    - V1(8,4) A reads: uniform 4×, B reads: uniform 2× (no skew)")
    print("    - Partial-A-cache (n=1..16): renamed cost rises 464*n")
    print("    - Partial-B-cache (n=1..16): renamed cost rises 464*n")
    print("    - Best: 73,602 (no improvement)")
    print()
    print("  exp_0_b_share.py:")
    print("    - B-share (outer bj, multi-tile sC): 75,587 (matches prior result)")
    print("    - A-share (outer bi, multi-tile sC): 80,217 (matches prior h1)")
    print()
    print("  exp_0_asym_tiles.py:")
    print("    - 850 two-band split configurations, none beats 73,602")
    print("    - Asymmetric tiles do create non-uniform reads but cost more")
    print()
    print(f"FINAL: Best record remains {BEST_PRIOR}")
    print(f"Conclusion: input-layout-only space is exhausted for Lane 0.")
    print(f"Future progress requires algorithmic restructuring (different mult")
    print(f"counts or sC accumulation patterns).")

    log_token("exp_0_layout_summary.py", 1500)


if __name__ == "__main__":
    main()
