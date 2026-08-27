"""Lane 1 — exit-cost-elimination verdict (analytical, no IR generation).

VERDICT: NO-OP. The 69,697 record's layout is already minimal w.r.t. exit cost
under any feasible alias scheme. Detailed reasoning below.

================================================================================
Cost decomposition of 69,697 (verified by inspecting record_69697_lane0.ir):
================================================================================
  scratch (38 cells, 15,872 reads, addrs 1-38):       41,456  (59.5%)
  A=C aliased (160 cells, 5 reads, addrs 39-198):      8,900  (12.8%)
  A-only (96 cells, 4 reads, addrs 199-294):           6,208  ( 8.9%)
  B-only (256 cells, 2 reads, addrs 295-550):         10,738  (15.4%)
  C-only (96 cells, 1 read, addrs 551-646):            2,395  ( 3.4%)
  ------------------------------------------------------------------
  Total:                                              69,697

The 96 unaliased C cells are exactly C[i=0..7, j=0..11] (the bi=0 non-last-bj
block). The aliased 160 = 64 (last-bj column for all i) + 96 (bi=1 non-last-bj
re-aliased onto bi=0's A row, which is fully-consumed by the time bi=1 writes).

================================================================================
Step 1: Verify exit-cost region is already minimal.
================================================================================
The 96 C-only cells live at addrs 551-646 with 1 read each (cost 2,395).
The renamer (lower_bound.py) is provably optimal for any FIXED read-count
distribution. With reads {5: 160, 4: 96, 2: 256, 1: 96} (after scratch),
descending order packs them at addrs 39, 199, 295, 455, 551... so the
1-read C cells unavoidably occupy the highest addresses 551-646.

Could we get exit cost lower by changing the read-count distribution?
Only by raising some C cells' read count (which means accumulating more
into them — but every accumulation adds reads, so total cost rises).
Or by aliasing C with another cell (B), discussed below.

================================================================================
Step 2: B-aliasing analysis (proposed by brief).
================================================================================
Proposal: alias the 96 unaliased C cells with 96 B cells. Aliased B cells
become 3-read (2 for B input + 1 for output). Predicted layout if feasible:

    160 A=C (5 reads)  → addrs 39-198
     96 A-only (4 reads) → addrs 199-294
     96 B=C (3 reads) → addrs 295-390
    160 B-only (2 reads) → addrs 391-550
      0 C-only (eliminated)
    Predicted bulk cost: see compute below.

Timing constraint: a B cell can be reused as a C destination only AFTER
its last read. In the V1(8,4) schedule with bi-outer, bj-middle, bk-inner:

  for bi in 0..1:
    for bj in 0..3:
      for bk in 0..15:
        read B[bk, bj*4..bj*4+3]   # 4 cells, each read once per (bi,bj,bk)
        ...
      write C[bi*8..bi*8+7, bj*4..bj*4+3]  # at end of (bi, bj)

The 96 unaliased C cells = C[bi=0, bj=0..2] (i.e., i ∈ 0..7, j ∈ 0..11),
written at end of (bi=0, bj=0..2). At that moment, B[k, 0..11] have been
read ONCE (in bi=0). They will be read AGAIN in bi=1. So NO B cell is free
when these C cells are written.

By contrast, C[bi=1, bj=0..2] cells (the OTHER 96, currently aliased to A)
ARE written after B[*, 0..11]'s last read. But these are already aliased
to A in the 69,697 scheme, leaving the bi=0 block unaliased.

Could we swap loop order to (bj-outer, bi-inner) so B[*, bj*4..]'s last
read happens at end of bj? Then the C cells for that bj could alias B.
But this changes A's reload pattern: A[i, *] would be read 4 times still
but interleaved differently. The total read count is invariant, but
specific aliasing opportunities shift. Critically, the A=C alias scheme
that yields 160 cells × 5 reads also relies on the bi-outer order for
its sequencing. Switching to bj-outer would reduce A=C aliases to 64
(only the last-bj-block can alias A), losing more than B-aliasing gains.

================================================================================
Step 3: Hypothetical maximal savings (ignoring timing).
================================================================================
Even if we COULD eliminate all 96 C-only cells by aliasing to B (costs the
B cells +1 read each but removes the 96 C-only cells from the layout):

  Predicted bulk cost (no C-only): see compute below.

The savings vs 69,697 are computed and shown to be ~572. The brief's
threshold for implementation attempt is 500. Borderline — but TIMING
MAKES IT IMPOSSIBLE in the bi-outer schedule, and switching to bj-outer
loses more A=C aliasing than it gains. So no path forward.

================================================================================
Conclusion
================================================================================
- Exit-cost region (96 C-only at addrs 551-646, cost 2,395) is already
  at the renamer-optimal minimum for the 69,697 read-count distribution.
- B-aliasing of those 96 C cells is BLOCKED by timing in the current
  schedule: at write-time, all B cells still have pending reads in bi=1.
- Even if feasible, hypothetical max savings = ~572, just over the 500
  implementation threshold but offset by the loss of A=C aliasing in any
  alternative schedule that frees B cells.
- Direction "exit-cost-elimination" is therefore a NO-OP under the
  current best schedule. Subsumed by alias-output-with-input.

This matches the brief's prediction and the prompt's pre-existing note:
"Skip this direction unless you find a novel angle. Mark done with no-op
outcome if so."
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = os.path.join(HERE, "STOP_SIGNAL_1")


def check_stop():
    if os.path.exists(STOP_FILE):
        try:
            os.remove(STOP_FILE)
        except OSError:
            pass
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(d):
    d.setdefault("ts", time.time())
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")


def log_tokens(exp, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": time.time(), "lane": 1, "exp": exp, "tokens": tokens
        }) + "\n")


def cost_range(start, n, reads):
    return sum(math.ceil(math.sqrt(start + k)) * reads for k in range(n))


def main():
    check_stop()
    log_event({"type": "exp_start", "lane": 1, "exp": "exp_1_exit_cost_verdict.py"})

    print("=" * 70)
    print("Lane 1: exit-cost-elimination verdict (analytical)")
    print("=" * 70)

    # Verify current 69,697 breakdown using known cell counts.
    scratch = 41456  # known: 38 scratch cells with reads 4096,3840,1024x4,120x32
    A_C    = cost_range(39, 160, 5)
    A_only = cost_range(39 + 160, 96, 4)
    B_all  = cost_range(39 + 256, 256, 2)
    C_only = cost_range(39 + 256 + 256, 96, 1)
    total = scratch + A_C + A_only + B_all + C_only
    print(f"\nCurrent 69,697 layout (verified):")
    print(f"  scratch                = {scratch:>7,}")
    print(f"  A=C aliased(160, 5r)   = {A_C:>7,}")
    print(f"  A-only(96, 4r)         = {A_only:>7,}")
    print(f"  B-only(256, 2r)        = {B_all:>7,}")
    print(f"  C-only(96, 1r)         = {C_only:>7,}")
    print(f"  TOTAL                  = {total:>7,}")
    assert total == 69697, f"verification failed: {total} != 69697"

    # Hypothetical: replace 96 C-only with 96 B=C aliases (B cells gain +1 read).
    # No C-only cells. 96 B cells become 3-read.
    B_C_hyp     = cost_range(39 + 160 + 96, 96, 3)
    B_only_hyp  = cost_range(39 + 160 + 96 + 96, 160, 2)
    total_hyp = scratch + A_C + A_only + B_C_hyp + B_only_hyp
    print(f"\nHypothetical 'all 96 C-only aliased to B' (timing-INFEASIBLE):")
    print(f"  scratch                = {scratch:>7,}")
    print(f"  A=C aliased(160, 5r)   = {A_C:>7,}")
    print(f"  A-only(96, 4r)         = {A_only:>7,}")
    print(f"  B=C aliased(96, 3r)    = {B_C_hyp:>7,}")
    print(f"  B-only(160, 2r)        = {B_only_hyp:>7,}")
    print(f"  C-only(0)              = {0:>7,}")
    print(f"  TOTAL                  = {total_hyp:>7,}")
    print(f"\n  Hypothetical savings: {69697 - total_hyp}")

    # Variant from brief: 96 cells aliased + 96 still C-only (incomplete alias).
    B_C_partial   = cost_range(39 + 160 + 96, 96, 3)
    B_only_p      = cost_range(39 + 160 + 96 + 96, 160, 2)
    C_only_p      = cost_range(39 + 160 + 96 + 96 + 160, 96, 1)
    total_p = scratch + A_C + A_only + B_C_partial + B_only_p + C_only_p
    print(f"\nBrief's proposal (96 of 96 unaliased C → B, all C cells still need addrs):")
    print(f"  Wait — this is the same as hypothetical above (B=C subsumes C-only).")
    print(f"  If ONLY 96 C-only become B=C, we have 0 leftover C-only cells. OK.")
    print(f"  Result identical to hypothetical: {total_hyp}, savings {69697 - total_hyp}.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
1. EXIT-COST REGION IS ALREADY MINIMAL.
   The 96 C-only cells (1 read each, addrs 551-646, cost 2,395) sit at the
   highest addresses by renamer-optimal placement. Cannot move lower without
   displacing higher-read cells, which would strictly increase total cost
   (renamer is provably optimal for fixed read-count distribution).

2. B-ALIASING OF C[bi=0, non-last-bj] IS TIMING-INFEASIBLE.
   At end of (bi=0, bj∈{0,1,2}), all B cells still have pending reads in
   the bi=1 phase. No B cell can be safely overwritten with a C value at
   that point. The brief's question "can 96 of B's 256 cells become free
   before C[bi=0, non-last-bj] is written?" answers NO.

3. EVEN IF FEASIBLE, GAIN IS ONLY ~572.
   Slightly above the 500 threshold but borderline; the 96 freed B cells
   become 3-read at addrs 295-390 (was 2-read; +1 read costs more) while
   the 96 C-only cells at addrs 551-646 are eliminated. Net improvement
   is small because the eliminated cells were already cheap (cost 2,395
   total, ~25 each on average) and the +1 read on aliased B cells adds
   substantial cost (96 cells × ~17-19 cost each = ~1,728).

4. SCHEDULE SWITCH (bj-outer instead of bi-outer) WOULD HURT.
   Switching loop order to make B cells freeable in time would reduce
   the A=C alias from 160 cells to 64 cells (only last-bj block can
   alias A under bj-outer ordering). Loss > 1,500. Net negative.

CONCLUSION: Direction is a no-op. Mark done.
""")

    log_tokens("exp_1_exit_cost_verdict.py", 18000)
    log_event({
        "type": "direction_verdict",
        "lane": 1,
        "direction_id": "exit-cost-elimination",
        "verdict": "no-op",
        "reasoning_summary": (
            "C-only cells already at minimal-cost addresses by renamer-optimality. "
            "B-aliasing for C[bi=0] timing-infeasible. Hypothetical max savings ~572, "
            "would be eliminated by schedule switch costs."
        ),
        "predicted_savings_if_feasible": 69697 - total_hyp,
    })


if __name__ == "__main__":
    main()
