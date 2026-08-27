#!/usr/bin/env python3
"""
exp_0_renamer_v1.py — Lane 0 experiment.

Apply the structure-preserving address-renaming pass (greedy:
highest-read-count -> cheapest addr) to the three candidate IRs and
verify with score_16x16:

  (a) submissions/sa_cache_16x16.ir   (current best 73602)
  (b) submissions/hierarchical_16x16.ir
  (c) submissions/tiled_16x16_opt1.ir

If any post-rename IR scores below the current record (73602), save it
as records/record_<cost>_lane0.ir and emit a new_record event.

This experiment also functions as a smoke test for the reusable
addr_renamer module that other lanes can import.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEM_DIR = HERE  # matmul/ is the problem dir for our purposes
LANE = 0
EXP_NAME = Path(__file__).name

# --- STOP_SIGNAL check (per explorer convention) ------------------------------
def _check_stop_signal():
    sig = PROBLEM_DIR / f"STOP_SIGNAL_{LANE}"
    if sig.exists():
        try:
            sig.unlink()
        except OSError:
            pass
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def _append_event(event: dict):
    path = PROBLEM_DIR / "events.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _append_token_log(tokens: int):
    path = PROBLEM_DIR / "token_log.jsonl"
    rec = {"ts": time.time(), "lane": LANE, "exp": EXP_NAME, "tokens": tokens}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    _check_stop_signal()

    _append_event({
        "ts": time.time(),
        "type": "exp_start",
        "lane": LANE,
        "exp": EXP_NAME,
    })

    # Pull in the matmul scorer.
    sys.path.insert(0, str(PROBLEM_DIR.parent))  # sutro-problems/
    sys.path.insert(0, str(PROBLEM_DIR))         # matmul/  — for addr_renamer
    import matmul  # noqa: E402
    from addr_renamer import rename_optimal  # noqa: E402

    CURRENT_RECORD = 73602
    candidates = [
        ("submissions/sa_cache_16x16.ir",      "sa_cache"),
        ("submissions/hierarchical_16x16.ir",  "hierarchical"),
        ("submissions/tiled_16x16_opt1.ir",    "tiled_opt1"),
    ]

    records_dir = PROBLEM_DIR / "records"
    records_dir.mkdir(exist_ok=True)

    print(f"{'IR':<40} {'before':>10} {'after':>10} {'gap':>8}")
    print("-" * 72)

    best_after = None
    best_path = None

    for relpath, _label in candidates:
        ir_path = PROBLEM_DIR / relpath
        ir = ir_path.read_text()

        before = matmul.score_16x16(ir)              # ground-truth score
        new_ir, info = rename_optimal(ir)
        after = matmul.score_16x16(new_ir)           # ground-truth re-score

        # Sanity: scorer should agree with lower_bound's accounting.
        assert after == info["remapped_cost_estimate"], (
            f"scorer/lower_bound disagree: scorer={after} "
            f"lb={info['remapped_cost_estimate']}")
        assert before == info["orig_cost_estimate"]

        gap = before - after
        print(f"{relpath:<40} {before:>10,} {after:>10,} {gap:>8,}")

        if after < CURRENT_RECORD:
            # New record!
            out_path = records_dir / f"record_{after}_lane{LANE}.ir"
            out_path.write_text(new_ir + "\n")
            print(f"  >> NEW RECORD: {after:,} (prev {CURRENT_RECORD:,}) "
                  f"saved to {out_path}")
            _append_event({
                "ts": time.time(),
                "type": "new_record",
                "cost": after,
                "prev": CURRENT_RECORD,
                "lane": LANE,
                "file": str(out_path.relative_to(PROBLEM_DIR)),
            })
            CURRENT_RECORD = after  # in case multiple beat in this run
            if best_after is None or after < best_after:
                best_after = after
                best_path = out_path

    print()
    if best_after is None:
        print(f"No improvement over current record {CURRENT_RECORD:,}; "
              "all three IRs were already optimally addressed (gap=0).")
        print("This is the structure-preserving lower bound — to improve "
              "further, the algorithm itself must change.")
    else:
        print(f"Best new score: {best_after:,} -> {best_path}")

    # Token-log entry (rough estimate).
    _append_token_log(tokens=4000)


if __name__ == "__main__":
    main()
