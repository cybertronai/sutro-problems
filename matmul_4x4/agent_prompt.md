# Matmul 4x4 optimization — explorer brief

## Problem

Compute `C = A @ B` for 4×4 integer matrices using a 4-op IR (`add`, `sub`,
`mul`, `copy`) under the **simplified Dally cost model**:

* Memory is a 2-D upper half-plane indexed by positive integers.
* Reading address `addr` costs `⌈√addr⌉` (so addr 1 → 1, addrs 2–4 → 2, addrs
  5–9 → 3, addrs 10–16 → 4, addrs 17–25 → 5, …).
* **Writes are free. Arithmetic is free. Input placement is free.** Only reads
  pay. Output addresses cost one read each at exit.

The baseline naive triple-loop is **1,316** (`submissions/baseline_4x4.ir`).
Goal: minimize cost on `score_4x4`.

## API

From the parent `sutro-problems/` directory:

```python
import sys, os
sys.path.insert(0, '/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems')
import matmul

cost = matmul.score_4x4(ir_string)   # raises if incorrect; returns int cost
```

Inputs convention: A flattened row-major then B flattened row-major, so 2·n² = 32
values total. Outputs: C flattened row-major (16 values).

The IR format is one instruction per line (`;` is also a separator). First
line is the comma-separated input address list, last line is the output
address list. Example for 1×1:

```
1,2
mul 3,1,2
3
```

Two-operand short form: `add dest, src` ≡ `add dest, dest, src`.

## Cost-model intuition

* `cost(1)=1, cost(2..4)=2, cost(5..9)=3, cost(10..16)=4, cost(17..25)=5,
  cost(26..36)=6, cost(37..49)=7, cost(50..64)=8, …`
* Squeezing a hot cell to lower addrs is high-leverage. The dominant cost is
  **reads of frequently-accessed scratchpad cells in the inner loop**, not bulk
  reads.
* You can copy elements of A and B to low scratch addresses (like 1, 2, 3, etc.)
  using `copy` instructions, and then reuse them to save read costs in the actual
  multiplications.

## Files in this directory

* `matmul.py` / `__init__.py` — scorer + baselines.
* `records/` — record files in the format the manager tracks.
* `directions.json` — what each lane is exploring.
* `events.jsonl`, `token_log.jsonl` — append-only activity logs.

## How to run experiments

1. Create `exp_<lane_id>_<short_desc>.py` in this directory.
2. Inside `__main__`:
   * Check for `STOP_SIGNAL_<lane_id>` and exit if present.
   * Generate one or more candidate IRs.
   * Score each with `matmul.score_4x4`.
   * If the cost beats the best record so far, save the IR as
     `records/record_<cost>_lane<lane_id>.ir` (no leading zeros), and append
     a `new_record` event to `events.jsonl`.
   * Append an `exp_start` event when you launch a new variant.
   * Append an entry to `token_log.jsonl` with your lane id and an estimate of
     tokens used during this experiment.
3. Iterate within your direction until you exhaust ideas or get a STOP_SIGNAL.
