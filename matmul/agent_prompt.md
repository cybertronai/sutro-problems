# Matmul optimization — explorer brief

## Problem

Compute `C = A @ B` for 16×16 integer matrices using a 4-op IR (`add`, `sub`,
`mul`, `copy`) under the **simplified Dally cost model**:

* Memory is a 2-D upper half-plane indexed by positive integers.
* Reading address `addr` costs `⌈√addr⌉` (so addr 1 → 1, addrs 2–4 → 2, addrs
  5–9 → 3, addrs 10–16 → 4, addrs 17–25 → 5, …).
* **Writes are free. Arithmetic is free. Input placement is free.** Only reads
  pay. Output addresses cost one read each at exit.

The current 16×16 record is **73,602** (`submissions/sa_cache_16x16.ir`).
Baseline naive triple-loop is 340,704. Goal: minimize cost on `score_16x16`.

## API

From the parent `sutro-problems/` directory:

```python
import sys, os
sys.path.insert(0, '/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems')
import matmul

cost = matmul.score_16x16(ir_string)   # raises if incorrect; returns int cost
cost = matmul.score_4x4(ir_string)
cost = matmul.score_1x1(ir_string)
```

Inputs convention: A flattened row-major then B flattened row-major, so 2·n²
values total. Outputs: C flattened row-major.

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
* Squeezing a hot cell from addr 49 to addr 1 saves `(7−1)×reads` = **6× reads
  worth of cost** if it's read N times. The `tmp@1` swap alone shaved ~23k off
  the original 4×4-tiled version.
* The dominant cost is **reads of frequently-accessed scratchpad cells in the
  inner loop**, not bulk reads. Every saved inner-loop read (cost 1–2) times
  thousands of iterations beats large bulk-layout shuffles.

## Files in this directory

* `matmul.py` / `__init__.py` — scorer + baselines.
* `README.md` — record history with prior-art descriptions.
* `submissions/` — past records and their generators (study these!).
* `records/` — record files in the format the manager tracks.
* `directions.json` — what each lane is exploring.
* `events.jsonl`, `token_log.jsonl` — append-only activity logs.

## How to run experiments

1. Create `exp_<lane_id>_<short_desc>.py` in this directory (NOT in submissions/).
2. Inside `__main__`:
   * Check for `STOP_SIGNAL_<lane_id>` and exit if present.
   * Generate one or more candidate IRs.
   * Score each with `matmul.score_16x16`.
   * If the cost beats the best record so far, save the IR as
     `records/record_<cost>_lane<lane_id>.ir` (no leading zeros), and append
     a `new_record` event to `events.jsonl`.
   * Append an `exp_start` event when you launch a new variant.
   * Append an entry to `token_log.jsonl` with your lane id and an estimate of
     tokens used during this experiment.
3. Iterate within your direction until you exhaust ideas or get a STOP_SIGNAL.

## Useful prior submissions

| File                              | Cost   | Idea                                           |
| -                                 | -:     | -                                              |
| `submissions/baseline_16x16.ir`   | 340,704 | naive triple loop                              |
| `submissions/tiled_16x16.ir`      | 133,783 | 4×4 scratchpad tiles                           |
| `submissions/tiled_16x16_opt1.ir` | 110,743 | tmp@1 layout                                   |
| `submissions/hierarchical_16x16.ir` | 80,217 | asymmetric reload, B near origin, A far        |
| `submissions/sa_cache_16x16.ir` ★ | **73,602** | Ti=8 Tj=4, sA single-cell cache + sB rank-4  |

Read `submissions/sa_cache_16x16.py` first. It's the current best and most
existing improvements modify some piece of its layout.

## Read-count distribution of the current best (sa_cache_16x16, 73602)

```
addr 1     (sA_cache):  4,096 reads × cost 1 = 4,096
addr 2     (tmp):       3,840 reads × cost 2 = 7,680
addrs 3..6 (sB, 4 cells): 1,024 each × cost 2 = 8,192
addrs 7..38 (sC, 32 cells): 128 each, cost 3..6 ≈ 18,000
addrs 39..294 (A bulk, 256 cells): 4 each
addrs 295..550 (B bulk, 256 cells): 2 each
addrs 551..806 (C bulk, 256 cells): 1 each at exit
```

Hot scratchpad cells dominate. Reducing their read count or compressing them
to lower addrs is high-leverage. Reducing **bulk reload count** (the 4× on A,
2× on B) is the other lever — that's what Strassen and bigger tiles attack.
