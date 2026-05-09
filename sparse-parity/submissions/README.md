# Submitting a sparse-parity solver

> *Instructions AI-generated from prompt: "make sure there's python file, IR file and .md report"*

Each entry in this directory is **one** record-table row. To submit a new
solver, drop in three files that share a base name `<my_solver>`:

| file                       | purpose                                                                                              | required |
|----------------------------|------------------------------------------------------------------------------------------------------|----------|
| `<my_solver>.ir`           | The IR text that `score_small` / `score_medium` will run.                                            | ✅       |
| `<my_solver>.py`           | A Python generator that reproduces `<my_solver>.ir` from scratch — running `python <my_solver>.py` should rewrite the `.ir` file and print its cost. Any helper code your generator depends on goes here too. | ✅       |
| `<my_solver>.md`           | A short report describing the algorithm, the memory layout, an op-count breakdown, and what you learned (what worked, what didn't, surprises, what you'd try next). | ✅       |

Pick a base name that makes the algorithm or trick obvious (`ge_small`, `tiled_medium`, `aliased_outputs_small`, …) and append `_small` or `_medium` to make it clear which problem size the IR is scored against.

## How to verify a submission

From the repo root:

```bash
python -c "
import sys; sys.path.insert(0, 'sparse-parity'); sys.path.insert(0, 'sparse-parity/submissions')
import sparse_parity, my_solver
ir = my_solver.generate_my_solver()           # or whatever your generator is named
print('cost =', sparse_parity.score_small(ir))  # or score_medium
"
```

The scorer:

- runs the IR against several **random** canonical seeds (covering distinct secrets), so an IR that hard-codes the secret for one specific seed will fail; and
- raises if the cost varies across seeds (it shouldn't — for any given IR the cost is data-independent).

If the scorer returns a number, the submission is valid; that number is the entry's `Cost` column.

## What to put in the report (`<my_solver>.md`)

Use this header so the entry is parseable at a glance:

```markdown
# Sparse parity — <one-line method name> (<size>)

**Author:** [@your-github-handle](https://github.com/your-github-handle)
**Date:** YYYY-MM-DD
**Problem:** sparse parity (n=…, k=…, … train / … test)
**Cost:** <integer>
**IR:** [`<my_solver>.ir`](<my_solver>.ir)
**Generator:** [`<my_solver>.py`](<my_solver>.py)
**Method:** `generate_<my_solver>` (one-line algorithm summary)
```

Then one or two sections of prose:

- **Algorithm.** What the IR actually does, in enough detail that a reader can sanity-check it without re-deriving it from the IR.
- **Memory layout.** A small table mapping address ranges to roles (`pred`, `X_train`, `X_test`, scratch buffers, etc.).
- **Cost breakdown.** A short table of "section ↔ ops" with totals.
- **Findings.** What surprised you, what you tried that didn't work, what you'd try next, why the cost is what it is.

The [`baseline_small.md`](baseline_small.md), [`ge_small.md`](ge_small.md), and [`ge_medium.md`](ge_medium.md) reports are good templates.

## Adding the entry to the record table

Once the three files are in place, add a row to the appropriate "Record History" table in [`../README.md`](../README.md):

```markdown
| YYYY-MM-DD | <cost> | <time> | [ir](submissions/<my_solver>.ir), [report](submissions/<my_solver>.md), [py](submissions/<my_solver>.py) | [@your-handle](https://github.com/your-handle) | `generate_<my_solver>` (one-line summary) |
```

Time = single-run wall-clock of `generate_<my_solver>() + score_*(<ir>)` (see the footer of `../README.md`).

## IR conventions (v3, simplified Dally cost model)

- Three-address-code, one instruction per line; `;` is also a line separator. Addresses are positive integers.
- Inputs are passed as a flat list in this fixed order: `X_train` (row-major) → `y_train` → `X_test` (row-major). The IR's first line declares the addresses for these in the same order.
- Outputs are `m_test` predictions in the IR's last line; they must equal `y_test` exactly.
- Read cost per operand = `⌈√addr⌉`. `set` is free (no read); writes and arithmetic are free; every output address pays one standard read at exit.
- Supported ops: `add`, `sub`, `mul`, `div`, `copy`, `and`, `or`, `xor`, `not`, `abs`, `set`, `cmp d, a, b, <pred>`, `select d, c, t, f`. See the module docstring in [`../sparse_parity.py`](../sparse_parity.py) for the precise semantics.
