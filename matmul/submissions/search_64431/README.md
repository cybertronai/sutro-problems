# Reproduce the 64,431 matmul program

This directory contains a self-contained semantic schedule generator and
portable address-allocation replay for [`best_64431.ir`](../best_64431.ir).
It constructs the seven row/column tiles, adds the selected persistent B
captures, applies five operation-list moves, and then performs the frozen
allocation passes. The submitted IR is read only for the final byte-equality
assertion.

## Run

From the repository root, using Python 3.12:

```sh
python3 -m pip install -r matmul/submissions/search_64431/requirements.txt
python3 matmul/submissions/search_64431/reproduce.py
```

Optional outputs:

```sh
python3 matmul/submissions/search_64431/reproduce.py \
  --output replayed_64431.ir --manifest replayed_64431.json
```

NumPy and OR-Tools are pinned because equal-cost flow choices can change exact
addresses and therefore the frozen bytes. No network is needed after dependency
installation.

## Schedule

| Parameter | Value |
|---|---|
| Column panels | `6,10` |
| Row groups | `4,4,4,4` / `5,5,6` |
| Contraction chunks | `1` / `2` |
| Contraction rotations | `3,11,0,0` / `0,0,4` |
| Descending-start tiles | first and last wide-panel tiles |
| Persistent B captures | 128 values in columns 8–15 |

The five moves are applied sequentially to the current operation list:

| Source | Destination | Length |
|---:|---:|---:|
| 7902 | 7905 | 2 |
| 7893 | 7991 | 1 |
| 1823 | 1825 | 1 |
| 226 | 222 | 4 |
| 1281 | 1287 | 1 |

## Exact checkpoints

| Checkpoint | Score | SHA-256 |
|---|---:|---|
| Captured generator | 1,177,063 | `6dca745d2e7fff3e23794dc28c6481fde140bb522922ab34c06d9e2c3cfb089a` |
| Generator after five moves | 1,177,063 | `3c86365a522ca3909e8b1a892ef38e9edc21f3c54eb873f074854ff7e638be13` |
| Lifetime-chain allocation | 64,458 | `0393e6775e7b73ea8f117dd541247a07f5e8de146359b2ac422cf87d9155f184` |
| Pair allocation: 2 rounds, seed 0 | 64,431 | `66929dab27c72e9714bf8a1ae77f1942b4201fe81d0d7a255a8547108eabadc3` |
| Pair allocation: 5 rounds, seed 11 | **64,431** | `9d94114a87fecd30168fbcf63931bbc98a50778984a11fe0c3b16940218bcf11` |

Every checkpoint passes the official scorer and the exact symbolic proof.
`core.py` supplies the semantic SSA model, liveness checks, emission, and proof
adapter. `pair_alloc.py` contains the deterministic two-tier min-cost-flow
refinement. `reproduce.py` owns the literal construction and all frozen checks.

The search and allocation are heuristic. Fixed-order allocation results do not
establish a lower bound for another operation order.
