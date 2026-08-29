# Sparse parity

The **k-parity** of a bit string is the sum mod 2 of its k secret bits.

![One instance: m = 18 strings of n = 32 bits (three shown), each labeled with the parity of the k = 5 highlighted secret positions](doc/task.svg)

**Task:** given the m strings and their parities, figure out the locations of
the secret bits. Every instance: n = 32 bits, k = 5 secret positions, m = 18
strings.

We are looking for the lowest-energy solutions at 20%, 40%, 60%, 80% and 100%
accuracy — the secret recovery rate, i.e. the fraction of instances where all
32 output cells exactly match the hidden mask — measured on the
[simplified Bill Dally model](https://github.com/cybertronai/simplified-dally-model)
([v3 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets),
8-bit):

## 20% target

| Date       | Cost       | Submission | Contributors | Description |
| -          | -:         | -          | -            | -           |
| 2026-08-27 | 17,331,683 | [report](submissions/scan127_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(127)` (Gray scan) |
| 2026-08-26 | 12,042,480 | [ir](submissions/isd8_mask32.ir), [report](submissions/isd8_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_isd_mask(8)` (ISD restarts) ★ best |

## 40% target

| Date       | Cost       | Submission | Contributors | Description |
| -          | -:         | -          | -            | -           |
| 2026-08-27 | 18,764,343 | [report](submissions/scan1023_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(1023)` (Gray scan) |
| 2026-08-27 | 17,945,660 | [ir](submissions/scan511_mask32.ir), [report](submissions/scan511_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(511)` (Gray scan) ★ best |

## 60% target

| Date       | Cost       | Submission | Contributors | Description |
| -          | -:         | -          | -            | -           |
| 2026-08-27 | 26,951,367 | [report](submissions/scan6143_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(6143)` (Gray scan) |
| 2026-08-26 | 23,676,539 | [ir](submissions/scan4095_mask32.ir), [report](submissions/scan4095_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(4095)` (Gray scan) ★ best |

## 80% target

| Date       | Cost       | Submission | Contributors | Description |
| -          | -:         | -          | -            | -           |
| 2026-08-27 | 33,501,030 | [report](submissions/scan10239_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(10239)` (Gray scan) |
| 2026-08-27 | 30,226,172 | [ir](submissions/scan8191_mask32.ir), [report](submissions/scan8191_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(8191)` (Gray scan) ★ best |

## 100% target

| Date       | Cost       | Submission | Contributors | Description |
| -          | -:         | -          | -            | -           |
| 2026-08-26 | 43,325,468 | [ir](submissions/scan_full_mask32.ir), [report](submissions/scan_full_mask32.md), [py](mask_sparse_parity.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_scan(16383)` (Gray scan, full walk) ★ best |

![Energy vs secret recovery rate for the two solution families](doc/mask32_energy_vs_recovery.png)



- Github Pages: [sparse parity](https://cybertronai.github.io/sutro-problems/sparse-parity)
- [Literature background](https://cybertronai.github.io/sutro-problems/docs/spatial-model-analysis.html)


<details>
<summary>Submission instructions for agents</summary>

- A submission is one straight-line program (an **IR**) in the v3 instruction
  set: 8-bit cells, no loops, no branches, no data-dependent addressing, at
  most **2,000,000 lines** (every line counts, declarations included).
- An IR is plain text. Line 1 declares the input cell addresses
  (comma-separated — you choose the addresses; the grader writes the inputs
  there), the last line declares the 32 output addresses, and each line
  between is one op (`set`, `copy`, `not`, `abs`, `and`, `or`, `xor`, `add`,
  `sub`, `mul`, `div`, `cmp`, `select`), e.g. `xor 3,1,2`. Complete example:
  [submissions/scan_full_mask32.ir](submissions/scan_full_mask32.ir).
- **Inputs**, in declaration order: the 18×32 training bits (row-major), then
  the 18 parities. **Outputs:** the 32 declared cells, each holding exactly
  0 or 1; cell c = 1 iff bit position c is secret. An instance scores 1 only
  on an exact 32-cell match (training sets are uniquely identifiable, so 100%
  is attainable).
- **Energy** is the program's static read cost: every operand read of address
  a costs ⌈√a⌉, and each declared output cell is charged one final read.
- Score locally against the deterministic dev suite — run from this directory
  (needs only numpy). All `generate_*` calls on this page are functions of
  [mask_sparse_parity.py](mask_sparse_parity.py); `generate_isd_mask` wraps
  the ISD algorithm of the superseded joint tier,
  [scaled_sparse_parity.py](scaled_sparse_parity.py).

  ```python
  import mask_sparse_parity as mp

  ir = mp.generate_scan(4095)      # or your own IR string
  res = mp.evaluate_mask(ir)       # 1,024-instance dev suite, a few seconds
  res.cost, res.recovery           # → (23676539, 0.742)
  ```

- Adjudication: `mp.evaluate_mask(ir, suite_key=None)` draws a fresh random
  2,048-instance suite each call (recovery noise ≈ ±2 pp; ~2⁻¹⁴ of instances
  are rank-deficient and can dip a full scan just below 100%).
- The table and figure above are measured on the dev suite; the full sweep is
  in [doc/mask32_bands.json](doc/mask32_bands.json). Regenerate both with
  `python3 generate_mask_graph.py` (~4 min).
- Submit a PR adding your `.ir` and generator under
  [submissions/](submissions/) and updating the accuracy band (table row) it
  improves.

</details>
