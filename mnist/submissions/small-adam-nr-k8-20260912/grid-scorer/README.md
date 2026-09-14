# Adam-capable copy of the accepted static grid scorer

This directory is a modified copy of the accepted scorer in
`vendor/sutro-problems/mnist/submissions/grid-mlp-scoring-20260912/`
(which is read-only reference material and was not changed). It prices the same
ordered-FP32 MLP when it is trained with either the original fixed-batch SGD
rule (`--optimizer sgd`, the default) or the Adam rule used by the research
code in `new-spec/small-campaign/adam_confirm.py` (`adam_step` variant). The
Adam square root is approximated by a Newton-Raphson iteration whose count is
set by `--nr-iterations K` (default 12).
No Modal or GPU work was run; scoring is local CPU only and does not execute
the numbers.

## What changed

| File | Change |
| --- | --- |
| `score.py` | Added `--optimizer {sgd,adam}` (default `sgd`) and `--nr-iterations K` (default 12), both passed through to `build_mlp`. |
| `model_ir.py` | Added `optimizer='sgd'` and `nr_iterations=12` to `build_mlp`. The `sgd` path is the accepted generator unchanged and ignores `nr_iterations`. The `adam` path adds per-parameter `m` and `v` arrays (`mw1, vw1, mb1, vb1, mw2, vw2, mb2, vb2`), the per-step bias-correction table `bc`, three FP32 scalar temporaries in `z`, six constants (`0.9, 0.1, 0.999, 0.001, 1e-8, 1e-6`) in `k`, and `K` unrolled Newton-Raphson iterations of `add`/`mul`/`div`. |
| `affine.py` | Added one priced primitive to the opcode set: `div(dst,a,b)`, which ISA v4 already defines. It follows the existing conventions exactly: one charged read per source in listed order, then one charged destination write. No `sqrt` opcode exists anywhere; the Adam square root is a fixed Newton-Raphson sequence of the existing `add`, `mul` and `div` opcodes. |
| `README.md` | This file. |

The `sgd` program is byte-identical to the accepted scorer's output (see
Validation below), including when `--nr-iterations` is supplied, so no
published SGD score changes.

## Square root via Newton-Raphson (no sqrt opcode)

`sqrt(vhat)` in the Adam denominator is replaced by a fixed, deterministic
Newton-Raphson iteration using only `add`, `mul` and `div`:

```text
x      = vhat
y0     = x + 1e-6        (float32 literal; caps the low end of the domain)
y[k+1] = 0.5 * (y[k] + x / y[k])      for k = 0 .. K-1
sqrt(vhat) ~ yK
```

The IR evaluates each iteration as `q = x / y; q = y + q; y = 0.5 * q`, in
that order, with no fused operation. `y0 = x + 1e-6` (rather than the plain
scale-free seed `y0 = x`) also keeps the seed positive when `vhat = 0`, so no
division by zero or NaN can occur. `K` is chosen by `--nr-iterations` and
defaults to 12.

Error bound (float32 op order, measured on 800,001 log-spaced points over
`[1e-12, 1e3]`, which covers the largest measured `vhat`, 147.14):

| K | max relative error vs `np.sqrt(float32(x))` | max absolute error |
| ---: | ---: | ---: |
| 4 | 3.026e+01 | 3.610e+01 |
| 8 | 1.121e+00 | 1.543e-03 |
| 12 | 2.384e-07 | 1.907e-06 |

This is the accuracy/model-fidelity tradeoff: K = 4 and K = 8 are cheap but
can be off by factors of ~30 and ~2 respectively, worst near `vhat = 1e-6`
(the seed-offset region where `y0` overshoots). K = 12 reaches 2.4e-07; K = 13
reaches the float32 rounding floor of 1.192e-07, and larger K adds cost without
improving accuracy. On the six priced configuration below, one draw each, the
max relative error over the `vhat` values actually seen is identical to the
superset worst case (K=4: 3.026e+01, K=8: 1.121e+00, K=12: 2.384e-07), so the
real training data does reach the near-1e-6 region.

## Exact Adam lowering

For one parameter word `w` with its gradient word `g`, global minibatch step
index `t >= 1`, `step = float32(lr / batch)` and `eps = float32(1e-8)`:

```text
m    = 0.9 * m + 0.1 * g
v    = 0.999 * v + 0.001 * g * g
mhat = m / (1 - 0.9**t)
vhat = v / (1 - 0.999**t)
w    = w - step * mhat / (NR_sqrt(vhat, K) + eps)
```

`t = epoch * (n_train / batch) + batch_index + 1` is the global minibatch
index starting at 1. `m` and `v` start at zero and are additional scratch
arrays with the same size as each parameter array. The divisors
`1 - float32(0.9)**t` and `1 - float32(0.999)**t` are computed per step in
float32, exactly as `np.float32(1) - np.float32(beta)**t`, and are
materialized as literal FP32 words in the `bc` region (`2*t` words total);
every element update reads them back from scratch, and the literal
initialization writes are charged. The IR emits the ordered FP32 chain

```text
z1 = beta1*m; z2 = omb1*g; m = z1 + z2
z1 = beta2*v; z2 = omb2*g; z2 = z2*g; v = z1 + z2
z2 = v / c2                       (x = vhat)
z3 = x + 1e-6                     (y0)
K x: z1 = x/z3; z1 = z3 + z1; z3 = 0.5*z1
z1 = m / c1                       (mhat)
z3 = z3 + eps
z1 = step * z1
z1 = z1 / z3
w  = w - z1
```

with the same loop nesting and element order as the SGD path, no fused
multiply-add, and Python's left-to-right `step * mhat / (...)` evaluation.
All added reads, writes and instruction literals are charged by the same
affine histogram and distance-based access costs used for SGD.

Each Adam element update is `14 + 3*K` instructions: for K = 12 that is 50
(18 `mul`, 16 `add`, 15 `div`, 1 `sub`), versus SGD's 2. The Newton-Raphson
contributes the seed `add` plus K iterations of `div`+`add`+`mul`. Against the
same SGD program the executed-instruction deltas are `mul +(5+K)`,
`add +(4+K)`, `div +(3+K)` and zero `sub` per element, plus `set +17,955` for
the w48/e100 Adam regions (moment arrays, the `2*T`-word `bc` table and its
`2*T` initialization writes, three `z` words, and the enlarged `k` region).
For w48/e100 (P = 970 parameter words, T = 4,000 minibatch steps) at K = 12
that is exactly `mul +65,960,000`, `add +62,080,000`, `div +58,200,000`; at
K = 4 the `div` delta drops to 7 per element (`+27,160,000`).

## Validation: SGD parity with the accepted scorer

Run from the repository root with the project venv (Python 3.12.12,
NumPy 2.5.3; `PYTHONDONTWRITEBYTECODE=1` optional):

```sh
python vendor/sutro-problems/mnist/submissions/grid-mlp-scoring-20260912/score.py \
  --features 9 --width 32 --epochs 300 --batch 25 --n-train 1000 --n-test 1000 \
  --learning-rate 0.2 --seed 101 --output /tmp/scorer-adam/vendor-sgd

python new-spec/small-campaign/scorer-adam/score.py \
  --features 9 --width 32 --epochs 300 --batch 25 --n-train 1000 --n-test 1000 \
  --learning-rate 0.2 --seed 101 --optimizer sgd --output /tmp/scorer-adam/new-sgd
```

Result: `program.spatial.json` is byte-identical, with file SHA256
`4be8f2fddfcbcb8b49dacc974793bd7bff4159ffdd0c9c3c0fee9bea60f843c0`
for both, and the required grid-score fields are identical:

| Field | Vendor scorer | New scorer (`--optimizer sgd`) |
| --- | ---: | ---: |
| `energy_mj` | 0.242990182992 | 0.242990182992 |
| `time_ms` | 3073.508456 | 3073.508456 |
| `energy_fj` | 242990182992 | 242990182992 |
| `cycles` | 3073508456 | 3073508456 |
| `peak_allocated_scratch_bytes` | 91116 | 91116 |

`components`, `logical_instructions`, `program_sha256` and
`program_file_sha256` also match. Only `source_sha256` differs, because these
are modified copies of the three Python files. `--optimizer sgd
--nr-iterations 4` was also checked: its program is byte-identical to the
vendor program, because the SGD path never uses the option.

## Adam prices

All runs use `--features 9 --batch 25 --n-train 1000 --n-test 1000
--learning-rate 0.2 --seed 101 --optimizer adam`.

Requested configurations, `energy_mj` / `time_ms`:

| Config | K = 4 | K = 8 | K = 12 (default) |
| --- | ---: | ---: | ---: |
| w32 e75 | 0.110944549888 / 1155.798521 | 0.136044949888 / 1350.798521 | 0.161145349888 / 1545.798521 |
| w32 e100 | 0.150263820462 / 1538.760521 | 0.185228620462 / 1798.760521 | 0.220193420462 / 2058.760521 |
| w36 e75 | 0.125707149734 / 1298.026077 | 0.154177149734 / 1517.026077 | 0.182647149734 / 1736.026077 |
| w36 e100 | 0.17026406257 / 1728.228077 | 0.20990598257 / 2020.228077 | 0.24954790257 / 2312.228077 |
| w40 e75 | 0.140553103532 / 1440.253633 | 0.172454143532 / 1683.253633 | 0.204355183532 / 1926.253633 |
| w48 e75 | 0.170379699884 / 1730.003529 | 0.209327139884 / 2021.003529 | 0.248274579884 / 2312.003529 |

`peak_allocated_scratch_bytes` is identical for all K at a fixed config
(the regions do not depend on K): 120352, 128352, 122112, 130112, 123872,
127392 for the six rows above.

Additional default-K (12) prices:

| Config | `energy_mj` | `time_ms` | `peak_allocated_scratch_bytes` |
| --- | ---: | ---: | ---: |
| w48 e100 | 0.34091276063 | 3079.691929 | 135392 |
| w64 e100 | 0.47418260711 | 4124.575913 | 142432 |
| w48 e150 | 0.453455730074 | 4615.068729 | 151392 |

Example command:

```sh
python new-spec/small-campaign/scorer-adam/score.py \
  --features 9 --width 48 --epochs 75 --batch 25 --n-train 1000 --n-test 1000 \
  --learning-rate 0.2 --seed 101 --optimizer adam --nr-iterations 8 \
  --output /tmp/scorer-adam/adam-w48-e75-k8
```

For reference, the previous version of this scorer (a new `sqrt` opcode and
no NR option) scored w48/e150 at 0.253993774296 mJ / 2863.219912 ms; the cost
of replacing that opcode with 37 charged Newton-Raphson instructions at K = 12
is the difference.

## Measured `vhat` range and seen-value error

Measured with the exact research `adam_step` rule (float32, exact `np.sqrt`)
on draw 20261001, tracking every parameter element at every step:

| Config | min positive `vhat` | max `vhat` | max rel NR error K=4 | K=8 | K=12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| w32 e75 | 1.542e-13 | 83.25 | 3.026e+01 | 1.121e+00 | 2.384e-07 |
| w32 e100 | 5.490e-14 | 83.25 | 3.026e+01 | 1.121e+00 | 2.384e-07 |
| w36 e75 | 1.880e-15 | 29.02 | 3.026e+01 | 1.121e+00 | 2.384e-07 |
| w36 e100 | 6.691e-16 | 29.02 | 3.026e+01 | 1.121e+00 | 2.384e-07 |
| w40 e75 | 9.976e-12 | 43.51 | 3.026e+01 | 1.121e+00 | 2.384e-07 |
| w48 e75 | 2.284e-09 | 82.25 | 3.026e+01 | 1.121e+00 | 2.384e-07 |

The maximum is attained near `vhat = 1e-6`, inside the validated superset, so
the table above is the same bound reported in the error table. Exact zeros
also occur (permanently dead ReLU units): `v = 0` requires every gradient for
that element to have been zero, which also forces `m = 0`, so the update is
`0 / (NR_sqrt(0) + eps) = 0` and the NR seed value at zero (2.441406e-10) has
no numerical effect.

## Additional local checks (throwaway, not checked in)

- Expanding `scheduled_accesses` for a tiny Adam model reproduces the scorer's
  `energy_fj` and `cycles` exactly (event-by-event sum equals the histogram
  sum), for both optimizers.
- A tiny Adam MLP (features 3, width 4, 4 train / 3 test, batch 2, 2 epochs)
  was executed instruction-by-instruction with the same executor, and its
  final `w1,b1,w2,b2` words, all `m`/`v` words and predictions are bitwise
  identical to an independent ordered NumPy learner that performs the
  identical Adam rule and the identical K-step Newton-Raphson sequence, for
  K = 4, 8 and 12. Over the 264 positive `vhat` values seen in that run
  (range [2.5e-04, 23.17]), NR vs `np.sqrt` had max relative error 3.023e+00
  (K=4), 1.713e-04 (K=8) and 1.156e-07 (K=12).
- The w48/e100 Adam executed-instruction deltas vs SGD match the analytic
  per-element counts exactly, for all tested K.

## Assumptions and scope

1. `t` is the global minibatch index starting at 1 (not the epoch index); this
   matches the `adam_step` variant in `new-spec/small-campaign/adam_confirm.py`.
2. Bias-correction divisors are computed per step from the float32 expressions
   `1 - f32(0.9)**t` and `1 - f32(0.999)**t` and stored as literals. Their
   bit patterns affect only program hashes, never the counts; costs are
   value-independent.
3. The `m` and `v` arrays are zero-initialized through the existing charged
   region-initialization loop, so their initialization is modeled work.
4. The Newton-Raphson seed is the fixed constant `y0 = x + 1e-6` and K is a
   fixed command-line constant; there is no data-dependent exit. K = 12 is the
   default and the smallest tested value meeting the 1e-6 relative error
   target; smaller K may still be used for a deliberate cheaper, lower-fidelity
   price, and `--nr-iterations` values below 1 are rejected.
5. `div` is an ISA v4 opcode; `sqrt` is not used anywhere. The scorer
   validates, prices and hashes the program but does not execute the
   arithmetic (`numeric_execution_checked_by_scorer` remains `false`); the
   Newton-Raphson accuracy statements come from the numerical experiment and
   the tiny bitwise check above.
6. Charged cost of the approximation: the three `z` temporaries, the six `k`
   constants and all `14 + 3*K` update instructions per element are fully
   charged by the same access-and-distance machinery as SGD. Adam prices
   therefore reflect the serialized cost of the approximation, not of a
   hardware sqrt. The `--nr-iterations` option never changes the SGD program
   or its price.
7. The change is scoped to this small campaign: only complete fixed-size
   minibatches, the same fixed tape layout, placement and serialized schedule
   are used. No Modal/GPU and no vendor files were modified.
