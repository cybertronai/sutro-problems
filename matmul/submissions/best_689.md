# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)  
**Date:** 2026-09-01  
**Problem:** 4×4 matmul  
**Cost:** 689  
**IR:** [`best_689.ir`](best_689.ir)  
**Verifier:** [`best_689.py`](best_689.py)  
**Exact allocation certificate:** [`best_689_certificate.json`](best_689_certificate.json)

## Summary

This submission computes ordinary 4×4 matrix multiplication with the classical
64 multiplications and 48 additions, but changes the dataflow and physical
layout. It scores **689**, improving the previous 800 record by **111 read-cost
units (13.875%)**.

The schedule processes one output row at a time. Address 1 is the hottest
scratch cell. The first product of each dot product is written directly into
its accumulator, later products are formed in short-lived low-address cells,
and final additions overwrite dead input-A cells where possible. B is not
copied into a separate four-cell staging buffer; instead, its sixteen input
values are placed directly at addresses chosen for their four reads each.

Ten of the sixteen final outputs reuse addresses initially occupied by A.
The whole physical program uses addresses 1 through 38 and only 13 copy
instructions.

## Logical algorithm

For each output row `i`:

```text
acc[j] = A[i,0] * B[0,j]                 for j = 0..3
for k = 1..3:
    place A[i,k] in the hot scratch cell when profitable
    p      = A[i,k] * B[k,j]             for j = 0..3
    acc[j] = acc[j] + p
write/finalize acc[j] into dead A cells or released scratch cells
```

The arithmetic is therefore not a Strassen-style rank reduction. The gain is
from schedule/lifetime optimization:

1. eliminate the previous solution's four-cell B staging buffer;
2. avoid a separate initialization/copy for every accumulator;
3. overwrite A values immediately after their last multiplication;
4. reuse ten dead A addresses as output addresses;
5. allocate the resulting live intervals jointly rather than sorting only by
   aggregate read frequency.

## Input placement

The first 16 declared addresses contain A in row-major order; the next 16
contain B in row-major order.

| Value | Addresses |
|---|---|
| `A[0,:]` | `1, 2, 3, 26` |
| `A[1,:]` | `17, 18, 27, 28` |
| `A[2,:]` | `19, 37, 29, 30` |
| `A[3,:]` | `31, 32, 33, 38` |
| `B[0,:]` | `5, 6, 10, 20` |
| `B[1,:]` | `7, 21, 11, 22` |
| `B[2,:]` | `23, 12, 13, 24` |
| `B[3,:]` | `14, 25, 15, 16` |

Final outputs, in row-major order, are:

```text
26,34,35,36,27,17,28,18,29,9,8,19,2,3,4,1
```

## Cost breakdown

| Instruction | Count | Paid reads | Read cost |
|---|---:|---:|---:|
| `mul` | 64 | 128 | 340 |
| `add` | 48 | 96 | 202 |
| `copy` | 13 | 13 | 77 |
| output exit | — | 16 | 70 |
| **total** | **125** | **253** | **689** |

The hottest physical addresses are:

| Address | Reads | Cost/read | Contribution |
|---:|---:|---:|---:|
| 1 | 75 | 1 | 75 |
| 4 | 27 | 2 | 54 |
| 2 | 19 | 2 | 38 |
| 3 | 18 | 2 | 36 |
| 8 | 10 | 3 | 30 |
| 9 | 10 | 3 | 30 |

## Correctness

The repository scorer checks every output symbolically over arbitrary A and B,
not against one numerical sample. The IR produces exactly

```text
C[i,j] = Σ(k=0..3) A[i,k] * B[k,j]
```

for all sixteen outputs, and no intermediate has degree above two.

Run from the repository root:

```bash
python3 matmul/submissions/best_689.py
```

Expected output:

```text
best_689.ir: score=689, sha256=280f3a566c21858a37cc0642abff65857853f52545b656ec2bea72ef62d5122c
operations: {'add': 48, 'copy': 13, 'mul': 64}
read costs: {'add': 202, 'copy': 77, 'mul': 340, 'output': 70}
```

A record-locking pytest can be added to `matmul/test_matmul.py`:

```python
def test_best_689_cost_matches_record_history():
    from matmul.submissions.best_689 import generate_best_689

    assert matmul.score_4x4(generate_best_689()) == 689
```

## Fixed-schedule optimality certificate

Treat every value written by the program as an SSA value with a live interval.
A physical tier `t` contains `2t-1` addresses and charges `t` for every read.
The address-tier assignment LP is

```text
minimize  Σ(value v, tier t) reads(v) * t * x[v,t]
subject to
          Σ(t) x[v,t] = 1                                  for every value v
          Σ(v live at time q) x[v,t] <= 2t-1               for every t,q
          x[v,t] >= 0
```

The included integer dual assignment is feasible and has objective 689. Since
the submitted physical allocation also costs 689, weak duality proves that
**no address assignment can improve this exact operation schedule below 689**.
This is not a global lower bound over different schedules or arithmetic
circuits.

Verify the exact certificate independently:

```bash
python3 matmul/submissions/verify_best_689.py \
  matmul/submissions/best_689.ir \
  matmul/submissions/best_689_certificate.json
```
