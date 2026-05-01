# Matrix Multiplication

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-01
**Problem:** 16×16 matmul
**Cost:** 70,053
**IR:** [`dead_input_outputs_packed_16x16.ir`](dead_input_outputs_packed_16x16.ir)
**Method:** `generate_dead_input_outputs_packed_16x16` (dead-input output reuse + B packing)

## Idea

Take the 73,602 base from the (Tio=4, Tjo=8, Tii=4, Tji=1) two-level
schedule (the closed-form optimum of the decoupled blocking family, see
[`closed_form.py`](closed_form.py)) and stack three liveness-aware tricks
on top.

## Path to 70,053

| step                                    | score  | savings |
|-----------------------------------------|-------:|--------:|
| (4, 8, 4, 1) base (closed-form optimum) | 73,602 |         |
| + redirect last-mul to addr 1           | 72,642 | 960     |
| + last-super-block outputs in sC        | 71,724 | 918     |
| + dead-input output storage + B packing | **70,053** | 1,671 |

### Step 1: Redirect (saves 960, score 72,642)

The single sB cell at addr 1 dies after each inner block's last mul.
Route that last mul's output back into addr 1 instead of TMP@2; the
following add then reads cost 1 instead of cost 2. Same instruction
count, one read cheaper per inner block.

```
mul TMP, sA(0..2), sB         # standard for ii=0,1,2
add sC, TMP

mul sB_addr, sA(3), sB_addr   # ii=3 redirect: write product into dying sB cell
add sC, sB_addr               # add reads addr 1 (cost 1) instead of TMP@2 (cost 2)
```

Per-redirect saving = `cost(2) - cost(1) = 1`. Count = `n²(n-1)/(Tio_o × Tji) = 960`.
See [`redirect_16x16.ir`](redirect_16x16.ir).

### Step 2: Outputs in sC (saves 918, score 71,724)

The very last super-block computes 32 sC values then bulk-copies them
to C_bulk@775..806 (cost about 28 each), where EXIT re-reads them.
Pointing those 32 output addresses directly at sC@7..38 saves:

- **162** from skipping 32 copy-out instructions (each was a read of sC).
- **756** from 32 EXIT reads now at avg cost 5 instead of avg cost 28.

The other 224 outputs remap into C_bulk@551..774. See
[`sc_outputs_16x16.ir`](sc_outputs_16x16.ir).

### Step 3: Dead-input output storage with B packing (saves 1,671, score 70,053)

Output addresses may alias input addresses once the input cell has had
its final read. In row-major super-block order:

- `(0,0)` is the only non-final block with no dead inputs available, so
  its 32 outputs still need fresh spill cells.
- After each row super-block's second column block, that 4×16 A row
  slab is dead and can hold later outputs.
- After `(3,0)`, the 32 B cells for columns 0..7 are dead and can hold
  that block's outputs.
- `(3,1)` keeps the outputs-in-sC trick from Step 2.

This replaces 192 fresh C-spill cells (cost 24+ each) with reused A/B
addresses (cost up to 21):

| storage         | outputs |
|-----------------|--------:|
| fresh C spill   |    32   |
| dead A inputs   |   160   |
| dead B inputs   |    32   |
| sC (last block) |    32   |

Packing the 32 B cells that later double as outputs contiguously at
the front of the B region (they have 5 reads vs 4 for ordinary B
cells) gives the final **70,053**.

## Cost-model recap

- Address `a` costs `ceil(sqrt(a))`. Each instruction has 2 reads,
  charged separately.
- Inputs placed at user-chosen addresses (free placement). Outputs
  read once at exit.
- ISA: `add`/`sub`/`mul`/`copy` only (no FMA).

## Region cost breakdown (at 71,724, just before Step 3)

| region              | cost   | reads | cells | reads/cell |
|---------------------|-------:|------:|------:|-----------:|
| sB+redirect@1       | 5,056  | 5,056 | 1     | 5056       |
| TMP@2               | 5,760  | 2,880 | 1     | 2880       |
| sA@3..6             | 10,240 | 4,096 | 4     | 1024       |
| sC@7..38            | 20,574 | 4,064 | 32    | 127        |
| B_bulk@39..294      | 13,328 | 1,024 | 256   | 4          |
| A_bulk@295..550     | 10,738 | 512   | 256   | 2          |
| C_bulk@551..774     | 5,866  | 224   | 224   | 1          |
| sC EXIT (32 outputs)| 162    | 32    | n/a   | n/a        |

## Why ~70K is roughly the floor

Mandatory read events (n=16):

- mul reads ≥ 2·n³ = 8,192 (each mul reads sA + sB).
- add reads ≥ 2·(n³ - n²) = 7,680.
- exit reads ≥ n² = 256.
- bulk-load reads ≥ 256 + 256 = 512 (each input read once).

Total: about 16,640 read events. If all at cost 1 the floor is
16,640. Realistic average cost 3 to 5 per read gives a ~50K
theoretical floor. The gap from 50K to 70K is the reload tax (B×4,
A×2 in our scheme) and sC at avg cost 5.

## Floor verification (7-direction follow-up)

After landing 70,053, a second-pass search across seven independent
directions tested whether the floor was real. None beat it.

| direction                                 | best    | delta vs 70,053 |
|-------------------------------------------|--------:|----------------:|
| Time-expanded liveness allocator          | 70,053  | 0 (byte-identical IR) |
| Hungarian/CP-SAT assignment over schedule | 70,053  | 0 (proved optimal, see below) |
| Mixed block sizes / sacrifice-reload      | 70,565  | +512            |
| Asymmetric reload patterns (k-outer etc.) | 71,723  | +1,670          |
| Rolling sC (4×4 / 2×8 / 4×2 stripes)      | 76,607  | +6,554          |
| Larger scratchpad (hold full B-col)       | 78,026  | +7,973          |
| Strassen-outer 4×4 (49 sub-products)      | 184,576 | +114,523        |

Two strong negative results:

- **Address layout is provably optimal** for this schedule. The
  read-count-weighted bound `Σ reads_i · cost(addr_i)` is minimised
  by sorting reads desc against costs asc, and that minimum equals
  exactly the 28,261 bulk cost the greedy layout already achieves.
- **The 32 unavoidable C-spills are a conservation law**, not a
  search artifact. Block (0,0) has no dead inputs available because
  it is first; sacrificing reloads only shifts when the spill happens,
  it does not reduce it.

Reload count is the binding constraint everywhere. Strassen-outer
adds about 1K intermediate cells that push A/B/C bulk to addrs 1,127
to 1,894 (cost 35 to 43); arithmetic savings of about 2K cannot pay
the ~50K address-pressure tax.

## Things that don't work in this cost model

| idea                                            | result                                              |
|-------------------------------------------------|-----------------------------------------------------|
| Pair-tree summation (k pairs)                   | 74,160. Extra hot cells displace sC by more than the sC savings. |
| Quad-tree summation                             | 75,148. sA cache balloons from 4 to 16 cells.       |
| Multi-redirect within a cycle                   | strictly worse. Restore copies cost 2 vs 1 saved.   |
| Routing copy-outs through addr 1                | strictly worse. Adds 1 read at addr 1.              |
| TMP at addr 1, sB at addr 2                     | +256 worse. sB has more reads, belongs at addr 1.   |
| 3-op `add` for non-init pairs                   | shifts reads but doesn't reduce them.               |
| Strassen recursion for n=16                     | 15,271 ops vs 7,936 naive. add/sub overhead dominates at small n. |
| Full B in scratchpad                            | 95,784. Forces 256-cell sC, dominates layout.       |
| Outputs at low addrs (greedy violation)         | +5,985 worse. B has higher reads/cell, must be lower. |
| Extending outputs-in-sC to sb 6 too             | needs 32 cells between sC and B_bulk; B_bulk shift (+1,308) > savings (~660). |
| Super-block reordering after dead-input storage | tied. Brute force over all 8! orders: 32 spill cells unavoidable. |

## Open directions if pushing further

1. **Algebraic identity exploiting the test data structure** (A and B
   both rank-2 in our test inputs). Fully overfit closed form
   possible but considered cheating.
2. **Schedule with single sA, single sB, and cleverer reuse**. Every
   attempt found so far either reloads B 16× (kills it) or adds hot
   cells that displace sC.
