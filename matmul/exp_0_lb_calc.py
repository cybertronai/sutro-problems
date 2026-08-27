"""
Lane 0: Lower-bound analysis for 16x16 matmul under Dally cost.

We compute the minimum cost of any read multiset subject to combinatorial
constraints, allowing the cells to be assigned the cheapest addresses 1,2,3,...
in the optimal-permutation way.

Key facts:
- Cost of a cell at "rank" k (1-indexed) is ceil(sqrt(k)).
- For an algorithm: a *read multiset* M = {(r_i, n_i)} where n_i cells get r_i
  reads each. Optimal cost = sum over a sorted order of (r * cost(rank)).
  When we pack cells with the most reads into addrs 1..n, the cost is
  determined entirely by the sorted-descending read sequence.

We want: minimum over all FEASIBLE (algorithm-realizable) read multisets, of
the optimal-permutation-cost.

Feasibility constraints we'll encode:
  C1. The exit reads at least 256 distinct cells (the C output values), each
      at least once.
  C2. Each of 256 distinct A values must be read at least once (or aliased
      with output / scratch but at least one read).
  C3. Each of 256 distinct B values must be read at least once.
  C4. The number of binary ops >= some lower bound depending on algorithm
      class. For naive/non-Strassen, mul count = 4096, add/sub count = 3840
      (16*16*15 to combine).
  C5. Each operation reads 2 cells.

We can ALIAS A with output C (the breakthrough). So strictly we have 256 A
cells + 256 B cells + 256 derived C cells = 768 logically-distinct cells, but
A and C can share addresses (an A cell becomes C cell after all uses of A
are done).

Let me build a model that:
- Takes a number of total ops O and counts how many distinct cells N must
  exist, with a per-cell read count distribution.
- Computes optimal cost = sum over ranks 1..N of (read_count_at_rank *
  ceil(sqrt(rank))).

Then we'll search over plausible (O, N, distribution) parameters to find
LB(plausible algos).
"""

import math
import sys
from collections import Counter

sys.path.insert(0, '/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/SutroAna')
from lower_bound import count_reads, compute_cost, optimal_remap


def cost_of_multiset(reads_per_cell):
    """Given a list of read counts per cell, return optimal-permutation cost.
    Sort descending, assign to addrs 1, 2, 3, ...
    """
    sorted_reads = sorted(reads_per_cell, reverse=True)
    total = 0
    for rank, r in enumerate(sorted_reads, start=1):
        total += r * (math.isqrt(rank - 1) + 1)
    return total


def cost_of_multiset_compact(multiset):
    """Multiset = list of (read_count, num_cells) pairs."""
    expanded = []
    for cnt, n in multiset:
        expanded.extend([cnt] * n)
    return cost_of_multiset(expanded)


def main():
    # Verify against the record
    ir = open('records/record_69697_lane0.ir').read()
    reads = count_reads(ir)
    reads_list = list(reads.values())
    print(f"Record cost: {cost_of_multiset(reads_list):,}")
    print(f"Record cell count: {len(reads_list)}")
    print(f"Record total reads: {sum(reads_list)}")
    print()

    # Multiset from the record
    multi = Counter(reads_list)
    print("Multiset:")
    for r, n in sorted(multi.items(), reverse=True):
        print(f"  {r} reads x {n} cells = {r*n} reads")
    print()

    # ========================================================================
    # Lower bound 1: combinatorial counting (no algorithm class assumed)
    # ========================================================================
    # Naive: 4096 mults + 4080 adds = 8176 binary ops + 256 exit reads
    # Each binary op reads 2 cells -> 16,352 reads + 256 exit reads = 16,608.
    #
    # Now: minimum-cost layout for 16,608 reads packed into N cells.
    # If we pack ALL reads into 1 cell: cost = 16,608 * 1 = 16,608.
    # But that's infeasible: we need 256 distinct A, 256 distinct B, 256
    # distinct C (or 512 if A aliased with C).
    #
    # MINIMUM cells = 256 (B) + 256 (A=C aliased) = 512.
    # Plus a scratchpad cell for the running sum, but with aliasing tricks
    # we can reuse very few extra cells.
    #
    # If 16,608 reads land on 512 cells with read count concentrated:
    #   The most-used cells are scratchpad/accumulators. With clever
    #   programming maybe we can have:
    #   - 256 B cells, each read 16 times (used once per row of A) = 4096
    #   - 256 A=C cells, each read 16 times (used in inner loop) + 1 exit = 17 = 4352
    #   - 0 extra scratchpad
    #   Total reads = 8448. But we computed 16,608 above as minimum from binary
    #   op count -- inconsistency. Let me recheck.
    #
    # Op count: 4096 muls + 4080 adds ALL BINARY = 8176 ops, 2 reads each
    #   = 16,352 reads JUST from arithmetic. Plus 256 exit. Total 16,608.
    # If we must do 4080 adds, the second operand of each add is a unique
    # product (or at least different cells) -- you can't compute c[i,j] in
    # fewer than 16 mults + 15 adds with just add/sub/mul/copy.
    #
    # So the read distribution MUST sum to >= 16,608.
    #
    # And we need >= 512 distinct cells.

    # ========================================================================
    # Optimal layout for varying total reads / cell count combinations
    # ========================================================================

    print("=" * 60)
    print("LB 1: Pack 16,608 reads into N cells, all reads on cell #1")
    print("(absolute minimum, ignoring distribution constraints)")
    print("=" * 60)
    # Place all 16,608 on addr 1 -> cost 16,608. But that requires 16,608
    # reads from a single cell, which is impossible since each binary op
    # reads exactly 2 distinct cells (well, they CAN be same cell, e.g.,
    # x = x + x, but the operation only counts as 2 reads of one cell).
    print(f"All 16,608 on cell 1: {16608 * 1:,}  (infeasible)")

    print()
    print("=" * 60)
    print("LB 2: Constraint satisfaction")
    print("=" * 60)
    # Each B cell read >= 1 time -> 256 reads on B cells.
    # Each A cell read >= 1 time -> 256 reads on A cells (or aliased to C).
    # Each C output cell read >= 1 time at exit -> 256 reads on C cells.
    # 256 + 256 + 256 = 768 reads MUST be spread across >= 768 cells if no
    # aliasing, or as few as 512 if A=C aliasing.
    #
    # But we have 16,608 total reads. So 16,608 - 768 = 15,840 extra reads
    # need to be distributed.
    #
    # The optimal assignment: pack as many of those 15,840 onto a single
    # super-hot cell. But which kind of cell can have huge read count?
    #
    # In the record: cell 1 = scratchpad gets 4,096 reads (read for each
    # of 4096 mac operations). That's the 4096-read cell.
    # The next-hottest is the 'inner accumulator' at 3840 reads.

    # The question: what is the MAX possible read count on a single cell?
    # A single cell can be re-read indefinitely. Each binary op reads 2 cells
    # (the operands). If both operands are the same cell, it's 2 reads of one
    # cell (e.g. x = x + x).
    #
    # An add c = a + c reads c once and a once. If c is the scratchpad and a
    # cycles through 4096 different values, c gets read 4096 times in 4096
    # adds. Plus 4096 writes. That's the max if we have 4096 ops contributing
    # to one accumulator.
    #
    # But each c[i,j] is a separate accumulator. So 256 accumulators, each
    # gets ~16 reads as accumulator + 1 final exit = 17 reads each = 4352.
    #
    # Unless we do all 4096 muls into ONE scratchpad and copy out:
    #   for i,j: scratch=0; for k: scratch += A[i,k]*B[k,j]; copy C[i,j], scratch
    # Each iteration: mul tmp, A, B (2 reads); add scratch, tmp (2 reads of
    # scratch + tmp). So scratch read 2 per inner step? No, "add scratch, tmp"
    # is "scratch = scratch + tmp" -> reads scratch once, reads tmp once.
    # 4096 such adds -> scratch gets 4096 reads.
    # tmp gets 4096 reads.
    # PLUS 256 copies "C[i,j] = scratch" -> 256 more scratch reads = 4352.
    # Plus 256 exit reads on each C[i,j] = 256.
    # Plus 4096 muls: A read 4096 times (256 A cells, each 16 times),
    # B read 4096 times (256 B cells, each 16 times).
    # Adds count: 4096 (one per mul) -> 4096 adds (different from naive 4080
    # because we're adding into a scratch zero-initialized).

    # Actually: naive is 16 muls + 15 adds per output cell = 256 * 31 = 7936 ops.
    # 16 * 256 = 4096 muls; 15 * 256 = 3840 adds = 7936 binary ops.
    #
    # If we use scratch = 0; scratch += A*B 16 times, then copy out:
    #   16 muls + 16 adds per output (one extra add for the zero init).
    #   Or: 16 muls + 15 adds (skip first add, just copy first product to
    #   scratch). Either way, ~7936 binary ops.
    #
    # Total reads = 7936 * 2 + 256 (exit) = 16,128.
    #
    # WAIT. The brief said 16,608. The record has 17,664.
    # Let me recount: 4096 muls (8192 reads) + 3840 adds (7680 reads) + 256 exits
    # = 8192 + 7680 + 256 = 16,128.
    #
    # If we add 256 init copies (scratch=0), each copy = 1 read = 256 reads.
    # Total = 16,384.

    # OK, the absolute minimum reads is around 16,128 for naive matmul.

    print(f"Min reads (naive): 4096 muls + 3840 adds + 256 exit = {4096*2 + 3840*2 + 256:,}")
    print(f"With aliased scratchpad (no extra copies): {4096*2 + 3840*2 + 256:,}")

    # ========================================================================
    # LB 3: Cell-count + read-count joint optimization
    # ========================================================================

    print()
    print("=" * 60)
    print("LB 3: Pack reads optimally")
    print("=" * 60)

    # Suppose we have N = 512 cells (256 A, 256 B, A aliased with C).
    # Plus 1 scratchpad cell.
    #
    # Read distribution:
    #  - scratchpad: ? reads
    #  - 256 A=C cells: each read at least 17 times (16 muls + 1 exit).
    #     Actually if scratchpad approach: 16 muls per (i,j) but A[i,k] is read
    #     for each j! So A[i,k] read 16 times (once per j). So A cells: 16
    #     reads each. Plus 1 exit (since A=C aliased) = 17.
    #     256 * 17 = 4352 reads.
    #  - 256 B cells: each read 16 times (once per i). 256 * 16 = 4096.
    #  - scratch: 3840 reads (as accumulator) + 4096 (in muls? no muls store to
    #     a tmp). Actually with the alias trick: directly add into scratch.
    #     Scratchpad: 4096 read-as-accum + 4096 read-as-mul-target? No, mul
    #     reads its 2 sources. The scratchpad reads as accumulator in the add.
    #     Per (i,j): 16 muls = 16 reads of A + 16 reads of B. Then 16 adds:
    #     16 reads of mul_result + 16 reads of accum.
    #     OR we use a 2-operand 'add accum, mul_result' which reads accum and
    #     mul_result.
    #     Total scratch reads (across all i,j): 16 * 256 = 4096 reads as accum.
    #     Plus 1 read to copy into C[i,j] = 256 reads. = 4352.
    #
    #     Plus there's a 'mul_result' temp cell. Each mul writes to it, each
    #     add reads it. 4096 muls -> 4096 mul-result cells? Or we reuse: a
    #     single 'tmp' cell gets written 4096 times and read 4096 times.
    #
    # So multiset = {scratch: 4352, tmp: 4096, A=C bulk: 17 each (256), B bulk: 16 each (256)}
    # Total reads = 4352 + 4096 + 256*17 + 256*16 = 4352 + 4096 + 4352 + 4096 = 16,896.

    # Optimal layout cost:
    layout = [4352, 4096] + [17]*256 + [16]*256
    cost_var1 = cost_of_multiset(layout)
    print(f"\nVariant 1: {len(layout)} cells, {sum(layout):,} reads, cost {cost_var1:,}")

    # Now what if we have a SECOND-LEVEL accumulator (the inner-loop trick that
    # the record uses)?
    # The record has TWO super-hot cells: 4096 reads and 3840 reads.
    # That's because it processes the 4x4 sub-problem with a different layout.

    # Variant 2: the record's actual structure
    # 4096-cell + 3840-cell + 4 cells at 1024 + 32 cells at 120 + ...
    layout2 = [4096, 3840] + [1024]*4 + [120]*32 + [5]*160 + [4]*96 + [2]*256 + [1]*96
    cost_var2 = cost_of_multiset(layout2)
    print(f"Variant 2 (record): {len(layout2)} cells, {sum(layout2):,} reads, cost {cost_var2:,}")

    # ========================================================================
    # LB 4: Globally minimum schedule with naive op count
    # ========================================================================

    # If we MUST have at least 16,128 reads (naive count) and 512 distinct
    # cells (256 A=C + 256 B), can we get below 69,697?
    #
    # Lemma: the minimum cost for placing R reads onto N cells (no per-cell
    # constraint) is achieved by packing ALL reads onto the cheapest cell.
    # But we have a per-cell minimum: each B cell >= 1 read, each A cell >= 1
    # read, each C cell >= 1 read (exit).
    #
    # Truly minimum: 1 read per A,B,C cell = 256+256+256 = 768 reads on 768
    # cells. But aliasing: 256 + 256 = 512 cells with 256+256+256=768 reads
    # spread as: 256 cells with 1 read each (B) + 256 cells with 2 reads each
    # (A=C: 1 use as A, 1 exit).
    # But this doesn't account for any actual computation.

    # The REAL constraint is: all 4096 distinct mult products A[i,k]*B[k,j]
    # must be computed. Each mult needs A[i,k] and B[k,j] available somewhere.
    # That doesn't pin down read counts though, since one A read can feed
    # multiple muls only if it's still in the cell. Actually wait: a "read"
    # in this model = a load operation. Each `mul x,y,z` reads y and z. To
    # do 4096 muls, we need 8192 read operations, each from some cell.
    #
    # If all 16 j-values for row i use A[i,k] (which they must), then either:
    #   (a) A[i,k] is read 16 times from its cell, OR
    #   (b) A[i,k] is copied to another cell which is read fewer times.
    #
    # Option (b): copy A[i,k] to scratch (1 read), then read scratch 16 times.
    # Net: 17 reads of various cells (1 of A[i,k] + 16 of scratch).
    # Vs option (a): 16 reads of A[i,k].
    #
    # If A[i,k] is at low addr, option (a) is better: 16 reads at low cost.
    # If A[i,k] is at high addr, option (b) is better: 1 read at high cost +
    # 16 at low.
    #
    # The record uses option (b) for the bulk! That's the reason for the
    # 4096-reads scratchpad cell: A bulk is at addrs >=39, expensive. So we
    # copy each A cell once into addr 1 (cost = 1 read at high addr) then
    # read scratch (addr 1) 16 times.
    #
    # Similarly the 3840 cell is the second-tier accumulator for inner loops.

    # ========================================================================
    # LB 5: SHRINKAGE -- can we move more reads to scratch?
    # ========================================================================

    # The most reads possible on cell 1 = total reads / 2 (since each binary
    # op reads 2 distinct positions, but they CAN both be the same address
    # like x=x+x, only 1 distinct read but 2 operand reads).
    #
    # Actually it's possible to have ALL reads on a single cell if the
    # operation is x = x + x (or similar self-ops). But that's degenerate
    # and doesn't compute matmul.
    #
    # Realistically, scratch can be one operand and a different cell the other.
    # So at most ~1/2 of reads can be scratch reads.
    # Total reads ~16,128. Scratch reads <= 8,064. We have 4096 in record.
    #
    # Can we double scratch usage? If we did, costs would be:
    #   scratch (addr 1): 8064 reads
    #   remaining 8064 reads spread across at least 511 cells
    #   At minimum: 16+ reads per A=C cell, 16 per B cell.

    # Let's compute: if scratch has 8064 reads and there are 256 A=C cells and
    # 256 B cells with 8064/512 = 15.75 reads each on average:
    layout3 = [8064] + [16]*512  # 8064 + 8192 = 16256 total
    cost_var3 = cost_of_multiset(layout3)
    print(f"\nVariant 3 (max scratch): {len(layout3)} cells, {sum(layout3):,} reads, cost {cost_var3:,}")

    # Wait, that's BETTER than 69,697! Let me sanity-check.

    # The issue: we can't actually achieve "8064 scratch reads + 16 reads each"
    # without violating arithmetic. Let me think more carefully.

    # Each binary op = 2 reads. Suppose op count is 8000. Total reads = 16,000.
    # If half are on cell 1 (scratch), 4000 are on other cells.
    # Other cells: 256 A=C + 256 B = 512 cells. 4000 reads / 512 = ~8 reads/cell.
    #
    # But each A[i,k] must be USED to compute something with all 16 j values.
    # If A[i,k] is in scratch each time, that's 16 distinct copies.
    #
    # Wait. If A[i,k] is copied to scratch (1 A-read), THEN scratch is used
    # as operand, A[i,k] gets only 1 read. This is the fundamental optimization.

    # With this: each A cell needs only 1 read (the copy to scratch).
    # Each B cell? Same idea. But B is per-column, not per-row. We need B[k,j]
    # for each i. So if we cycle through k,j and do 16 i values... hmm, need
    # to think about loop ordering.

    # For one row of C[i,*]: 16*16 = 256 muls. Each A[i,k] used 16 times
    # (one per j). Each B[k,j] used 16 times across rows i, but for THIS row
    # only once per k,j. Wait no, for THIS row, A[i,k] is constant across j,
    # so B[k,j] is read once per j-loop iteration through k.

    # This is getting complex. Let me just try the empirical layout.

    # Variant 4: Per the record but with all A=C reads pushed to scratch
    # Hypothetical: A=C cells get 1 read (copy to scratch), B cells get 1 read.
    # Scratch1 (for A): each A copied -> 16 muls -> 16 reads. Across 256 A
    #   cells: 256 copies (256 scratch reads/writes? no, the COPY writes
    #   scratch, then 16 muls READ scratch). Per A cell: 16 reads of scratch.
    #   Total: 16*256 = 4096 reads on scratch_A.
    # Scratch_B: same, 4096 reads.
    # mul_result temp: 4096 muls write to it, 4096 adds read it.
    # accumulator: 4096 reads.
    # A cells: 256 reads (one each, copied).
    # B cells: 256 reads.
    # C cells (aliased w/A or not): 256 exit reads.
    #
    # But wait: each A=C cell has both 1 A-read and 1 C-exit read = 2 reads.

    # Total layout:
    #   scratch_A: 4096
    #   scratch_B: 4096
    #   mul_result: 4096
    #   accum: 4096
    #   A=C cells: 256 cells x 2 reads each = 512 reads
    #   B cells: 256 cells x 1 read each = 256 reads

    layout4 = [4096, 4096, 4096, 4096] + [2]*256 + [1]*256
    cost_var4 = cost_of_multiset(layout4)
    print(f"\nVariant 4 (4 hot scratchpads): {len(layout4)} cells, {sum(layout4):,} reads, cost {cost_var4:,}")

    # Hmm that's also better. But is it FEASIBLE? Let me think.
    # The flow per (i,j):
    #   acc = 0  (or skip, just first mul writes acc)
    #   for k in 0..15:
    #     copy scratch_A, A[i,k]   (1 read of A[i,k], write scratch_A)
    #     -- but we need scratch_A for ALL j in this i, so we'd copy 16 times if loop is j-inner
    #     Actually loop ordering matters. Let's pick i-outer, k-middle, j-inner:
    #       for i:
    #         for k:
    #           copy scratch_A, A[i,k]  -- 1 read
    #           for j:
    #             copy scratch_B, B[k,j]  -- 1 read
    #             mul tmp, scratch_A, scratch_B  -- 2 reads (scratch_A, scratch_B)
    #             add C[i,j], tmp  -- 2 reads (C[i,j], tmp)
    #
    # Reads per inner iteration (k inner, j inner):
    #   - copy scratch_A from A[i,k]: 1 read of A[i,k]
    #   - copy scratch_B from B[k,j]: 1 read of B[k,j]
    #   - mul: 1 read of scratch_A, 1 read of scratch_B
    #   - add: 1 read of C[i,j], 1 read of tmp
    #
    # Per-loop: copy scratch_A 1x, copy scratch_B 16x (per j), 16 muls, 16 adds.
    # Per i,k: 1 copy A + 16 copies B + 16 muls + 16 adds = 1 + 16 + 32 + 32 = 81 reads.
    # Per i: 16 * 81 = 1296 reads.
    # Total: 16 * 1296 = 20,736 reads.

    # Hmm, that's more than 16,128. The issue: we copy B[k,j] every (i,k,j),
    # which is 16x for each B[k,j] across i values. So B reads = 4096 (same
    # as muls), not 256.

    # So Variant 4 is infeasible as written. Let me reconsider.

    # KEY INSIGHT: every binary op reads 2 cells. If we want to minimize total
    # reads, we use scratch-style. Let me characterize the actual minimum.

    # ========================================================================
    # LB 6: Counting argument via opcode flow
    # ========================================================================

    # Each of 4096 mults needs both operands.
    # If A[i,k] is used in 16 mults (one per j), we have two options:
    #   (i) read A[i,k] directly 16 times
    #   (ii) read A[i,k] once into scratch, then read scratch 16 times
    #   (iii) sequence of moves through several cells
    # In option (ii), total reads = 1 + 16 = 17.
    # In option (i), total reads = 16.
    #
    # SO option (i) has fewer reads but higher per-read cost (A is at high
    # addr). Option (ii) trades extra reads for cheap reads on scratch.
    #
    # For B[k,j]: similarly read 16 times (one per i).
    #
    # For accumulator C[i,j]: built via 16 adds, so read 16 times as add operand.
    #
    # MINIMUM read count if we use option (i) everywhere:
    #   A reads = 256 cells * 16 = 4096
    #   B reads = 256 cells * 16 = 4096
    #   mul-result reads (used in adds) = 4096 (one per mul)
    #   accum reads = 16 per cell * 256 = 4096
    #   exit reads = 256
    #   Total = 4096*4 + 256 = 16,640
    #
    # But this leaves 4096 cells for mul_results unless they're reused.
    # If a single mul_result cell is reused (overwritten each iteration):
    #   mul_result has 4096 writes (reuse) and 4096 reads.
    #
    # Layout: 256 A + 256 B + 256 C + 1 mul_result_temp = 769 cells.
    # Read distribution:
    #   1 cell at 4096 reads (mul_result temp)
    #   256 cells at 16 reads (A): each read 16 times in muls
    #   256 cells at 16 reads (B)
    #   256 cells at 17 reads (C): 16 adds + 1 exit
    #
    # Optimal layout cost:
    layout5 = [4096] + [17]*256 + [16]*512
    cost_var5 = cost_of_multiset(layout5)
    print(f"\nVariant 5 (option i, 1 hot temp): {len(layout5)} cells, {sum(layout5):,} reads, cost {cost_var5:,}")

    # Now with ONE scratch cell to hoist A[i,k] reads:
    # Per (i,k) instead of 16 A reads, do 1 A read + 16 scratch reads.
    # Across all (i,k): 256 A reads + 4096 scratch_A reads = 4352 reads (vs 4096 before).
    # MORE total reads but scratch is on addr 1 (cost 1).
    #
    # New layout:
    #   1 cell at 4096 reads (mul_result)
    #   1 cell at 4096 reads (scratch_A)
    #   256 A cells at 1 read each
    #   256 B cells at 16 reads each
    #   256 C cells at 17 reads each
    #
    # OR: A=C aliased:
    #   1 cell at 4096 (mul_result)
    #   1 cell at 4096 (scratch_A)
    #   256 A=C cells at (1+1) = 2 reads each (1 for copy to scratch, 1 for exit)
    #   256 B cells at 16 reads each

    layout6 = [4096, 4096] + [16]*256 + [2]*256
    cost_var6 = cost_of_multiset(layout6)
    print(f"\nVariant 6 (scratch_A + mul_temp + alias): {len(layout6)} cells, {sum(layout6):,} reads, cost {cost_var6:,}")

    # Hoist B too:
    # Per (k,j): copy B[k,j] to scratch_B, then 16 i-iterations use scratch_B.
    # Per (k,j) we get 1 + 16 = 17 reads (same as A).
    #
    # But loop ordering must allow it. If we do (k,j) outer and i inner,
    # then for each (k,j) we iterate i. But we ALSO need A[i,k] for each i.
    # So: outer k, j; for each (k,j), copy B[k,j] to scratch_B; for each i,
    # need A[i,k]. A[i,k] depends on (i,k), and i changes inside while k is fixed.
    #
    # If we hoist A[i,k] for each (i,k), the i-loop is INSIDE k. If j is in
    # the outermost or middle loop, A[i,k] copy happens during inner iter.
    #
    # Hmm, hoisting BOTH A and B simultaneously is tricky because A varies
    # with i,k and B with k,j. There's no shared inner loop.
    #
    # Concrete: outer k, then j, then i.
    #   For each k: 16 j iterations.
    #     For each j: copy scratch_B, B[k,j]  (1 read of B[k,j])
    #       For each i: need A[i,k] and accumulate into C[i,j]
    #         If A[i,k] is in low cell: 1 read of A[i,k] in mul.
    #         OR copy to scratch_A: but we'd copy 16 times per (k,j), so
    #         16*16 = 256 copies per k = 4096 copies total. Each followed
    #         by 1 use. Not a win.
    #       Better: A[i,k] used 16 times across j (for fixed i,k). If j-loop
    #       is inner, A[i,k] stays in scratch.

    # The classic hoisting: outermost k, middle i, inner j.
    #   for k: for i: copy scratch_A, A[i,k]; for j: ...
    # Then scratch_A is reused for 16 j-iterations. A[i,k] read once.
    #
    # In this loop order, scratch_B for B[k,j] would be re-copied for each i.
    # Not a win. Unless we use a different scheme: vector of scratch_B.
    #
    # Actually use 16 scratch_B cells (one per j of the current k-row):
    #   for k:
    #     for j: copy scratch_B[j], B[k,j]   (16 copies, B reads = 16)
    #     for i: copy scratch_A, A[i,k]
    #       for j: mul tmp, scratch_A, scratch_B[j]; add C[i,j], tmp

    # Reads per k-iteration:
    #   16 copies of B (16 B reads, 16 scratch_B writes)
    #   16 copies of A (16 A reads)
    #   16 i * (16 muls + 16 adds) = 256 muls + 256 adds
    #   muls read scratch_A and scratch_B[j]: 256 + 256 = 512 reads on scratchpads
    #   adds read C[i,j] and tmp: 256 + 256 = 512 reads

    # Per k: 16 + 16 + 512 + 512 = 1056 reads
    # Total over 16 k: 16,896 reads

    # Layout per total:
    #   B reads: 256 (each B[k,j] read once)
    #   A reads: 256 (each A[i,k] read once)
    #   scratch_A: read 16 i * 16 j = 256 per k * 16 k = 4096
    #   scratch_B[j]: 16 cells. Each scratch_B[j] for fixed k is read 16 times
    #     (one per i). Across 16 k values, total = 16 * 16 * 16 = 4096 across
    #     16 cells = 256 per cell? Or 4096 per cell?
    #     16 scratch_B cells. For each k: each cell read 16 times. Across 16 k:
    #     each cell read 16*16 = 256 times.
    #   tmp: 4096 reads (one per add)
    #   C[i,j]: 16 reads each (16 adds) + 1 exit = 17 reads. 256 cells.

    # Layout 7:
    layout7 = [4096, 4096] + [256]*16 + [17]*256 + [1]*512  # 4096 scr_A, 4096 tmp, 16 scr_B, 256 C, 512 (256A+256B)
    cost_var7 = cost_of_multiset(layout7)
    print(f"\nVariant 7 (scratch_A + 16 scratch_B): {len(layout7)} cells, {sum(layout7):,} reads, cost {cost_var7:,}")

    # ALIAS A with C in variant 7:
    layout7a = [4096, 4096] + [256]*16 + [18]*256 + [1]*256  # A=C aliased: 17 reads + 1 from A copy = 18
    cost_var7a = cost_of_multiset(layout7a)
    print(f"Variant 7a (alias A=C): {len(layout7a)} cells, {sum(layout7a):,} reads, cost {cost_var7a:,}")

    # ========================================================================
    # NOW: this is below 69,697! Either my model is wrong or there's a
    # better IR to write. Let me check feasibility more carefully.
    # ========================================================================

    print("\n" + "=" * 60)
    print("Testing Variant 7a feasibility -- generating actual IR")
    print("=" * 60)


if __name__ == '__main__':
    main()
