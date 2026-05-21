# Meta-Skill: Objective-Reformulating Hill Climb

Use this when optimizing energy efficiency or another weighted resource score.

The main trick is not "try random local changes." First reformulate the scorer into a useful pressure model, then hill climb the representation that moves expensive events into cheaper places or shorter lifetimes.

## Agent Loop

1. Make the real scorer executable.
   Every candidate must be checked by the real scorer, not only by a proxy.

2. Reformulate the objective.
   Identify:
   - cost events: reads, writes, ops, moves, messages, live ranges, cache misses;
   - event weights: address, distance, time, frequency, span, fanout, criticality;
   - legal moves: reorder, split, copy, defer, pre-stage, recompute, place, color, cleanup.

3. Build a pressure metric.
   It should rank where one legal move is likely to matter, and consider the total cost of a move throughout lifetime of run, not just one step.

   Examples:

   ```text
   weighted_read_pressure = future_reads * future_lifetime_span * placement_cost
   weighted_comm_pressure = future_messages * distance * criticality
   weighted_live_pressure = future_uses * live_range_span * allocation_cost
   ```

4. Hill climb with checkpoints.
   For each accepted move, immediately write:
   - scored artifact;
   - raw/search artifact;
   - a short run note or stdout line with command, move, old score, new score.

5. Compose move families.
   Do not stop after one family plateaus. Try:
   - pressure shaping;
   - placement/allocation/coloring;
   - schedule or dataflow variants;
   - cleanup of copies/spills/temporary scaffolding;
   - pressure search again after cleanup.

6. Keep negative branches small and recorded.
   A failed branch is useful if it explains why an intuition does not match the scorer.

## Frontier Checklist

At every new best, record:

- verified score;
- final artifact path;
- raw/search artifact path;
- exact command to continue;
- transforms still improving;
- transforms exhausted.

## Pattern To Look For

Good runs often look like:

```text
baseline
-> objective reformulation
-> pressure shaping
-> placement/allocation
-> plateau
-> cleanup/postprocess
-> renewed search
```

## Matmul Example

For the matmul run, the useful objective was:

```text
energy ~= sum(reads * address_cost)
pressure ~= future_reads * future_lifetime_span
```

The winning sequence was:

```text
pressure-ranked suffix splits
-> DP chain coloring
-> mixed A/B/derived pressure search
-> generalized copy elimination
-> repeated copy-elim waves
```

## Specific Matmul Observation

Instead of directly attacking the high-level problem, start with a decent base IR. Convert each IR value into a value interval, score each interval by future weighted pressure, then search legal edits that shorten or relocate the highest-pressure intervals.

After each edit, run DP coloring so non-overlapping hot intervals share low-cost registers. This turns the scorer into a structured interval-packing problem instead of a blind IR mutation search.

The key observation is that interval pressure, roughly `sum(read_cost * future_reads)` across a value's remaining lifetime, is a good proxy for the true matmul scorer. That shifts the task from inventing a new schedule from scratch to taking a reasonably good IR and reformulating it as an interval-packing hill climb.

Example commands:

```bash
python3 matmul/submissions/search_preevict_pressure.py START.raw.ir \
  --kinds A B derived \
  --rank pressure \
  --top 500 \
  --max-iters 12 \
  --write-best OUT.ir \
  --write-raw OUT.raw.ir \
  --checkpoint-each-accept

python3 matmul/submissions/search_copy_elim_general.py OUT.raw.ir \
  --top 1700 \
  --max-iters 10 \
  --write-best NEXT.ir \
  --write-raw NEXT.raw.ir \
  --checkpoint-each-accept
```
