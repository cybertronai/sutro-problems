# Weighted-Lifetime Matmul Hill Climb

This run moved the 16x16 matmul artifact from the `67,6xx` range to a verified `66,707`.

## Files

- submission companion: `matmul/submissions/weighted_lifetime_copyelim_66707.py`
- scored IR: `matmul/submissions/weighted_lifetime_copyelim_66707.ir`
- raw continuation IR: `matmul/submissions/weighted_lifetime_copyelim_66707.raw.ir`
- hillclimb meta-skill: `matmul/doc/metaskills/meta_skill_trace_hillclimb.md`
- journal harness meta-skill: `matmul/doc/metaskills/meta_skill_journal_harness.md`

Verify:

```bash
python3 matmul/submissions/weighted_lifetime_copyelim_66707.py
```

Expected:

```text
weighted_lifetime_copyelim_66707.ir  cost=66,707
```

## Core Idea

The successful reframing was:

```text
energy ~= sum(reads * address_cost)
```

So the useful hill-climb target was not operation count, which my initial agents tended to target. The better target was considering the lifetime weighted future read pressure:

```text
pressure ~= future_reads * future_lifetime_span
```

The objective then becomes:

```text
Which values will be read many times over a long future span,
and how do we make those reads happen in cheaper addresses/windows?
```

## What Worked

1. Pressure-ranked suffix splits
   Copy a value after an early read, redirect the later reads, then recolor. This creates short hot intervals that can be placed cheaply.

2. DP chain coloring
   Convert trace values into intervals. This finds hot zones of reads.

   ```text
   define_time, last_read_time, read_count
   ```

   Then assign compatible high-read intervals to low addresses.

3. Mixed A/B/derived pressure search
   B-only and derived-only helped, but the best path combined them.

4. Generalized copy elimination
   After pressure splitting, many copy windows became removable. If the source stayed live, reads of the copy destination could be redirected back to the source. This made the trace cheaper without changing the final result.

5. Repeated cleanup waves
   Late improvements were mostly forward copy-elim waves. Each accepted move usually redirected 4 reads and saved about 2 points.

## Score Path

Approximate path:

```text
~67,631
-> pressure-ranked suffix/pre-evict search
-> under 67,000
-> copy elimination
-> 66,876
-> mixed A/B/derived pressure
-> 66,795
-> generalized copy elimination
-> 66,761
-> repeated copy-elim waves
-> 66,707
```

## What Did Not Work

- output deferral and prethrow mostly tied or were invalid;
- prepush did not reopen the late plateaus;
- wider pressure search from copy-elim frontiers found no improvement;
- rank variants like `reads` and `span` did not beat pressure;
- retile, micro-wave, B-panel grouping, bursty accumulators, outer reuse, and EDF-style accumulator scheduling scored worse because they increased live pressure.

## Reproduce The Search Pattern

Pressure search:

```bash
python3 matmul/submissions/search_preevict_pressure.py START.raw.ir \
  --kinds A B derived \
  --rank pressure \
  --top 500 \
  --max-iters 12 \
  --write-best OUT.ir \
  --write-raw OUT.raw.ir \
  --checkpoint-each-accept
```

Copy elimination:

```bash
python3 matmul/submissions/search_copy_elim_general.py OUT.raw.ir \
  --top 1700 \
  --max-iters 10 \
  --write-best NEXT.ir \
  --write-raw NEXT.raw.ir \
  --checkpoint-each-accept
```

Useful loop:

```text
pressure search until plateau
copy-elim until plateau
pressure/prepush check
copy-elim continuation
```

## Prompt Pattern That Helped

Generally prompt types I used to get here:

- calculate how often each value will be read again;
- weight reads by address/register cost;
- treat hotness as time-weighted future read pressure;
- split or push values near their hot window;
- remove values once their useful hot window is over;
- keep creative branches, but verify every branch with the scorer.


What worked best was reformulating the scorer into something the agent could hill climb directly: total lifetime cost of future reads. In this case, that meant time-weighted read pressure.
