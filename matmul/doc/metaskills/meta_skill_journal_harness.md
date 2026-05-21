# Meta-Skill: Journal Harness For Research Loops

Use this when a search has many branches, multiple agents, or enough negative results that memory matters. The journal harness is not the optimizer. It is the control layer around experiments.

This is the compact version of the journal pattern used in the `*-problems-journal` repos.

## Core Idea

Separate three things:

```text
locked evaluator / harness
mutable candidate artifacts
structured research memory
```

The harness defines the rules and metric. Candidates are what agents edit. The journal records what was tried, what happened, and what should be tried next.

## Minimal Architecture

```text
agent(s)
  -> propose / claim hypothesis
  -> edit candidate
  -> run locked domain harness
  -> submit structured result
  -> review / publish useful findings
  -> add follow-up hypotheses
```

The durable store can be SQLite, JSONL, or another local database. SQLite was used in the journal repos because it is local, durable, searchable, and easy to inspect.

## Locked Harness Contract

A domain harness should be callable from the command line and produce structured output.

Recommended shape:

```bash
python journal/domain_harness.py \
  --candidate path/to/candidate \
  --seed 0 \
  --budget 300 \
  --json
```

It should report:

```json
{
  "metrics": {},
  "budget": {},
  "candidate": {
    "path": "...",
    "sha256": "..."
  },
  "environment": {},
  "seed": 0
}
```

The harness itself should also have a stable hash:

```bash
python journal/domain_harness.py --hash
```

Agents may edit candidates. Agents should not silently edit the harness, scorer, data path, or metric.

## Journal Records

A useful journal tracks:

- `harnesses`: accepted evaluator path, SHA256, locked files, config;
- `hypotheses`: queued or claimed ideas to test;
- `submissions`: raw experiment reports;
- `reviews`: checks of validity and interpretation;
- `publications`: accepted findings future agents should trust first.

For lightweight use, a single JSON file per branch is acceptable, but the full pattern is a structured journal with these concepts.

## Submission Fields

Every result should answer:

- what hypothesis was tested;
- what candidate changed;
- what command ran;
- what harness hash evaluated it;
- what metrics were observed;
- what changed from baseline;
- whether it was a win, loss, inconclusive, or invalid;
- what follow-up questions were created.

## Agent Protocol

Before experimenting:

1. Check journal status.
2. Search accepted publications or prior findings.
3. Confirm the accepted harness hash.
4. Run a smoke test.

Main loop:

```text
claim hypothesis
inspect prior work
implement candidate
run locked harness
submit structured result
record interpretation
propose follow-ups
```

## Why This Helped Here

The matmul search had many small branches and interruptions. The useful local adaptation was:

```text
--write-best OUT.ir
--write-raw OUT.raw.ir
--checkpoint-each-accept
--journal RUN.json
```

That is not the whole journal harness. It is the per-branch artifact layer that fits inside the larger journal pattern.

The full idea is:

```text
locked scorer + candidate artifacts + structured memory
```

The checkpoint files preserve the latest search state. The journal record explains why that state exists and whether future agents should build on it.

