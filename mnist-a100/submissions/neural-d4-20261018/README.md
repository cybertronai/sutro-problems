# Research archive: neural learner submitted for D4

**Research inclusion; D4 qualification and author-source attribution remain unresolved.** The submitted 18,558-byte source passed **1/3** independent reviewer runs; two complete runs missed D4 by **26 and 20 labels** (all 45 timed calls retained). The author’s **35.678700 ms** median below belongs to records identifying a different, unsupplied 18,611-byte source (`c7b78944…c52d78b1`), not the packaged source (`1966a800…83aa77`). It is historical author evidence, not a verified median for `submission.py`. No qualifying reviewer-group median or leaderboard promotion is made. See [reviewer evidence](reviewer-evidence.json).

Attribution: four documented teacher helpers follow @spartanhaden’s kernel/Ladder D4; related tangent/Muon/cross-teaching work is credited to @likelytobelaura and joint graph propagation to @sjbaebae. Complete development lineage remains unsettled. Original source and records are preserved.

author measurements: three passing sandboxed a100-80gb runs per source. [results](results.json).

| source | median ms | pooled error |
| --- | ---: | ---: |
| [default](submission.py) | 35.678700 | 2.160606% |

gpu reproduction from `mnist-a100/`, with modal configured:

```sh
python run_modal.py submissions/neural-d4-20261018/submission.py:classify \
  --difficulty 4 --runs 3 --json /tmp/d4-default
```

offline check from this directory: `python3 -B verify_records.py`. [records](evidence/) omit console logs and gpu identifiers; filenames normalized.
