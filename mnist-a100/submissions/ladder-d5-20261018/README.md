# Research archive: Ladder variant submitted for D5

**Research inclusion; D5 qualification remains unresolved.** The existing author records reconstruct 3/3 (median **308.420616 ms**) separately from the initial reviewer group, which passed **1/3**. Two complete reviewer runs missed D5 by **116 and 57 labels**; all 45 timed calls are retained. There is no qualifying reviewer-group median and no leaderboard promotion. The submitted source and original author records remain unchanged. See [reviewer evidence](reviewer-evidence.json).

Teacher/Ladder attribution: @spartanhaden’s [PR #121](https://github.com/cybertronai/sutro-problems/pull/121), earlier kernel-PCG work by @islamborghini, and Ladder work by @SethTS and Rasmus et al. This is a substantial optimizer/schedule/kernel/readout variant; remaining full lineage is not settled here.

author measurements: three passing sandboxed a100-80gb runs per source. [results](results.json).

| source | median ms | pooled error |
| --- | ---: | ---: |
| [default](submission.py) | 308.420616 | 1.827879% |

gpu reproduction from `mnist-a100/`, with modal configured:

```sh
python run_modal.py submissions/ladder-d5-20261018/submission.py:ladder_kernel \
  --difficulty 5 --runs 3 --json /tmp/d5-default
```

offline check from this directory: `python3 -B verify_records.py`. [records](evidence/) omit console logs and gpu identifiers; filenames normalized.
