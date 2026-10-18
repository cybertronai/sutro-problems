# a100 d5

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
