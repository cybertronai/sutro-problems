# a100 d4

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
