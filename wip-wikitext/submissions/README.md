# Submissions

Each submission lands two files here:

| file | content |
|------|---------|
| `<name>_<YYYY-MM-DD>.log` | full stdout from `run_eval.py`, including the `training energy (J)` / `test char-accuracy` block at the end |
| `<name>_<YYYY-MM-DD>.nvml.json` | the JSON line from `verify_nvml.py` produced on the same host before the run, as evidence the energy counter was exposed and monotonic |

`<name>` is `ngram_n<N>`, `transformer_<config>`, or a custom string the
submitter chooses. After landing the files, append a row to the Record
History table in [`../README.md`](../README.md).
