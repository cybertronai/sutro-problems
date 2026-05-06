# Submissions

Each submission lands three files here:

| file | content |
|------|---------|
| `<name>_<YYYY-MM-DD>.json` | canonical `(training_energy_J, training_duration_s, test_char_accuracy, gpu_name, …)` tuple written by `run_eval.py --results-json`. This is what `submit.py` saves and links from the Record History row. |
| `<name>_<YYYY-MM-DD>.log` | full stdout from `run_eval.py`, including the `training energy (J)` / `test char-accuracy` block at the end |
| `<name>_<YYYY-MM-DD>.nvml.json` | the JSON summary line from `verify_nvml.py` produced on the same host before the run, as evidence the energy counter was exposed and monotonic |

`<name>` is the submission file's stem (e.g. `my_submission` from
`my_submission.py`), or `baseline_transformer_<config>` /
`baseline_ngram_n<N>` for direct `run_eval.py` runs.

`submit.py` writes all three automatically and appends the Record
History row in [`../README.md`](../README.md). For manual runs via
`RUNBOOK.md`, drop the files in by hand and append the row yourself.
