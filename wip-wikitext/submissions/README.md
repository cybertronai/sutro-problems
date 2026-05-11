# Submissions

One subdirectory per submission. Submitter authors `submission.py`;
`submit.py` populates the rest.

```
submissions/
└── <name>/
    ├── submission.py     # train(train_text, valid_text=None) -> CharModel
    ├── result.json       # canonical (training_energy_J, training_duration_s,
    │                     #   test_char_accuracy, gpu_name, _nvml, …) tuple
    ├── nvml.json         # NVML probe summary from the same Modal host
    │                     #   (same data as result.json["_nvml"], split out
    │                     #   as a standalone evidence file)
    └── run.log           # captured stdout from `submit.py`
```

`<name>` is whatever directory name the submitter picks; the same
string ends up in the `Config` column of the Record History table. Pick
something short and snake_case (e.g. `modded_nanogpt`).

To submit:

```bash
python3 wip-wikitext/submit.py wip-wikitext/submissions/<name>
```

`submit.py` writes `result.json`, `nvml.json`, and `run.log`
automatically and appends the row to the Record History in
[`../README.md`](../README.md). For manual runs via
[`../RUNBOOK.md`](../RUNBOOK.md), drop the files in by hand and append
the row yourself.
