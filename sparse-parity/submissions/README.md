# Submitting a sparse-parity solver

Each submission is an `.ir` file (the straight-line program) plus a `.md`
report giving its energy, recovery rate, and how the circuit works.

If a generator produced the IR, add it here as a `.py` file too — unless it
is already a `generate_*` function of [`mask_sparse_parity.py`](../mask_sparse_parity.py),
in which case the report just names the call.

See [`scan_full_mask32.md`](scan_full_mask32.md) for the expected shape, and
the [problem README](../README.md) for the scoring rules.
