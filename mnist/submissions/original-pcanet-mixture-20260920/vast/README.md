# Run the frozen learner on an allocated Vast.ai host

Use one exposed **NVIDIA A100-SXM4-40GB** per container, preferably the same
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image used for the first rerun.
Allow at least 32 GiB host RAM, four CPU threads, and sufficient disk for the
CUDA build cache, Python environment and evidence. The CUDA 12.4 compiler and
a C++ compiler are required. The GPU must be idle before validation starts.

Copy this submission directory to each of the two allocated hosts. Run the
following separately on each host, from the submission directory:

```sh
python vast/run_host.py --output generated/vast-rerun --raw raw
```

If a matching environment is already prepared, add `--no-install` to prohibit
dependency changes. Otherwise the runner installs only missing or mismatched
pinned dependencies: PyTorch 2.5.1 with CUDA 12.4, NumPy 2.1.2,
nvidia-ml-py 13.610.43, and Ninja 1.11.1.1. It makes no Vast.ai API calls.
Keep the SSH session alive or invoke the command through the host's normal
persistent-session mechanism.

The runner executes the unchanged `validate.py` with **K=100, four rounds,
60 complete training-and-prediction runs per round, and ten-second paired idle
windows**. Canonical MNIST files in `raw/` are reused when present; otherwise
the unchanged data loader downloads and verifies them before measurement.
The frozen learner, CUDA kernel, data loader and validation source are never
edited. Preflight records software/toolchain versions and GPU identity without
creating a CUDA context. The validation process performs its existing GPU
ownership probe, sensor check, accuracy checks, feature comparisons and control
measurements.

While running, output is echoed and written to
`generated/vast-rerun.runner/run.log`. On completion the log moves into the
results directory alongside `execution.json` and `preflight.json`. The runner
then runs `verify_results.py --raw raw` and saves `verification.json`. A failed
preflight, experiment or verification exits nonzero and preserves available
evidence. Existing results are never overwritten.

Retrieve the complete results directory from each host separately. Use each
recorded GPU UUID to establish that the two measurements used different physical
boards. Keep per-host results distinct; an average of two boards does not
replace the underlying raw measurements.

The fixed harness addresses NVML device 0 and CUDA device 0. This runner
therefore requires exactly one exposed GPU, with `CUDA_VISIBLE_DEVICES` unset,
`0`, or that GPU's UUID. A100 PCIe, SXM4-80GB and MIG instances do not match the
frozen hardware requirement. A host that fails the existing idle, sensor or
process checks is a failed measurement, not an energy result.

## Completed measurements

The UK and Slovenia runs are retained at `../evidence/vast-uk/` and
`../evidence/vast-slovenia/`. Both rentals were destroyed after copying their
evidence. `../evidence/vast-provisioning.json` records separate machine and host
IDs and the final absence of both instance IDs.

Recompute the three-board comparison from this submission directory:

```sh
mkdir -p generated
python vast/compare.py --raw /path/to/raw --output generated/new-comparison.json
```

The comparison includes the initial Modal result. It verifies each run, checks
distinct GPU UUIDs and matching source/data/configuration, and compares saved
predictions. Per-host medians and power limits remain separate.
