# MNIST-medium: 512-filter CG pair

**The 2% error target passes on two independent A100 hosts and the spatial grid.**
The frozen learner combines random convolutional features and an RBF kernel,
each fitted with 300 conjugate-gradient iterations.
Contributor: [@jurajselep](https://github.com/jurajselep).

## Results

| Implementation | Accuracy, mean ± SD | Correct / 110,000 | Energy (mJ) | Runtime (ms) |
| --- | ---: | ---: | ---: | ---: |
| A100, measured on two hosts | **98.12% ± 0.15 pp** | **107,932** | **65,000** | **260** |
| Spatial grid, modeled | **98.11% ± 0.14 pp** | **107,917** | **1,500** | **1.2 × 10⁷** |

Each implementation was evaluated on 11 prespecified draws. The ± value is the
sample standard deviation in percentage points. Both pass the exact threshold
of **107,800 correct**: A100 mean error is **1.88%** and grid mean error is
**1.8936364%**. Energy and runtime use two significant figures.

## Measurement scope

**A100:** fresh training and prediction on resident GPU tensors, measured with
synchronized CUDA graph replays, including target encoding. Preprocessing,
transfers, fixed-filter preparation, allocation, warmup and graph capture are
excluded. Results average the two hosts' five-round medians.

**Spatial grid:** a complete fresh training-and-prediction program, including
initialization and tape I/O; dataset selection and resizing are excluded. Its
**globally serialized schedule takes about 3.5 hours**, with
**1,264,557,504 bytes** peak scratch. Grid energy follows the model's memory and
communication accounting. Different numerical primitives require its own
accuracy qualification. The [grid report](grid/README.md) gives the exact
scope and costs, computed by the repository's unmodified shared scorer.

## Verify saved evidence

Run from the repository root. Complete the setup below once, then check
**SHA256SUMS, A100 evidence and grid evidence together**:

```bash
python "$SUB/verify.py" --all --raw-dir /tmp/mnist-raw
```

This checks saved results without a GPU or a full grid-scoring run. It verifies
source/input/output hashes, independently rescores predictions, recomputes
A100 energy, and audits recorded grid totals and scorer provenance. Without
`--all`, the verifier checks A100 evidence only. A fresh
[full grid score](grid/README.md#reproduce) needs a
128 GB RAM host and takes about 11 minutes; it does not require fresh predictions.

<details>
<summary>Setup: Python 3.11, pinned NumPy and canonical MNIST data</summary>

Pin OpenBLAS to Haswell on x86 to reproduce the recorded resize hashes.
Verification requires NumPy, not PyTorch.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
SHARED="$PWD/mnist/submissions/grid-mlp-scoring-20260912"
export PYTHONPATH="$PWD:$SHARED"
export OPENBLAS_CORETYPE=Haswell
python3.11 -m venv .venv-cg-verify
source .venv-cg-verify/bin/activate
python -m pip install numpy==2.1.2
python - <<'PYDATA'
from pathlib import Path
from mnist.code import data
for kind in ("train_images", "train_labels"):
    data.download_source(Path("/tmp/mnist-raw"), *data.SOURCES[kind])
PYDATA
```

Saved outputs are losslessly packed in `evidence/a100/predictions.npz` and
`grid/evidence/outputs.npz`, preserving original file bytes and hashes. The
verifiers read these archives directly and also accept fresh `.npy` files.
Raw NVML samples remain in the two compressed host records.

</details>

<details>
<summary>Learner and frozen evaluation protocol</summary>

1. Transform each pixel with `asin(sqrt(clamp(x, 0, 1)))`.
2. Apply 512 fixed Gaussian 3×3 filters with zero padding, ReLU and 3×3 mean
   pooling, producing 4,608 features. NumPy PCG64 seed 0 generates weights
   scaled by 1/3 and biases scaled by 0.1. Filters are independent of data.
3. Center features using training means and fit one-versus-rest ±1 targets
   with ridge strength `0.001 * mean(diag(Phi.T @ Phi))`.
4. Fit a second ridge model using `exp(-0.3 * squared_distance)` on the
   transformed pixels, with diagonal regularization 0.01.
5. Solve both systems from zero with exactly 300 Jacobi-preconditioned FP32
   conjugate-gradient iterations; no adaptive stopping or denominator guard.
6. Standardize each model's scores by its per-example sample standard
   deviation, sum the vectors, and return the first maximum.

The configuration, protocol and seeds **2026091600–2026091610** were frozen
before evaluation. For each seed, draw a direct `PCG64(seed).permutation(60000)`
from the official MNIST training pool: the first 10,000 examples train the
learner and the next 10,000 test it. Subsets are disjoint within each draw;
independent draws may overlap. Convert pixels to FP32, divide by 255, and apply
the repository's exact area resize to 9×9. Fit a fresh learner per draw and
expose test labels only after predictions are returned.

**Frozen configuration versus measured execution:** `config.json` retains the
learner's default convolution batch size of **128**. `gpu_benchmark.py`
explicitly uses **10,000**, recorded in each host's `configuration` field.
Likewise, `protocol.json` retains the historical local-eager `timing` field,
which includes preparation and host prediction transfer. The A100 results
above use the CUDA graph scope recorded in each host's **`scope`** field and
described under Measurement scope. The frozen JSON files remain unchanged so
the recorded source hashes stay valid.

</details>

<details>
<summary>A100 hardware, measurement rounds and sensor checks</summary>

Both hosts use **NVIDIA A100-SXM4-40GB**, Python 3.11.10,
PyTorch 2.5.1+cu124, CUDA 12.4 and NumPy 2.1.2, with TF32 disabled.
Each independently achieves the reported accuracy. All scores are finite and
every draw's eager and CUDA graph predictions agree.

| Physical host | Driver | Power limit | Median energy (mJ) | Median runtime (ms) |
| --- | --- | ---: | ---: | ---: |
| Vast.ai machine 139975 | 595.71.05 | 350 W | 65,000 | 260 |
| Vast.ai machine 141074 | 580.159.03 | 400 W | 65,000 | 260 |

Timing uses seed 2026091600. Every replay encodes targets, transforms pixels,
computes features and both systems, performs both zero-start solves, and
predicts 10,000 queries.

GPU energy uses NVML cumulative-counter differences minus paired idle power
times active duration. Each host ran five rounds of **59 complete replays**
(about 15 seconds), bracketed by 10-second idle measurements after 5-second
settles. Raw counter stamps and 50 Hz power samples are retained. CUDA/NVML
PCI identities match; both dense-matmul sensor checks passed, with counter and
sampled power differing by about 0.34%.

Peak PyTorch allocation was **3,525,344,256 bytes** per host; reservation was
**5,054,136,320 bytes**. These cover preparation, three eager warmups, capture
and first replay, excluding the sensor diagnostic and CUDA driver/context.
The host records retain exact per-round values and hardware/software details.

</details>

<details>
<summary>Reproduce fresh measurements on two A100 hosts</summary>

Run the following on each host after preparing the same raw data there.
Replace `HOST` with a distinct name and `VAST_MACHINE_ID` with the provider's
actual physical machine ID. Retain each output JSON and its adjacent prediction
files; place both sets in one directory and check them with
`verify.py --raw-dir /tmp/mnist-raw --a100-dir /path/to/both-host-results`.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
export PYTHONPATH="$PWD"
export OPENBLAS_CORETYPE=Haswell
python3.11 -m venv .venv-a100
.venv-a100/bin/pip install -r "$SUB/requirements-gpu.txt"
.venv-a100/bin/python "$SUB/gpu_benchmark.py" --raw-dir /tmp/mnist-raw \
  --output /tmp/cg-a100-HOST.json --qualify-all --settle-seconds 5
.venv-a100/bin/python - /tmp/cg-a100-HOST.json VAST_MACHINE_ID <<'PYHOST'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
record = json.loads(path.read_text())
record["vast"] = {"machine_id": int(sys.argv[2])}
path.write_text(json.dumps(record, indent=2) + "\n")
PYHOST
```

</details>

Evidence: [A100 records and predictions](evidence/a100/), [configuration](config.json),
[protocol](protocol.json), [A100 verification](evidence/verification.json) and
[grid verification](grid/evidence/verification.json).
