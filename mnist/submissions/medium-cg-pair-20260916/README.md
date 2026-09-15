# MNIST-medium: 512-filter CG pair

**2% target passed on two distinct A100 hosts: 98.12% ± 0.15 pp accuracy.**
This frozen FP32 learner combines random convolutional features and an RBF
kernel, each fitted with 300 conjugate-gradient iterations.
Contributor: [@jurajselep](https://github.com/jurajselep).

## Results

Each Vast.ai host independently achieves **107,932 / 110,000 correct** over
all 11 prespecified draws: **1.88% mean error**. The ± value is the sample
standard deviation across draws, in percentage points. Qualification requires
at least 107,800 correct. All scores were finite, and every draw's eager and
CUDA graph predictions agreed.

Both GPUs are **NVIDIA A100-SXM4-40GB**. Energy and runtime below use two
significant figures; each host value is the median of five measurement rounds.

| Host (machine ID) | Driver | Power limit | Adjusted energy (mJ) | Runtime (ms) |
| --- | --- | ---: | ---: | ---: |
| Croatia (139975) | 595.71.05 | 350 W | 65,000 | 260 |
| Virginia (141074) | 580.159.03 | 400 W | 65,000 | 260 |
| Mean of host medians | — | — | **65,000** | **260** |

Exact per-round values are retained in the A100 records below.
Peak PyTorch allocation was **3,525,344,256 bytes**
on each host; peak reservation was **5,054,136,320 bytes**. These cover model
preparation, three eager warmups, capture and first replay, excluding the sensor
diagnostic and CUDA driver/context allocations.

**Spatial-grid energy and runtime are unmeasured for this learner.**

## Learner

1. Transform each pixel with `asin(sqrt(clamp(x, 0, 1)))`.
2. Apply 512 fixed Gaussian 3×3 filters with zero padding, ReLU and 3×3 mean
   pooling, producing 4,608 features. NumPy PCG64 seed 0 generates weights
   scaled by 1/3 and biases scaled by 0.1. The filters are independent of data.
3. Center features using training means and fit one-versus-rest ±1 targets
   with ridge strength `0.001 * mean(diag(Phi.T @ Phi))`.
4. Fit a second ridge model using `exp(-0.3 * squared_distance)` on the
   transformed pixels, with diagonal regularization 0.01.
5. Solve both systems from zero with exactly 300 Jacobi-preconditioned FP32
   conjugate-gradient iterations; no adaptive stopping or denominator guard.
6. Standardize each model's scores by its per-example sample standard
   deviation, sum both score vectors, and return the first maximum.

A100 convolution uses all 10,000 examples. The GPU environment uses Python 3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4 and
NumPy 2.1.2, with TF32 disabled. Qualification is checked independently on each host.

## Protocol and measurement scope

The configuration, protocol and seeds **2026091600–2026091610** were frozen
before evaluation. For each seed, draw a direct `PCG64(seed).permutation(60000)`
from the official MNIST training pool: the first 10,000 examples train the
learner and the next 10,000 test it. Training and test subsets are disjoint
within each draw; independent draws may overlap. Convert pixels to FP32, divide
by 255, and apply the repository's exact area resize to 9×9. Fit a fresh learner
per draw and expose test labels only after predictions are returned.

The frozen protocol also retains metadata for the initial local CPU run; the
A100 timing and artifact scope is defined below and in each host record.
A100 timing measures synchronized CUDA graph replays on seed 2026091600.
Every replay encodes targets, transforms pixels, computes features and both
systems, performs both zero-start solves, and predicts 10,000 queries. Dataset
preprocessing, host/device transfers, data-independent filter generation/upload,
allocation, warmup, capture and validation are excluded.

GPU energy uses NVML cumulative-counter differences minus paired idle power
times active duration. Each host ran five rounds of **59 complete replays**
(about 15 seconds), bracketed by 10-second idle measurements after 5-second
settles. Raw counter stamps and 50 Hz power samples are retained. CUDA/NVML
PCI identities match; both dense-matmul sensor checks passed, with counter and
sampled power differing by about 0.34%. The combined result averages host medians.

## Reproduce and verify

Run from the repository root with Python 3.11. Verification needs only NumPy;
it reconstructs all datasets and checks the saved results without a GPU.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
python3.11 -m venv .venv-cg-verify
.venv-cg-verify/bin/pip install numpy==2.1.2
.venv-cg-verify/bin/python - <<'PYDATA'
from pathlib import Path
from mnist.code import data
for kind in ("train_images", "train_labels"):
    data.download_source(Path("/tmp/mnist-raw"), *data.SOURCES[kind])
PYDATA
.venv-cg-verify/bin/python "$SUB/verify.py" --raw-dir /tmp/mnist-raw
(cd "$SUB" && sha256sum -c SHA256SUMS)
```

For a fresh measurement, run the following on each of two A100 hosts after
preparing the same raw data there. Retain each output JSON and its adjacent
prediction files. Use `verify.py --raw-dir /tmp/mnist-raw
--a100-dir /path/to/both-host-results` to check both host records together. Replace `HOST` with a distinct name for each host.
After benchmarking, attach the provider’s actual machine ID (replace
`VAST_MACHINE_ID` below); this records physical-host identity alongside the GPU UUID.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
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

The verifier independently rescores every prediction, validates frozen source,
dataset and artifact hashes, and recomputes counter and sampled energy. It reads
`.json` or compressed `.json.gz` host records directly. The two compressed records
retain hardware/software details, per-draw counts and raw NVML samples.

See [A100 records and predictions](evidence/a100/), [configuration](config.json),
[protocol](protocol.json) and [independent verification](evidence/verification.json).
