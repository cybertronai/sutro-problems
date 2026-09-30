# Same-board comparison — 2026-09-29

The unchanged [previous ensemble](../../energy/entries/mlpg_k4_w256_s800_b512.py)
and [kernel classifier](kernel_pcg.py) each passed three runs of scorer 1.2.0 on
the same physical A100 80GB PCIe, at an unchanged 300 W power limit.

| Method | Median ranked ms/call | Median mJ/call above idle |
| --- | ---: | ---: |
| Four CUDA-graph MLPs, 800 steps | 253.355690 | 14,897.366278 |
| Metric kernel, 16-step preconditioned CG | **35.588786** | **7,664.458255** |

The ratio of medians is **7.119x faster and 48.55% less energy**. This comparison
does not replace the submission's three-distinct-board leaderboard measurements
(38.319672 ms and 8,070.955853 mJ).

## Runs

| Order | Method | Ranked ms | Energy, mJ | MNIST error |
| ---: | --- | ---: | ---: | ---: |
| [1](evidence/same-board/run-1-baseline.json) | Ensemble | 251.800952 | 14,916.849704 | 3.084545% |
| [2](evidence/same-board/run-2-ours.json) | Kernel | 37.000474 | 7,725.035437 | 3.019091% |
| [3](evidence/same-board/run-3-ours.json) | Kernel | 34.792796 | 7,642.708336 | 2.945455% |
| [4](evidence/same-board/run-4-baseline.json) | Ensemble | 253.518440 | 14,715.605311 | 2.959091% |
| [5](evidence/same-board/run-5-baseline.json) | Ensemble | 253.355690 | 14,897.366278 | 3.101818% |
| [6](evidence/same-board/run-6-ours.json) | Kernel | 35.588786 | 7,664.458255 | 2.826364% |

Each run used 11 MNIST and four hold-out calls in fresh sandboxed processes,
plus the standard energy window. Every scoring and energy check passed. The
fixed order was ensemble/kernel, kernel/ensemble, ensemble/kernel; no scored
run was discarded or replaced. Pairwise speedups are 6.805–7.287x and energy
reductions are 48.06–48.55%.

Both methods received independent fresh random draws from the unmodified scorer,
not identical examples. GPU clocks and temperatures were not locked. The GPU,
power limit, software image and measurement protocol were shared, and the
methods ran sequentially rather than concurrently. All idle-utilization,
context-count and telemetry-reference checks passed.

The kernel method's mean board power during the energy windows was higher
(276–280 W versus 131–133 W), but its shorter time per call reduced energy.
Reported energy subtracts idle and empty-call overhead; it excludes the host
and cold start, as the official protocol specifies.

## Reproduction and evidence

The controller reserved one Modal A100-80GB container using the stock runner's
CUDA/PyTorch image, then launched six clean `python mnist.py FILE:FUNCTION
--difficulty 2 --json OUTPUT` subprocesses with `MNIST_SANDBOX=required`.
To reproduce the hardware control, keep all six subprocesses on one allocated
GPU in the order above; six independent `run_modal.py` invocations would not
guarantee the same board. Entry points are `mlp` for the ensemble and `classify`
for the kernel method. Use separate output files for every run.

The [manifest](evidence/same-board/manifest.json) preserves both source snapshots
and the prescribed order. Each raw result includes the scorer hash measured
inside the remote container, source hash, UTC timestamps, complete stdout/stderr,
timed-call records, energy windows, and before/after hardware metadata. The
copies here are byte-identical to the saved outputs. The local evidence verifier
reconstructs all six scores and energy values and checks the hardware identities.

- GPU UUID: `GPU-3536d6c5-505f-4e14-94e6-326bbd769c02`.
- Modal app: `ap-3DM0j6iU8LjOkKSd3xD5os` (stopped, zero tasks after evaluation).
- Ensemble SHA-256: `24453851a3faec1808bb157296ac3bab21f8c88b170dea11352bd98dbd6d0ba5`.
- Kernel SHA-256: `fc0ddef0d3dba8f00fc44f9cd576548c2a5184c5cb88741767d33d6f10b38012`.
- Remote scorer SHA-256: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`.
