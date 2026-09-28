# Triton Ladder submission — difficulties 3, 4 and 5, 2026-09-28

The Ladder network that sets the difficulty bands, with its step fused into Triton kernels and captured in a CUDA graph, is the first entry at difficulties 4 and 5 and the fastest at difficulty 3. Each row is the median of three sandboxed Modal A100-80GB runs of the unchanged scorer 1.2.0. Every run passed.

| Difficulty | Band | Steps | ms/call | mJ/call above idle | MNIST error, 33 calls | File |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2.70% | 6,500 | **6,556.9** | 676,850 | 2.45% | [ladder_d3.py](ladder_d3.py) |
| 4 | 2.30% | 11,000 | **11,094.6** | 1,143,255 | 2.14% | [ladder_d4.py](ladder_d4.py) |
| 5 | 1.90% | 36,000 | **36,201.2** | 4,410,730 | 1.71% | [ladder_d5.py](ladder_d5.py) |

The three files differ only in `STEPS`. The model, objective and hyperparameters are `ladder-xlong-s11` of [`release-cutoffs-20260925`](../../../mnist/experiments/release-cutoffs-20260925/README.md), unchanged; the step count sets the schedule, which decays linearly over the last third, as the recipe does at any budget. The plain-PyTorch port of the same recipe, its faithfulness check against the study's code, and the first difficulty 3 entry are in [`../ladder-graph-20260928`](../ladder-graph-20260928/README.md).

## Runs

| Difficulty | Run | GPU | Score, ms/call | Energy, mJ/call | MNIST accuracy | Hold-out |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 3 | 1 | A100 80GB PCIe | 6,556.932 | 629,979 | 97.51% (107,261/110,000) | KMNIST 96.06% |
| 3 | 2 | A100 80GB PCIe | 6,529.751 | 676,850 | 97.52% (107,277/110,000) | Fashion-MNIST 87.25% |
| 3 | 3 | A100 80GB PCIe | 6,558.116 | 680,203 | 97.62% (107,386/110,000) | Fashion-MNIST 87.42% |
| 4 | 1 | A100 80GB PCIe | 11,124.947 | 1,143,255 | 97.89% (107,684/110,000) | KMNIST 96.79% |
| 4 | 2 | A100-SXM4-80GB | 11,066.852 | 1,250,708 | 97.85% (107,632/110,000) | Fashion-MNIST 87.94% |
| 4 | 3 | A100 80GB PCIe | 11,094.609 | 1,139,689 | 97.83% (107,614/110,000) | KMNIST 97.00% |
| 5 | 1 | A100 80GB PCIe | 36,201.185 | 4,707,002 | 98.25% (108,072/110,000) | KMNIST 97.35% |
| 5 | 2 | A100-SXM4-80GB | 36,257.673 | 4,042,259 | 98.33% (108,160/110,000) | KMNIST 97.37% |
| 5 | 3 | A100-SXM4-80GB | 35,930.671 | 4,410,730 | 98.28% (108,110/110,000) | Fashion-MNIST 88.31% |

Within each difficulty the runs landed on different boards (GPU UUIDs in the records), and all nine passed every energy check, with telemetry references of 7.19-8.76 J/TFLOP. Records and output: [difficulty 3](evidence/d3/), [difficulty 4](evidence/d4/), [difficulty 5](evidence/d5/).

**A failed attempt.** Difficulty 3 was first run at 5,000 steps, the step count of the graph-only entry. One of its three runs was disqualified at 97.278% MNIST accuracy, 24 predictions short of the band; the other two passed at 5,021 and 5,039 ms ([evidence](evidence/d3-5000-steps/)). Across seven scored runs at 5,000 steps (four graph-only, three Triton), the 11-call mean error ranged over 2.50-2.72%, so 5,000 steps sits about 0.1 points from the band and a run can miss. 6,500 steps was chosen from the study's step curve (2.83% at 4,000, 2.29% at 8,000) and read 2.45%.

## The kernels

The graph-captured step in PyTorch takes 3.14 ms; it is launch-bound, several hundred small kernels around matmuls that total about 4.4 GFLOP. Two fused Triton kernels per layer and direction replace most of them:

* **Encoder:** batch normalisation of each stream (labelled and unlabelled, normalised separately as the recipe does), the per-layer noise, the bias and the activation. The noise is drawn in the kernel from a Philox seed that advances with the step counter on the GPU, so each graph replay gets fresh noise.
* **Decoder:** batch normalisation of the top-down signal and the unit-wise combinator MLP (3 -> 2 -> 2 -> 1, leaky ReLU). Its 17 per-unit parameters live in one tensor; Adam is elementwise, so this changes nothing.

Each program owns 8 columns and all 250 rows, so batch statistics are exact; backward kernels recompute the forward instead of storing it. Matmuls, softmax, the losses and fused Adam stay in PyTorch, and the whole step stays in one CUDA graph.

Measured on one A100-SXM4-80GB with CUDA events ([verify_kernels.py](verify_kernels.py), [output](evidence/kernels/probe.log)):

| Step | ms/step | Warm-up | 24,000 steps |
| --- | ---: | ---: | --- |
| CUDA graph, PyTorch ops | 3.14 | a few s | 2.02%, 75.2 s |
| CUDA graph + `torch.compile` on the loss | 1.54 | 87 s | 1.91%, 37.0 s |
| CUDA graph + these Triton kernels | **1.00** | **6 s** | **1.79%, 24.0 s** |

* **Correct:** against the PyTorch reference, the kernels' outputs and every gradient agree to 2e-7 relative error or better, at every layer width. The in-kernel noise has mean 0.0006 and standard deviation 0.2998 (target 0.3) and is independent from step to step.
* **Sandbox:** the sandboxed worker's home is not writable, so the entry points Triton's cache at `/tmp`. A one-kernel smoke test ([source](evidence/kernels/sandbox_smoke.py), [output](evidence/kernels/sandbox-smoke.log)) passed difficulty 1 with the sandbox on before the kernels were written.
* **Why not `torch.compile`:** it compiles for about 80 s in every fresh worker, and `/tmp` is wiped between calls, so later warm-ups (90 s, `LATER_WARMUP_MAX_CALL_MS`) cannot also fit a full fit. A harness-owned inductor cache per run would make the allow-listed decorator usable.

## Choices and snags

1. **Difficulty 5 at 36,000 steps.** The recipe's 24,000 steps average 1.870%, 0.03 points under the band; the study estimates such a run passes about 64% of the time. The study's 12,000 -> 24,000 comparison improved all 11 draws, so 36,000 steps was run for margin without a separate accuracy check: 1.71% over the 33 calls. Time left for a slower board: a 20% lower clock would put a call at about 44 s, under the 60 s limit.
2. **Energy does not fall with time.** The fused step keeps the GPU busier, so the Triton difficulty 4 entry uses about as much energy per call as the graph-only difficulty 3 entry while taking 30% less time.
3. **Source size.** The files are 19,792 bytes against the 20,480-byte limit, mostly docstrings and kernel code.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/ladder-triton-20260928/ladder_d3.py:ladder --difficulty 3 --runs 3
python run_modal.py submissions/ladder-triton-20260928/ladder_d4.py:ladder --difficulty 4 --runs 3
python run_modal.py submissions/ladder-triton-20260928/ladder_d5.py:ladder --difficulty 5 --runs 3
modal run submissions/ladder-triton-20260928/verify_kernels.py
```

The runs used a copy of `run_modal.py` with the function timeout at 2,700 s (the scorer's `RUN_MAX_S`) and files named `ladder_t_d3.py` etc.; the source hash in each record matches the file here.

## Cost

Modal's billing API reported **$7.62** for all of this work, graph-only entry included, when this was committed ([billing](evidence/modal-billing.json)): difficulty 5 record $3.25, difficulty 4 record $1.37, graph-only difficulty 3 record $1.59 and an earlier passing run $0.50, probes $0.82, sandbox smoke test $0.09. The two Triton difficulty 3 sets (5,000 and 6,500 steps) had not been billed yet; at the observed rate they come to about $1.30 more.

SHA-256:

- `ladder_d3.py`: `15fa0873e11f06098d89eb721e8731f0cd8a0b3f5a4c95b27dc74e976b5f1341`
- `ladder_d4.py`: `ec161a3ad3a98421cda56175a7c572b1bba87ea856c3660711f95e0732c2a670`
- `ladder_d5.py`: `4f55374046b3094124c52f4a9592482cb033a4d1b08094c7904ae29cd4b26979`
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`
