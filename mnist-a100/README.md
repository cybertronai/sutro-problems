[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems/tree/main/mnist-a100)

# MNIST end-to-end on an A100

Architecture + optimizer + kernel codesign

- Learn to read digits from 10,000 labelled 9x9 MNIST images, then label 10,000 more, from scratch, on one A100.
- Every call runs in a fresh, sandboxed process on a fresh draw, whitened and secretly rotated.
- Ranked by time per call. Energy per call above idle is reported beside it ([how](energy/README.md)).

<img width="1084" height="392" alt="Screenshot 2026-09-29 at 5 14 36 PM" src="https://github.com/user-attachments/assets/51940abe-372c-4434-8003-d61cc1b4733c" />


## API

```python
import mnist

def my_method(train_x, train_y, test_x):
    # train_x (10000, 60) float32, train_y (10000,) int64 in 0..9, test_x (10000, 60) float32, on the GPU
    ...
    return labels  # (10000,) integer tensor on the GPU

if __name__ == "__main__":
    ms = mnist.score(my_method, difficulty=1)  # 1 (loosest) to 5; ms per call, or raises mnist.Disqualified
```

```bash
python example.py                                            # on your own GPU
python run_modal.py example.py:mlp --difficulty 1 --runs 3   # official: three Modal A100-80GB runs, median
```

The rules are in [`mnist.py`](mnist.py)'s docstring.

Each row is a separate submission by the listed entrant; source attribution and
verification scope are in its report. Rows with one verification run are
provisional. ★ identifies an established record.

## Difficulty = 1 (error ≤ 5.40%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-25 | 29,662 | 700.1 | [py](example.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `example.py` baseline: MLP 60-1024-1024-10, 400 steps |
| 2026-09-26 | 2,648 | 191.3 | [py](energy/entries/mlp_k1_w1024_s100_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-1024-1024-10, 100 steps |
| 2026-09-26 | 2,131 | 61.7 | [py](submissions/graph-mlp-20260926/fast_mlp.py), [report](submissions/graph-mlp-20260926/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-256-256-10, 200 steps in a CUDA graph |
| 2026-09-30 | 3,634 | 351.0 | [py](submissions/improved-20260930/mlp_d1.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP, 200 steps, TF32 and fused AdamW; one sandboxed run |
| 2026-10-08 | 530 | 6.7 | [py](submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py), [report](submissions/native-mlp-d1-20261007/README.md) | [@sjbaebae](https://github.com/sjbaebae) | Two 60-256-256-10 MLPs, 30 updates, three-neighbor voting; median of three sandboxed Modal runs |
| 2026-10-12 | 60 | 1.6 | [py](submissions/nystrom-fisher-d1-20261012/submission.py), [report](submissions/nystrom-fisher-d1-20261012/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Nyström RBF kernel ridge in a Fisher metric, 480 class-wise k-means landmarks, one Cholesky solve; no backprop; median of three independent reviewer sandboxed Modal runs ★ best |

## Difficulty = 2 (error ≤ 3.40%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-26 | 200,870 | 927.3 | [py](energy/entries/mlp_k16_w1024_s400_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 16 MLPs 60-1024-1024-10, 400 steps |
| 2026-09-26 | 14,436 | 252.1 | [py](energy/entries/mlpg_k4_w256_s800_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 4 MLPs 60-256-256-10, 800 steps in a CUDA graph |
| 2026-09-29 | 8,071 | 38.3 | [py](submissions/kernel-pcg-20260929/kernel_pcg.py), [report](submissions/kernel-pcg-20260929/README.md) | [@islamborghini](https://github.com/islamborghini) | Non-neural RBF kernel ridge, one learned metric update, 16-step Nyström-preconditioned CG |
| 2026-09-30 | 132,635 | 761.9 | [py](submissions/improved-20260930/mlp_d2.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 8 MLPs, 500 steps, EMA 0.992; one sandboxed run |
| 2026-10-08 | 1,333 | 14.6 | [py](submissions/native-mlp-d2-20261007/mlp_neighbor3_register_d2.py), [report](submissions/native-mlp-d2-20261007/README.md) | [@sjbaebae](https://github.com/sjbaebae) | Two 60-256-256-10 MLPs, 120 updates, three-neighbor voting; median of three sandboxed Modal runs |
| 2026-10-12 | 603 | 4.6 | [py](submissions/rfm-pcg-d2-20261012/submission.py), [report](submissions/rfm-pcg-d2-20261012/README.md) | [@spartanhaden](https://github.com/spartanhaden) | RBF kernel ridge with one learned (AGOP) metric, 5 Nyström-preconditioned CG steps; no backprop; median of three independent reviewer sandboxed Modal runs |
| 2026-10-19 | 208 | 3.2 | [py](submissions/fisher-pcg-d2-20261019/submission.py), [report](submissions/fisher-pcg-d2-20261019/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Fisher-metric RBF ridge on a compressed weighted training set, triangular FITC-style preconditioning and extrapolated PCG; ports sigkillme0's D3 method; median of three independent reviewer sandboxed Modal runs ★ best |

## Difficulty = 3 (error ≤ 2.70%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 1,159,909 | 15,818.8 | [py](submissions/ladder-graph-20260928/ladder.py), [report](submissions/ladder-graph-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder 60-1000-500-250-250-250-10, 5,000 steps in a CUDA graph |
| 2026-09-28 | 676,850 | 6,556.9 | [py](submissions/ladder-triton-20260928/ladder_d3.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 6,500 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 259,847 | 1,895.4 | [py](submissions/ladder-fast-20260929/ladder_d3.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 1,200 steps, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 201,792 | 1,443.3 | [py](submissions/ladder-sampled-20260929/ladder_d3.py), [report](submissions/ladder-sampled-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 900 steps, labelled rows sampled by loss, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-30 | 253,761 | 1,732.2 | [py](submissions/improved-20260930/ladder_d3.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 1,100 steps; 8.3% fewer updates; one sandboxed run |
| 2026-10-05 | 151,404 | 1,157.1 | [py](submissions/ladder-kernel-tuned-20261003/ladder_d3.py), [report](submissions/ladder-kernel-tuned-20261003/README.md) | [@islamborghini](https://github.com/islamborghini) | Sampled Ladder, batch 1,000, 800 steps, TF32; tuned forward and backward Triton kernels; median of three sandboxed runs |
| 2026-10-08 | 98,226 | 416.1 | [py](submissions/native-mlp-d3-20261007/graph1250-repair/submission.py), [report](submissions/native-mlp-d3-20261007/graph1250-repair/README.md) | [@sjbaebae](https://github.com/sjbaebae) | Eight 60-256-256-256-10 RMSNorm/ReLU MLPs, 1,250 updates, five-neighbor voting and three-step query graph propagation; median of three sandboxed Modal runs |
| 2026-10-12 | 8,839 | 51.8 | [py](submissions/polar-kernel-d3-20261012/submission.py), [report](submissions/polar-kernel-d3-20261012/README.md) | [@sjbaebae](https://github.com/sjbaebae) | Spherical direction/radius RBF, two tangent-gradient metric updates; derived from PR #116; median of three independent reviewer sandboxed Modal runs |
| 2026-10-12 | 2,362 | 18.1 | [py](submissions/rfm-hknn-d3-20261012/submission.py), [report](submissions/rfm-hknn-d3-20261012/README.md) | [@spartanhaden](https://github.com/spartanhaden) | RBF kernel ridge, two AGOP metric updates, HKNN local-hull correction on the 512 lowest-margin rows; no backprop; median of three independent reviewer sandboxed Modal runs |
| 2026-10-14 | 7,899 | 44.0 | [py](submissions/kernel-rbf-d3-20261009/kernel_d3_cg_graph.py), [report](submissions/kernel-rbf-d3-20261009/README.md) | [@islamborghini](https://github.com/islamborghini) | RBF kernel ridge, two analytic metric updates, CUDA-graph Nyström-preconditioned CG and local class geometry; median of three independent reviewer sandboxed Modal runs |
| 2026-10-18 | 320 | 5.2 | [py](submissions/kernel-d3-20261018/submission.py), [report](submissions/kernel-d3-20261018/README.md) | [@sigkillme0](https://github.com/sigkillme0) | Learned-metric RBF kernel with fused Triton PCG, compressed kernel tiles, and local-hull correction; builds on Haden's RFM/HKNN learner; median of three independent reviewer sandboxed Modal runs ★ best |

## Difficulty = 4 (error ≤ 2.30%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 1,143,255 | 11,094.6 | [py](submissions/ladder-triton-20260928/ladder_d4.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 11,000 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 547,382 | 3,784.2 | [py](submissions/ladder-fast-20260929/ladder_d4.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 2,400 steps, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 400,682 | 2,882.5 | [py](submissions/ladder-sampled-20260929/ladder_d4.py), [report](submissions/ladder-sampled-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 1,800 steps, labelled rows sampled by loss, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-30 | 460,763 | 3,481.3 | [py](submissions/improved-20260930/ladder_d4.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 2,200 steps; 8.3% fewer updates; one sandboxed run |
| 2026-10-12 | 63,113 | 580.1 | [py](submissions/ladder-kernel-d4-20261012/submission.py), [report](submissions/ladder-kernel-d4-20261012/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Ladder MLP 60-1024-512-256-10 on AGOP-metric inputs, 560 Adam steps, kernel-ridge teacher; median of three independent reviewer sandboxed Modal runs |
| 2026-10-18 | 42,617 | 219.2 | [py](submissions/rfm-ensemble-d4-20261017/submission.py), [report](submissions/rfm-ensemble-d4-20261017/README.md) | [@islamborghini](https://github.com/islamborghini) | Five RBF classifiers with three analytic AGOP metric updates each, Nyström-preconditioned CG, transductive refits and HKNN affine-hull correction; no backprop; median of three independent reviewer sandboxed Modal runs |
| 2026-10-19 | 14,754 | 97.0 | [py](submissions/hybrid-tangent-d4-20261019/submission.py), [report](submissions/hybrid-tangent-d4-20261019/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Two RFM teachers distilled into three tangent-jittered MLP students with cross-teaching and HKNN; one full-call CUDA graph; median of three independent reviewer sandboxed Modal runs ★ best |

## Difficulty = 5 (error ≤ 1.90%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 4,410,730 | 36,201.2 | [py](submissions/ladder-triton-20260928/ladder_d5.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 36,000 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 2,059,576 | 14,130.6 | [py](submissions/ladder-fast-20260929/ladder_d5.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 9,000 steps, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-30 | 1,824,760 | 13,264.3 | [py](submissions/improved-20260930/ladder_d5.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 8,400 steps; 6.7% fewer updates; one sandboxed run |
| 2026-10-12 | 638,234 | 3,335.4 | [py](submissions/tangent-crossteach-d5-20261012/submission.py), [report](submissions/tangent-crossteach-d5-20261012/README.md) | [@likelytobelaura](https://github.com/likelytobelaura) | Eight RMSNorm/GELU MLPs, Muon/AdamW, 2,500 updates, local tangent jitter, disjoint-pool cross-teaching and graph readout; median of three independent reviewer sandboxed Modal runs |
| 2026-10-12 | 226,352 | 1,818.1 | [py](submissions/ladder-kernel-d5-20261012/submission.py), [report](submissions/ladder-kernel-d5-20261012/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Ladder MLP 60-1024-512-256-10 on AGOP-metric inputs, 2,300 Adam steps, kernel-ridge teacher; median of three independent reviewer sandboxed Modal runs |
| 2026-10-14 | 343,152 | 1,915.4 | [py](submissions/tangent-relu-d5-20261012/submission.py), [report](submissions/tangent-relu-d5-20261012/README.md) | [@likelytobelaura](https://github.com/likelytobelaura) | Eight width-256 RMSNorm/ReLU MLPs, Muon, tangent jitter, cross-teaching and graph readout; median of three independent reviewer sandboxed Modal runs |
| 2026-10-18 | 154,289 | 897.5 | [py](submissions/ladder-kernel-tuned-d5-20261017/submission.py), [report](submissions/ladder-kernel-tuned-d5-20261017/README.md) | [@islamborghini](https://github.com/islamborghini) | Haden's kernel-taught Ladder tuned to batch 2,000 and 800 steps; architecture, teacher and optimizer unchanged; median of three independent reviewer sandboxed Modal runs ★ best |
| 2026-10-19 | 112,581 | 632.1 | [py](submissions/ladder-k2-d5-20261019/submission.py), [report](submissions/ladder-k2-d5-20261019/README.md) | [@spartanhaden](https://github.com/spartanhaden) | Kernel-taught Ladder 60-1024-512-256-10, batch 2,500 × 600 Adam steps, local tangent jitter, teacher-fallback guard, fused Triton combinator backward; median of three sandboxed Modal runs |
