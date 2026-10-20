# Fisher-metric kernel ridge with an HKNN readout, in plain PyTorch: difficulty 3

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Our plain-PyTorch d3 learner: @sigkillme0's [#125](../kernel-d3-20261018/README.md) method, re-implemented, with the HKNN readout of our [rfm-hknn-d3](../rfm-hknn-d3-20261012/README.md) and our d2 entry's linear algebra. No hand-written kernels.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 3 --runs 3`) all passed.
- **Median ranked time:** 3.716 ms
- **Median energy above idle:** 468 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.541% | +175 | 2.78% | Fashion 86.71% | 4.183 | 614 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.450% | +275 | 2.71% | KMNIST 96.05% | 3.537 | 468 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.530% | +187 | 2.85% | Fashion 87.00% | 3.716 | 389 |

Compared with the current record, @sjbaebae's [native-hull-cg-d3](../native-hull-cg-d3-20261019/README.md) (#134; reviewer median 4.558 ms, 639 mJ), this entry is **1.23× faster** and uses 27% less energy per call.

`submission.py` is 20,453 bytes, 27 under the size limit, with SHA-256 `eae98d2337442db3c2ed5984e455cf2fc945bfa6846aeaa93d0decb51799d232`, identical to `source_sha256` in all three records (which name it by its working name, `kd3_k.py`). It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

No backprop. The whole call is one CUDA graph.

- **Metric.** LDA plus the Fisher information of a tempered softmax over class Gaussians, with a rank-48 subspace and one natural-gradient step. Rows go on a sphere, plus a log-radius coordinate.
- **Compression.** Class-restricted k-means (24 centres per class, one Lloyd step) keeps the 4,096 lowest-margin rows; the rest become count-weighted centroids.
- **Fit.** RBF kernel ridge by 5 PCG steps plus an extrapolated direction, with a Nyström-plus-diagonal preconditioner.
- **Readout.** Scores come from PyTorch's flash-attention kernel. Up to 512 test rows with a top-2 margin below 0.2 get HKNN distances for their top 3 classes (70 nearest training or confident test rows per class), added as probabilities.

**Changes from our 10-18 port `kc_d3_v3m`:** the d2 changes (side-stream class moments, Woodbury class covariances and preconditioner with fp16 products, batched small Choleskys, L⁻ᵀL⁻¹ precision, one operand padded from 99 to 104 columns); a max-based top-2 margin; and HKNN rows taken as the first 512 with margin below 0.2 by a cumsum compaction, not the 512 lowest (+0.002 points).

In-process on one A100-SXM4 (24 paired draws), the call takes 3.34 ms, against 4.46 ms for #134 (−25%) and 4.01 ms for our 10-18 port.

Certified on 1,056 fresh in-process A100 calls: mean error 2.432%, worst 11-draw split 2.64%, P(3/3) ≈ 0.99. The official runs averaged 2.507%, within noise for 33 draws.

## Credits

- @sigkillme0's [#125](../kernel-d3-20261018/README.md) for the method: the LDA + QDA Fisher metric with its rank-48 subspace and natural-gradient step, the log-radius feature, k-means compression, the FITC-style preconditioner, the extrapolated PCG step, per-row HKNN temperature and warm-up replays. Re-implemented in plain PyTorch.
- HKNN (Vincent & Bengio 2002) and the Nyström-PCG ridge: builds on our [rfm-hknn-d3](../rfm-hknn-d3-20261012/README.md) (#121; see its credits).
- The standard-form Woodbury preconditioner, the batched-path trick, the fp16 preconditioner products and the compaction selection are ours.

## Notes

- Transductive in the readout only: confident test rows serve as HKNN vertices with their predicted labels. Test labels are never used.
- The scorer's clock sets the ranked time. CUDA-event MNIST means were 3.45–3.50 ms; run 1's 4.183 ms came from single 5.4–6.0 ms clock readings.
- Related files developed for this PR also had official sets (3 of 3, unsubmitted): a CUDA hull-kernel variant (`da87aa08`, median 3.698 ms) and our 10-18 #125 ports `kc_d3_v3m` (`8ec2b632`, 4.626 ms) and `kc_d3_final` (`5be577c6`, 6.048 ms). Our rfm-hknn-d3 README lists its own set.
- During development we also ran three single scorer runs of this exact source in our own Modal sandbox (no energy); all passed.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/fisher-hknn-d3-20261020/submission.py:kc_d3 --difficulty 3 --runs 3 --json /tmp/d3-verification
```

Co-developed with Claude.
