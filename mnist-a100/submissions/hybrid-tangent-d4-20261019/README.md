# Two RFM kernel teachers distilled into tangent-jittered MLP students: difficulty 4

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 4 --runs 3`) all passed.
- **Median ranked time:** 100.563 ms
- **Median energy above idle:** 14,893 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.106% | +213 | 2.28% | Fashion 88.67% | 101.302 | 14,893 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.110% | +209 | 2.33% | KMNIST 96.80% | 100.563 | 14,573 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.095% | +225 | 2.26% | KMNIST 96.63% | 97.944 | 14,943 |

Compared with the current record (219.2 ms, 42,617 mJ; @islamborghini's [RFM ensemble](../rfm-ensemble-d4-20261017/README.md)), this entry is **2.18× faster** and uses 65% less energy per call.

`submission.py` is 20,460 bytes with SHA-256 `102363cfe2825eee42b5ee641731655679825e7087c3d83e19c3340d93859306`, identical to `source_sha256` in all three records (which name it by its working name, `sd4_z400.py`). It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

Plain PyTorch. Each call trains from scratch, and the whole call is one CUDA graph captured in the warm-up.

- **Teacher.** Two of our d3 learners (RBF kernel ridge in an AGOP metric, Nyström-preconditioned CG, HKNN on the 512 lowest-margin test rows). As in the current record, each sees a different cyclic order of the training rows and makes three metric updates. Their softmax(scores / 0.3) are averaged.
- **Students.** Three 60-256-256-10 RMSNorm/ReLU MLPs on AGOP^0.5 inputs.
  - Training: 400 Adam steps (LR 0.015) on 1,024 labelled and 1,024 test rows, with the teacher's soft targets on the test rows.
  - Jitter: each row moves randomly along its top-4 local tangents, the offsets to its 10 nearest rows of its true or teacher class.
  - The output layer starts at zero.
  - Cross-teaching: after steps 130 and 270, test targets become 0.3 teacher + 0.7 students.
- **Predict.** 0.75 students + 0.25 teacher, then HKNN.

Certified before the official set on 4,224 fresh in-process A100 calls: mean error 2.116%, P(3/3) ≈ 0.99.

## Credits

- **Teacher:** our [rfm-hknn-d3](../rfm-hknn-d3-20261012/README.md) on @islamborghini's [kernel-pcg](../kernel-pcg-20260929/README.md).
  - From @islamborghini's [rfm-ensemble-d4](../rfm-ensemble-d4-20261017/README.md) (#123): cyclic row orders, the third AGOP update, softmax(s/0.3) averaging and the `fin()` helper.
  - RFM is from Radhakrishnan et al. 2022; HKNN from Vincent & Bengio 2002.
- **Soft teacher targets and AGOP^0.5 inputs:** our [ladder-kernel-d4](../ladder-kernel-d4-20261012/README.md) (#121).
- **Students:** MLP ensembles, local tangent jitter and cross-teaching from @likelytobelaura's merged [tangent-crossteach-d5](../tangent-crossteach-d5-20261012/README.md) (#118) and [tangent-relu-d5](../tangent-relu-d5-20261012/README.md) (#120). Our tangents come from subspace iteration so they can run inside the graph, and test rows are jittered too.
- **Zero-initialised output layer:** [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (record 5).
- **Held #126 (@sigkillme0):** it also pairs kernel-metric features with MLP students and neighbour jitter at d4, and also zero-inits its head. We used none of its code.

## Notes

- Transductive: test rows are used without labels, as distillation targets, in the neighbour search (with the teacher's class) and in cross-teaching.
- Training is nondeterministic on the GPU, so reruns on the same draw flip a few labels.
- Ranked time varies by host: about 95–110 ms in our runs. Cold warm-up takes 17–22 s.
- Earlier official sets of related, unsubmitted d4 files, all 3/3:
  - 2026-10-19: 117.761 ms (`4490ac9c`) and 163.516 ms (`785a6e7f`);
  - 2026-10-16: 390.294 ms (`1e365269`);
  - 2026-10-05: 581.387 ms (`2e759432`).
- Co-developed with Claude.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/hybrid-tangent-d4-20261019/submission.py:hybrid_d4 --difficulty 4 --runs 3 --json /tmp/d4-verification
```
