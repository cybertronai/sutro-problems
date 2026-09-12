# Historical MNIST-Small Panel Optimization

**Draft research PR for the previous specification.** The measurements use
600/600 images and the former single-core-with-tape model. Upstream now requires
1,000/1,000 images, 67% mean accuracy and spatial-computer grid scoring. This PR
does not claim current qualification and adds its row only to the historical
table, leaving spatial-grid columns unmeasured.

## Proposed Title

Add historical MNIST-small panel optimization with paired A100 results

## Summary

- Optimize all seven matrix products of the existing 9-32-10 ReLU MLP through
  persistent panels, operand reuse, a shared minibatch cache and v4 placement.
- Keep architecture, seed 101, 300 epochs, batch 30, learning rate 0.2 and
  ascending unfused FP32 arithmetic unchanged.
- Reduce full-task Dally energy by 14.14%, with 1.31% more modeled time and
  1.98% more scratch space.
- Reduce paired A100 time by 20.34% and idle-adjusted board energy by 17.95%.
- Pass a fresh, predeclared eleven-draw evaluation: 4292/6600 correct,
  65.0% +/- 2.1 pp. All fresh-draw outputs equal the original learner bit-for-bit.

## Headline Metrics

| Accuracy | Time | Energy | Area | Time to score | Time on A100 | Energy on A100 |
|---|---:|---:|---:|---:|---:|---:|
| 65.0% +/- 2.1 pp | 94 ms | 0.098 mJ | 0.021 mm2 | 0.031 s | 57 ms | 1,800 mJ |

Rounded costs use two significant figures. The report and JSON files retain
exact counts, measurements, individual trials and metric definitions.

## Validation

- Original Triton baseline kernels preserved by source-hash/AST checks.
- Every GPU graph independently verified to contain 24,004 kernel nodes.
- Full 300-epoch GPU execution matches all 650 parameters, 6,000 scores and
  600 predictions against the ordered CPU reference.
- Changed-query and changed-label graph checks pass before and after timing.
- Raw NVML energy arithmetic, source/PTX hashes and saved GPU arrays verified.
- Bounded actual IL executions match parameters, scores, predictions and all
  address-level access counts; exact whole-program scores reproduced locally.
- Eleven new draws were predeclared after the final policy was frozen; all
  predictions were saved and hashed before test-label scoring.
- The learner ignores even an object-typed poisoned `test_labels` archive entry.

## Measurement Scope

The A100 numbers are medians of three matched cyclic rounds on one A100-SXM4-40GB.
Each replay resets parameters and performs complete learning and prediction.
The primary result uses separate aligned buffers like the original baseline.
Transfers, allocation, JIT, graph capture, validation, cold start and CPU energy
are excluded. The GPU implementation adapts panel reuse rather than simulating
Dally physical distances.

The supporting model `no_slowdown` profile performs worse on GPU and is retained
as a negative comparison. Two initial compilation failures and a preliminary
shared-buffer run are recorded but do not contribute to headline metrics.

## Intended Change Scope

- Add `mnist/submissions/mlp-panels-v4-20260911/` with portable source, configs,
  executable IL, reports, evidence, reproduction commands and verification.
- Add `docs/submissions/mlp-panels-v4-20260911/index.html` as a static report.
- Add one row under historical MNIST-small in `mnist/README.md`, preserving existing rows,
  task definitions, requirements and the competition diagram.
- Do not include unrelated experimental directories or generated data caches.

## Review Notes

The affine-v4 representation and inherited scoring conventions remain subject
to review. Full IL numerical execution is bounded; full task costs are statically
aggregated. Three GPU trials on one device do not establish population variance.
SecurityQQ provided research direction and is submitting this study.
OpenCode performed implementation, measurements, verification and documentation;
the original MLP and matmul contributors are credited in the report.

## Links

- [Standalone report](https://github.com/SecurityQQ/sutro-problems/blob/securityqq/mnist-panel-energy-historical/mnist/submissions/mlp-panels-v4-20260911/report.md)
- [Reproduction instructions](https://github.com/SecurityQQ/sutro-problems/blob/securityqq/mnist-panel-energy-historical/mnist/submissions/mlp-panels-v4-20260911/README.md)
- [Requirement checklist](https://github.com/SecurityQQ/sutro-problems/blob/securityqq/mnist-panel-energy-historical/mnist/submissions/mlp-panels-v4-20260911/REVIEW_CHECKLIST.md)
- [Fresh 600/600 accuracy evidence](https://github.com/SecurityQQ/sutro-problems/blob/securityqq/mnist-panel-energy-historical/mnist/submissions/mlp-panels-v4-20260911/evidence/accuracy/accuracy.json)
- [Primary GPU measurements](https://github.com/SecurityQQ/sutro-problems/blob/securityqq/mnist-panel-energy-historical/mnist/submissions/mlp-panels-v4-20260911/evidence/gpu/results.json)
