# A100 measurement notes

The first A100 attempt completed successfully with the frozen 81 → 512 → 10
network, 200 epochs, learning rate 0.1, batch size 30, and seed 101. Accuracy is
**5,755/6,000 (96%)**, below both the requested 98% goal and repository 98.14%
requirement. No additional model or seed tuning followed test evaluation.

| Quantity | Accuracy | Result |
| --- | ---: | ---: |
| Mean complete-task time | 96% (5755/6000) | 4,700 ms |
| Mean idle-adjusted board energy | 96% (5755/6000) | 150,000 mJ |
| Mean raw board energy | 96% (5755/6000) | 430,000 mJ |
| Before-only / after-only idle sensitivity envelope | 96% (5755/6000) | 150,000–160,000 mJ |

All headline values are means of three trials, rounded to two significant
figures. Exact means, medians, sample deviations, and counters remain in JSON.
Each trial executes three complete tasks, for about 14,000 ms of active work.
Each bracketing idle measurement lasts 3,000 ms and follows a 3,000 ms settling
gap. Paired idle powers differed by at most 0.10 W across the three trials.

A task runs an initialization graph once, an entire-epoch graph 200 times, and
an inference graph once: 202 actual graph replays and 160,004 kernel launches.
Each trial therefore contains 606 graph replays. The captured graphs contain
804 kernel nodes, independent of the epoch count. Every task resets all
parameters, prepares inputs and one-hot targets, executes every training update,
and predicts all 6,000 labels. Inference also materializes all 60,000 scores.

Transfers, allocation, JIT compilation, graph capture, verification, cold
start, and host CPU energy are excluded. GPU-resident repetitions can reuse
caches. This is complete-task steady-state throughput, not launch latency or
an optimized A100 performance bound.

All 47,114 learned parameter words, 60,000 score words, and 6,000 predictions
match the CPU artifacts exactly for the direct task, the captured task, and
after all timed tasks. Random-query verification also passes. Changed-label
verification uses two complete minibatches and 17 queries, not a second full
200-epoch mutation experiment; it verifies changed parameter and score words.

CPU result artifacts are host-only verification oracles. They are never copied
to GPU learner buffers or substituted for learned parameters. GPU inputs are
the three allowed data arrays and initial literals derived only from the seed.

The independent audit recomputed every timing conversion, complete-task and
graph count, NVML counter delta, and idle subtraction. Source, configuration,
input, oracle, and all eight PTX hashes match. The exported kernels contain no
FP32 fused multiply-add or flush-to-zero instructions. The eight PTX files total
about 41,000 bytes; runtime reduction loops keep their size bounded.

See the main report and README for the exact reproduction commands.

- [GPU implementation](gpu_benchmark.py)
- [Raw measurements](gpu_results.json)
- [Independent GPU audit](gpu_validation.json)
- [Full report](index.html)
