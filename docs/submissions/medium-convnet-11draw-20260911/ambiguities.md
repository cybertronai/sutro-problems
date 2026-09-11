# Ambiguities and problems: MNIST-medium ConvNet submission

This is separate from the measured-results report. It identifies incomplete
requirements and interpretation limits for this particular submission.

## Exact theoretical scoring remains incomplete

The frozen learner is implemented and evaluated in PyTorch on an A100. It has
not been translated into a complete Dally v4 program. Consequently **model
time, model energy, model area, and time to score are unavailable**. Passing
the accuracy threshold and publishing an entry do not establish full benchmark
compliance. The submission is an attempt for review with this requirement open.

The v4 opcode list contains arithmetic, comparisons, selection, bitwise
operations, copies, constants, and tape I/O. It does not supply GELU, square
root, exponential, logarithm, sine, or cosine as primitives. BatchNorm, AdamW,
cross-entropy, and augmentation use these operations. The FP64 ensemble also
does not fit into one 32-bit model cell. An implementation must explicitly
lower these functions, its fixed random schedules, and all data movement.
GPU reductions and fused arithmetic additionally require numerical validation.

A tensor-level operation count, GPU memory measurement, or parameter count
cannot stand in for a v4 memory-access score. The proposed compact tensor-loop
representation is a design, not a completed translation. Any changed numerical
learner must be evaluated again under a declared protocol before inheriting an
accuracy claim. The detailed feasibility audit gives the remaining work.

## Dataset sampling and development history

Small and medium use random subsets of the original 60,000 training examples.
The present evaluation uses **11 independent draws**, with both training and
test subsets resampled. Rows do not overlap within a draw; overlap across draws
is expected. Every learner is reset and receives only its own permitted arrays.
The reported SD is the sample SD of 11 dataset accuracies, in percentage points;
it is not an uncertainty interval or SD across ensemble members.

The architecture and epoch count were developed in the previous search. The
choice to submit the three-model ensemble was informed by that study's
single-dataset test result. This is disclosed rather than presented as a choice
made before all historical test observations. These 11 dataset seeds and the
entire learner procedure were frozen before any new test results were opened.
No prior learned weights or optimizer state were used.

Dataset indices and source files are publicly reproducible. The freeze and
separate evaluator provide an auditable development record, not a hidden
evaluation service or a cryptographic guarantee against every possible form of
outside knowledge. The actual implementation does not access test labels while
fitting or predicting.

## Accuracy threshold and rounding

The current medium target is **98% mean**, inclusively. It requires **64,680
correct predictions out of 66,000**. The former 98.14% policy is superseded;
historical reports retain their original threshold comparisons. The criterion
applies to the unrounded mean, not to every draw and not to mean minus SD.
All 11 results are included. Mean and SD are displayed with one decimal place;
exact counts and the full-precision summary remain in the data files.

## A100 measurement boundaries

Performance is measured on **draw 00**, a predeclared representative draw. It
does not estimate average runtime or energy over all 11 sampled datasets. The
reported task includes training all three members afresh and forming their
ensemble; inference-only timing would describe a different task.

The unchanged learner also computes per-epoch training diagnostics, transfers
arrays and outputs, hashes tensors, and serializes checkpoints. These costs
are included in the measured invocation. They make this a reproduction of the
submitted implementation, not an optimized estimate of its indispensable work.
Container startup, imports, input delivery to the container, and result upload
to the local machine lie outside the measured invocation. The report records
the precise measured boundary and verification results.

NVML measures the GPU board's cumulative energy counter, including memory and
GPU idle periods during host work. Host CPU and network energy are excluded.
Paired idle subtraction depends on GPU temperature and power state, so both
the before and after baselines, their sensitivity, and gross energy are retained.
Three repeat tasks measure variation on one A100 allocation, not hardware-wide
reproducibility. A warmup precedes measured tasks; every task still starts with
fresh model and optimizer state.

The container requests four CPU cores, but its host CPU model was not recorded.
Host work contributes to elapsed time, so this limits performance reproduction
across hosts. GPU power limits and clocks use unmodified platform defaults.

CUDA-event elapsed time includes intervals in which the GPU waits for the host
between work submissions; it is not a sum of kernel-active durations. Wall
time is also retained. Repeated-task wall time is the primary complete-task
runtime in the entry, with the exact boundary stated in the results report.

## Units and area

Theoretical and A100 time use **ms** and energy uses **mJ**. Thus
`energy_mJ = average_power_W × time_ms`. Time to score uses **s**, and model
area uses **mm²**. Displayed costs use two significant figures. Time and energy
remain different physical quantities even when expressed on comparable scales.

Model area means occupied scratch-cell area under a declared placement and
lifetime policy. It is not A100 die area, allocated GPU memory, parameter count,
or the enclosing rectangle around scattered cells. The 1,404,618 parameter
words per member alone do not establish peak training workspace or a model area
score. No such score is claimed here.

## Evidence

- [Measured results and reproduction](index.html)
- [Detailed v4 feasibility audit](scoring-feasibility.md)
- [Frozen protocol](protocol.json)
- [Human-readable session](session.html)
