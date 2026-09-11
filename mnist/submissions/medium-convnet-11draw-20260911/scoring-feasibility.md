# Scoring feasibility: the frozen ConvNet ensemble

The frozen ensemble passed the **98% mean accuracy requirement**, with
**64,776 / 66,000 correct** across the 11 draws. Its exact v4 translation
remains incomplete.

This is a specification audit and proposed implementation path, not a model
score. No complete v4 translation or theoretical time, energy, area, or scoring
runtime has been produced for this ConvNet. Accuracy and A100 measurements,
when available, establish different results; they cannot fill those missing
model metrics. The accuracy prerequisite for beginning translation is now met.

## Verified instruction and memory rules

The audit uses the same pinned model revision as the earlier submissions:
`26abcca402de647381d31286d42dfbb7a001763d`. Its
[v4 instruction set](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/instruction-sets/v4/README.md)
enumerates `add`, `sub`, `mul`, `div`, `copy`, `and`, `or`, `xor`, `not`, `set`,
`abs`, `cmp`, `select`, `recv`, and `send`. It does not offer arbitrary scalar
or tensor computation. Division is native; square root, reciprocal square
root, exponential, logarithm, error function, trigonometric functions, indirect
loads, and branches are absent. A tensor operation cannot be assigned the
cost of one invented primitive and called a v4 translation.

The pinned
[single-core-with-tape model](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/models/single-core-with-tape/README.md)
requires a fixed, one-to-one placement of 32-bit scratch words. Every listed
source and destination access is charged, including both value operands of a
`select`; only tape instructions are exempt. The geometry contains
32,001 × 16,000 = **512,016,000 available cells**. It imposes no separate area
or execution-time limit.

One frozen C64, three-convolution member has **1,404,618 trainable FP32
parameters**, before gradients, AdamW state, normalization state, activations,
or input buffers. Parameters alone exceed the previous affine prototype's
one-million-word guard. That guard is an implementation restriction, not the
model's capacity. Sequentially training ensemble members could reuse physical
workspace, but a concrete allocation and lifetime plan is still needed. This
parameter count is not a peak-memory or area score.

## Semantics still needing implementation

Convolutions, linear layers, their backward passes, and fixed reductions can
be expressed with bounded scalar loops. The remaining work is substantive:

- GELU and its derivative need explicit error-function and exponential
  implementations. BatchNorm and AdamW require square-root-related arithmetic;
  training normalization, cross-entropy, and augmentation add further work.
  These functions need versioned v4 algorithms, including rounding and edge
  cases, rather than unpriced library calls.
- BatchNorm's training statistics, running statistics, backward pass, and
  evaluation behavior must agree. Optimizer state, bias correction, epsilon,
  weight decay, the fixed learning-rate schedule, and the final partial batch
  must also be included.
- Seeded permutations, dropout masks, and affine transforms are independent
  of pixel values. They could potentially become verified constant schedules,
  with explicit pixel gathers, interpolation, and scratch writes. This requires
  a declared convention for seed-only constants and exact reproduction of the
  random stream. It does not permit extra caller tape inputs or constants
  derived from learned parameters. Otherwise, random generation and address
  selection themselves need allowed-instruction implementations.
- The frozen ensemble averages FP32 logits in **float64**. The model's cells
  are 32-bit words; this requires an explicit multiword implementation or a
  separately evaluated change to the algorithm. A float64 addition cannot
  silently become one 32-bit primitive.

The ISA does not fully specify floating-point formats or rounding. Earlier
submissions declared their own FP32 interpretation. Deterministic PyTorch and
cuDNN execution does not prove agreement with a different reduction order,
fusion policy, or nonlinear approximation. Successful native training alone
therefore cannot certify the translated learner's accuracy.

## Proposed compact IL, not an implemented translator

A viable design would describe tensor shapes and fixed memory regions, then
lower each operator through inspectable scalar templates. Every template must
expand to the allowed v4 instructions; loops compress their representation,
not their charges. Padding, gathers, workspace reuse, parameter updates,
initialization, and complete input/output consumption must remain explicit.
The implementation must validate bounds, initialized reads, aliases, and the
physical placement, then derive exact per-address read/write multiplicities
from those templates. Operator-level FLOP totals or supplied cost certificates
are insufficient.

The larger workspace needs a scalable exact counter rather than simply
raising the old limit while retaining every dense histogram. Compressed
address distributions or tiled aggregation are possible designs, but neither
has been implemented or validated for this learner.

Validation would begin with nonlinear routines and small complete
forward/backward/update programs, comparing numerical results and every
charged access against independent expansion. The complete translated learner
would then need evaluation under the frozen 11-draw protocol. If reductions or
approximations change its predictions, its own accuracy must be reported; the
native PyTorch result cannot be inherited. Until these steps are complete,
the theoretical metrics remain unavailable and complete v4 compliance remains
unestablished.
