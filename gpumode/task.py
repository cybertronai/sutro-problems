"""Input and output types for the MNIST-medium time leaderboard."""

from typing import TypedDict

import torch

# (train_x, train_y (N,) int64, test_x), all on the evaluation device. The SAME
# three tensor objects are passed on every call; only their contents change, so
# CUDA graph capture is legal.
#
# train_x and test_x carry the draw's *released* features, whose shape follows
# the case field release_dims:
#
#   release_dims = D > 0 (60 in every generated task.yml)
#       (N, D) and (Q, D) float32, z = Q W (x - mu): the draw's images whitened
#       onto the top D principal directions of its own training rows and then
#       rotated by a Haar-random orthogonal matrix that is secret and fresh per
#       draw. Zero mean and identity covariance over the training rows; 99.9%
#       of the values fall inside +-4.6, but the tails are heavy and unbounded
#       (about 2 values in 10,000 exceed 6 in magnitude, and the largest in a
#       10,000-row MNIST draw runs to 10-12), so do not size a fixed-point or
#       int8 scale from a nominal range. No spatial structure either: there is
#       no pixel grid to convolve over, and neighbouring columns are not
#       neighbouring pixels.
#
#   release_dims = 0
#       (N, 1, size, size) and (Q, 1, size, size) float32 in [0, 1]: the raw
#       9x9 pixels, exactly as harness 1.1.1 released them.
#
# A submission that starts with ``x = train_x.reshape(train_x.shape[0], -1)``
# and takes its feature count from ``x.shape[1]`` works under both.
input_t = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Predicted labels (Q,), any integer dtype, on the evaluation device.
output_t = torch.Tensor


class TestSpec(TypedDict):
    """Every field a case line in task.yml may set.

    size               image side after box-area downsampling (9 for medium)
    release_dims       width of the linear release z = Q W (x - mu); 0 releases
                       the raw (N, 1, size, size) pixels instead
    train              training examples per draw
    test               test examples per draw
    error_bp           accuracy band: maximum mean error in basis points
    draws              fresh draws in leaderboard mode; each draw is one timed call
    bench_draws        draws in benchmark mode (KernelBot gives both modes the
                       same case line, and benchmark is only a dress rehearsal)
    seed               public case seed, combined with the secret POPCORN_SEED
    holdout            1 to run the Fashion-MNIST learning check (leaderboard)
    holdout_draws      how many hold-out calls, interleaved at secret positions;
                       they are timed and ranked like every other call
    holdout_min_bp     maximum hold-out error in basis points (3000 = 70%)
    max_call_ms        a single call longer than this fails the submission
    warmup_max_call_ms the same limit for the untimed warm-up call
    draw_slack_bp      per-draw floor: a single draw may fall this many basis
                       points below the band before the run fails
    dispersion_x10     ten times the largest allowed worst/median duration
                       ratio; an honest learner does the same work every call
    max_source_bytes   size limit on submission.py (rule 2, machine-checked)
    max_literal_bytes  size limit on any one string or bytes literal in it
    test_timeout       the mode timeouts KernelBot enforces, repeated here so
    benchmark_timeout  the evaluator can clamp each per-command deadline to the
    ranked_timeout     budget it has left and always report a failure itself
    """

    size: int
    release_dims: int
    train: int
    test: int
    error_bp: int
    draws: int
    bench_draws: int
    seed: int
    holdout: int
    holdout_draws: int
    holdout_min_bp: int
    max_call_ms: int
    warmup_max_call_ms: int
    draw_slack_bp: int
    dispersion_x10: int
    max_source_bytes: int
    max_literal_bytes: int
    test_timeout: int
    benchmark_timeout: int
    ranked_timeout: int
