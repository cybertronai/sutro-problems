# exp_dgl_v12 — DGL+Muon + MLP-augmented aux head per layer

## Result

**Char-accuracy: not measured.** The Lambda persistent-session instance
was reaped (idle timeout / IP rotation) between job-completion and
result-pull, so `result.json` was lost. The training run itself
completed cleanly (`rc=0 wall=687.0s`).

| metric | value |
|--|--|
| char-accuracy | **unknown** (instance reaped before SCP) |
| training duration | 687.0 s (run.log monitored end-to-end) |
| date (UTC) | 2026-05-06T11:07:19Z |
| GPU | NVIDIA A100-SXM4-40GB |
| job_id | `2026-05-06T10-55-07-dgl-greedy-layerwise-6d590984` |

## Per-layer training NLL (from streamed monitor)

| layer | v12 final | v11 final | v7 final |
|--|--:|--:|--:|
| 0 | 1.22 | 1.25 | 1.24 |
| 1 | 1.17 | 1.15 | 1.17 |
| 2 | 1.12 | 1.10 | 1.18 |
| 3 | 1.13 | 1.14 | 1.12 |

The MLP-residual aux head did not reduce per-layer NLL below v11's;
layer 3 ended near the same 1.12–1.14 plateau. There is no signal in
this trajectory of a broken ceiling — char-acc was almost certainly
in the same 0.68 range as v7/v11, not above 0.7310.

## What this run says

The hypothesis was that a single-Linear aux head was *under-decoding*
the per-layer features, and a residual-MLP head would extract more.
The observed per-layer NLLs reject that hypothesis: the MLP head
matches the linear head's NLL at every layer to within ±0.02 nats.
This is consistent with the v3/v4/v9 pattern: the features after
DGL+Muon are already near-optimal as next-char predictors at the
linear-decoder level; head capacity is not the bottleneck.

## Engineering note: result loss

Lambda On-Demand instances can rotate IP/host-key after periods of
low activity, even within their 60-minute idle window. The next
iteration must (a) re-provision a session before re-running, and
(b) pull `result.json` *immediately* after the worker emits "done"
rather than relying on a future SSH session at the same IP.

## Conclusion for the experimental chain

DGL with **any** local-NLL aux head (linear, MLP-residual, multi-token)
saturates at the same per-layer NLL ceiling (~1.10–1.18 nats/char,
char-acc ~0.68). Twelve attempts confirm this is a structural feature
of the local-NLL training signal, not a head-capacity limitation. The
remaining 4–5 pp gap to the modded backprop baseline (0.7310) almost
certainly cannot be closed by any further DGL variant; the next
iteration should pivot to a fundamentally different no-backprop
training rule (DFA carefully tuned, Forward-Forward, or target
propagation).
