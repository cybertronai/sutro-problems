# Reproduce (from this directory)

Raw data: official MNIST gz files; set `SUTRO_RAW=/path/to/raw` if not found by
walking up the workspace. Deps: numpy (CPU) and modal + torch/triton for GPU.

```sh
python run.py prepare    # draws, indices and input hashes (no training)
python run.py freeze     # train + hash predictions (evaluation labels untouched)
python run.py score      # verify every prediction hash FIRST, then count
python verify.py         # re-check manifests, inputs and accuracy consistency
modal run gpu_benchmark.py   # fused A100 session (uses generated/*.npz)
```

`generated/adam11-payload.npz` and `generated/adam11-expected.npz` are the
draw inputs and ordered-CPU expectations used by the GPU session; they are
regenerable from `run.py freeze` plus the draw builder in `sources/`.
