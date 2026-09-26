#!/usr/bin/env python3
"""Time every MLP in mlp_family.grid() on one Modal A100-80GB, harness style.

    /tmp/penv/bin/python sweep.py --out results/sweep.json
    /tmp/penv/bin/python sweep.py --only mlp-k16-w1024-s4000-b128 --out results/check.json

One container runs every configuration, so all times in one output file come
from the same host: the same entry's time differs by up to a quarter between
Modal A100-80GB hosts (popcorn3/README.md, "Timing spread between containers").
The draws are dev draws from popcorn3's reference.Pool with the 60-dim release,
generated from --dev-seed inside the container; they never touch the study's
final seeds. Per configuration: one untimed warm-up of at most 50 steps on a
separate draw (compilation, cuBLAS heuristics and allocations depend on shapes,
not on the step count), then one timed call per dev draw: copy the draw into
fixed CUDA tensors, synchronize, time custom_kernel, synchronize. That is the
harness's protocol without its L2 flush, which is negligible at these lengths.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13").uv_pip_install(
    "numpy~=2.3", "torch==2.12.0")
if modal.is_local():  # Modal imports this file again inside the container, where these paths do not exist
    REPO = HERE.parents[3]
    image = (image.add_local_dir(REPO / "popcorn3" / "problems" / "sutro", "/opt/sutro")
             .add_local_file(HERE / "mlp_family.py", "/opt/mlp/mlp_family.py"))
app = modal.App("release-cutoffs-mlp-timing", image=image)


@app.function(gpu="A100-80GB", timeout=7200)
def sweep(configs: list, draws: int, dev_seed: int) -> dict:
    import types

    sys.path[:0] = ["/opt/sutro", "/opt/mlp"]
    import numpy as np
    import torch

    import eval as harness  # the scorer package's run diagnostics
    import mlp_family
    import reference

    rng = np.random.default_rng(dev_seed)
    pool = reference.Pool("mnist", *reference.load_pool("mnist", 9), rng)
    warm = pool.draw(rng, 10000, 10000, release_dims=60)
    dev = [pool.draw(rng, 10000, 10000, release_dims=60) for _ in range(draws)]
    device = torch.device("cuda")
    fixed = []

    def load(draw):
        tensors = (torch.from_numpy(draw["train_x"]), torch.from_numpy(draw["train_y"]),
                   torch.from_numpy(draw["test_x"]))
        if not fixed:
            fixed.extend(t.to(device) for t in tensors)
        for target, value in zip(fixed, tensors):
            target.copy_(value)
        torch.cuda.synchronize()
        return tuple(fixed)

    def module(text):
        entry = types.ModuleType("entry")
        exec(compile(text, "entry.py", "exec"), entry.__dict__)
        return entry

    host = harness.host_description()
    sampler = harness.ClockSampler(7200)
    started, rows = time.time(), []
    for config in configs:
        k, width, steps, batch, graphed = tuple(config) + (False,) * (5 - len(config))
        entry = module(mlp_family.source(k, width, steps, batch, graphed))
        if graphed:  # capture in the untimed warm-up call, as the harness allows
            entry.custom_kernel(load(warm))
        else:  # compilation and allocations depend on shapes, not on the step count
            module(mlp_family.source(k, width, min(steps, 50), batch)).custom_kernel(load(warm))
        torch.cuda.synchronize()
        correct, ms = [], []
        for draw in dev:
            data = load(draw)
            began = time.perf_counter()
            labels = entry.custom_kernel(data)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - began) * 1e3)
            correct.append(reference.check_implementation(draw["test_y"], labels.cpu().numpy()))
        row = {"name": mlp_family.name(k, width, steps, batch, graphed), "k": k, "width": width, "steps": steps,
               "batch": batch, "graphed": graphed, "correct": correct, "ms": ms,
               "error_pct": 100.0 * (1.0 - sum(correct) / (10000.0 * len(correct))),
               "mean_ms": sum(ms) / len(ms)}
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("name", "error_pct", "mean_ms")}), flush=True)
    return {"rows": rows, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "host": host, "gpu_clock": sampler.stop(), "dev_seed": dev_seed, "draws": draws,
            "wall_seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", help="comma-separated configuration names (default: the whole grid)")
    parser.add_argument("--family", choices=["eager", "graphed", "both"], default="eager")
    parser.add_argument("--extra", default="", help="configuration names appended to the chosen family, e.g. "
                        "eager references that calibrate one container's speed against another's")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--dev-seed", type=int, default=20260925)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    import mlp_family

    configs = {"eager": mlp_family.grid(), "graphed": mlp_family.graph_grid(),
               "both": mlp_family.grid() + mlp_family.graph_grid()}[args.family]
    if args.only:
        configs = [mlp_family.parse(label) for label in args.only.split(",")]
    configs += [mlp_family.parse(label) for label in args.extra.split(",") if label]
    print(f"{len(configs)} configuration(s), {args.draws} dev draws each, one A100-80GB container", flush=True)
    with modal.enable_output(), app.run():
        result = sweep.remote(configs, args.draws, args.dev_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: {len(result['rows'])} rows on {result['device']}, {result['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
