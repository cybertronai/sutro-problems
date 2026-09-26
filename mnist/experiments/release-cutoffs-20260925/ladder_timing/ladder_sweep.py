#!/usr/bin/env python3
"""Time the state-of-the-art Ladder trained on all 10,000 labels, at increasing step budgets, on A100-80GB.

    /tmp/penv/bin/python ladder_sweep.py --out results/ladder-sweep.json

The question is the one the MLP sweep answered for MLPs: how long does the recipe that
SETS the thresholds take to REACH each of them when it gets what an entrant gets (10,000
labels, one A100-80GB). The recipe is ../full-batches/selection.json's config (at
N=10,000 full batches change nothing) with target_steps = S for each S in STEPS; its
schedule (warm-up epochs, decay start) scales with S as it does between levels.

One container per dev draw, five in parallel. The draws are the MLP sweep's: popcorn3
reference.Pool, dev seed 20260925, the five draws after the warm-up draw, so each Ladder
row pairs with an MLP row on the same data. Each container runs one untimed 20-step fit
on the warm-up draw, then times every S like a harness call (the whole fit_predict:
build, train, batch-norm calibration, inference on the 10,000 query rows, which the
recipe also uses as unlabelled training data), then the eager MLP configuration
mlp-k16-w256-s3200-b512, which the MLP sweep timed at 4,711 ms, as a host-speed
reference.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
STEPS = [100, 250, 500, 1000, 2000, 4000, 8000, 12000]
REFERENCE_MLP = (16, 256, 3200, 512)
image = modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13").uv_pip_install(
    "numpy~=2.3", "torch==2.12.0")
if modal.is_local():  # Modal imports this file again inside the container, where these paths do not exist
    STUDY = HERE.parent
    image = (image.add_local_dir(STUDY.parents[2] / "popcorn3" / "problems" / "sutro", "/opt/sutro")
             .add_local_file(STUDY / "mlp_timing" / "mlp_family.py", "/opt/mlp/mlp_family.py"))
    for name in ("neural.py", "ladder_model.py", "pmnist_learners.py", "kernels.py"):
        image = image.add_local_file(STUDY / "full-batches" / name, f"/opt/ladder/{name}")
app = modal.App("release-cutoffs-ladder-timing", image=image)


@app.function(gpu="A100-80GB", timeout=3600)
def sweep(draw_index: int, config: dict, steps: list, dev_seed: int) -> dict:
    import types

    sys.path[:0] = ["/opt/ladder", "/opt/sutro", "/opt/mlp"]
    import numpy as np
    import torch

    import eval as harness  # the scorer package's run diagnostics
    import mlp_family
    import neural
    import reference

    rng = np.random.default_rng(dev_seed)
    pool = reference.Pool("mnist", *reference.load_pool("mnist", 9), rng)
    warm = pool.draw(rng, 10000, 10000, release_dims=60)
    draw = [pool.draw(rng, 10000, 10000, release_dims=60) for _ in range(draw_index + 1)][-1]

    def ladder(d, target_steps):
        began = time.perf_counter()
        out = neural.fit_predict(d["train_x"], d["train_y"], d["test_x"], dict(config, target_steps=target_steps),
                                 seed=11, device="cuda", deadline_unix=time.time() + 1500)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        labels = np.asarray(out["logits"]).argmax(1)
        return seconds, reference.check_implementation(d["test_y"], labels)

    host = harness.host_description()
    sampler = harness.ClockSampler(3600)
    ladder(warm, 20)
    rows = []
    for target_steps in steps:
        seconds, correct = ladder(draw, target_steps)
        rows.append({"steps": target_steps, "seconds": seconds, "correct": correct,
                     "error_pct": 100.0 * (1.0 - correct / 10000.0)})
        print(json.dumps({"draw": draw_index, **rows[-1]}), flush=True)

    fixed = [torch.from_numpy(a).cuda() for a in (draw["train_x"], draw["train_y"], draw["test_x"])]
    for label_steps in (min(REFERENCE_MLP[2], 50), REFERENCE_MLP[2]):  # warm-up, then the timed reference call
        entry = types.ModuleType("entry")
        exec(mlp_family.source(REFERENCE_MLP[0], REFERENCE_MLP[1], label_steps, REFERENCE_MLP[3]), entry.__dict__)
        torch.cuda.synchronize()
        began = time.perf_counter()
        predicted = entry.custom_kernel(tuple(fixed))
        torch.cuda.synchronize()
        reference_ms = (time.perf_counter() - began) * 1e3
    reference_correct = reference.check_implementation(draw["test_y"], predicted.cpu().numpy())
    return {"draw_index": draw_index, "rows": rows, "device": torch.cuda.get_device_name(),
            "torch": torch.__version__, "host": host, "gpu_clock": sampler.stop(),
            "reference_mlp": {"name": mlp_family.name(*REFERENCE_MLP), "ms": reference_ms,
                              "error_pct": 100.0 * (1.0 - reference_correct / 10000.0)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--dev-seed", type=int, default=20260925)
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    args = parser.parse_args()
    config = json.loads((HERE.parent / "full-batches" / "selection.json").read_text())["candidates"][0]["config"]
    steps = [int(s) for s in args.steps.split(",")]
    print(f"Ladder at N=10,000, steps {steps}, {args.draws} dev draws, one A100-80GB container each", flush=True)
    with modal.enable_output(), app.run():
        results = list(sweep.starmap([(i, config, steps, args.dev_seed) for i in range(args.draws)],
                                     return_exceptions=True))
    good = [r for r in results if not isinstance(r, Exception)]
    for r in results:
        if isinstance(r, Exception):
            print("container failed:", repr(r)[:500])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": config, "steps": steps, "dev_seed": args.dev_seed, "containers": good},
                              indent=1))
    print(f"wrote {out}: {len(good)} of {args.draws} containers")


if __name__ == "__main__":
    main()
