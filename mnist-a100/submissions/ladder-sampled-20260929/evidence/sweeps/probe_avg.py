"""One Modal A100-80GB container: does weight averaging help the retuned Ladder (batch 1,000, TF32)?

    modal run probe_avg.py      # ~10-12 min, ~$0.50

Same six draws as the earlier probes (seeds 301-306). Averaging coefficients and the lr schedule are tables the
graph reads, so every variant at a step count reuses one captured graph. Predictions use the averaged weights,
with BatchNorm recalibrated on them.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(HERE / "mnist.py", "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")
    .add_local_file(HERE / "ladder_avg.py", "/root/ladder_avg.py")
)
app = modal.App("sutro-ladder-probe", image=image)

CONFIGS = [(1200, 0.008), (2400, 0.008), (9000, 0.006)]


def tail(total, fraction):
    import numpy as np
    start = total - round(fraction * total)
    t = np.arange(total)
    return np.where(t >= start, 1.0 / (t - start + 1).clip(min=1), 0.0)


def ema(total, start_fraction, horizon_fraction):
    import numpy as np
    start, horizon = round(start_fraction * total), max(1.0, horizon_fraction * total)
    t = np.arange(total)
    return np.where(t > start, 1.0 / horizon, np.where(t == start, 1.0, 0.0))


@app.function(gpu="A100-80GB", timeout=1800, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import numpy as np, torch, mnist, ladder_avg as L

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {"gpu": torch.cuda.get_device_name()}
    L.BATCH = 1000
    for steps, lr in CONFIGS:
        L.STEPS = steps
        L.CACHE.clear()
        L.ladder(*draws[0][:3])
        tr = next(iter(L.CACHE.values()))
        base = tr.schedule.clone() * (lr / L.LR)          # the recipe's shape at this peak rate
        flat = torch.full_like(base, lr)                   # no decay
        T = tr.total
        variants = {"none": (base, None), "tail25": (base, tail(T, 0.25)), "tail50": (base, tail(T, 0.50)),
                    "ema10@50": (base, ema(T, 0.5, 0.1)), "nodecay+tail33": (flat, tail(T, 1 / 3))}
        for name, (sched, coef) in variants.items():
            tr.schedule.copy_(sched)
            tr.averaging = coef is not None
            tr.avg_coef.copy_(torch.tensor(coef if coef is not None else np.zeros(T), dtype=torch.float32, device=dev))
            errs, secs = [], []
            for tx, ty, qx, qy in draws:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                labels = L.ladder(tx, ty, qx)
                end.record(); torch.cuda.synchronize()
                errs.append((labels != qy).float().mean().item() * 100)
                secs.append(start.elapsed_time(end) / 1000)
            key = f"steps {T} {name}"
            results[key] = (sum(errs) / len(errs), errs, sum(secs) / len(secs))
            print(f"{key:26s} mean {results[key][0]:.3f}%  {results[key][2]:.2f} s/call  draws {[round(e, 2) for e in errs]}",
                  flush=True)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
