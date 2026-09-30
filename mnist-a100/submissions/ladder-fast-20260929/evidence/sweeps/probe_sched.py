"""One Modal A100-80GB container: does the Ladder's learning-rate schedule matter at a fixed step budget?

    modal run probe_sched.py      # ~8-10 min, ~$0.40

Schedules (base lr 0.002, all reach 0 at the last step), on the same six draws at 5,000 and 6,500 steps:
  recipe      constant for 2/3, then linear decay (Pezeshki et al.'s 100/150), per epoch
  early       constant for 1/3, then linear decay
  lognorm1x   shifted lognormal density, peak at 10% of training, sigma 0.65, peak 0.002 (Seth's caffeine schedule)
  lognorm2x   the same with peak 0.004
The schedule tensor is overwritten in place, so one captured graph serves every schedule at a step count.
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
    .add_local_file(HERE / "ladder_t.py", "/root/ladder_t.py")
)
app = modal.App("sutro-ladder-probe", image=image)


def schedules(total, spe, base=0.002):
    import math
    import numpy as np

    epochs = total // spe
    t = np.arange(total, dtype=np.float64)
    epoch = t // spe
    out = {}
    decay = round(epochs * 100 / 150)
    out["recipe"] = base * np.clip((epochs - epoch) / (epochs - decay), 0, 1)
    decay = round(epochs / 3)
    out["early"] = base * np.clip((epochs - epoch) / (epochs - decay), 0, 1)
    sigma, shift, peak = 0.65, 1.0, 0.1 * total
    mu = math.log(peak + shift) + sigma ** 2

    def pdf(x):
        return np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * math.sqrt(2 * math.pi))

    raw = (pdf(t + shift) - pdf(total + shift)) / (pdf(peak + shift) - pdf(total + shift))
    raw = np.clip(raw, 0, None)
    out["lognorm1x"] = base * raw
    out["lognorm2x"] = 2 * base * raw
    return out


@app.function(gpu="A100-80GB", timeout=1200, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import torch, mnist, ladder_t

    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {}
    for steps in (5000, 6500):
        ladder_t.STEPS = steps
        ladder_t.ladder(*draws[0][:3])  # build and capture
        trainer = next(iter(ladder_t.CACHE.values()))
        for name, values in schedules(trainer.total, trainer.steps_per_epoch).items():
            trainer.schedule.copy_(torch.tensor(values, dtype=torch.float32, device=dev))
            errs = []
            for tx, ty, qx, qy in draws:
                errs.append((ladder_t.ladder(tx, ty, qx) != qy).float().mean().item() * 100)
            results[f"{steps} {name}"] = errs
            print(f"{steps} steps {name:10s} mean {sum(errs) / len(errs):.3f}%  draws {[round(e, 2) for e in errs]}",
                  flush=True)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
