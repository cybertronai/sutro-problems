"""Follow-up to probe_sched.py: more total learning rate (later decay, higher base) at 5,000 steps, same draws.

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
    import numpy as np

    epochs = total // spe
    epoch = np.arange(total, dtype=np.float64) // spe

    def linear(decay_fraction, scale):
        decay = round(epochs * decay_fraction)
        return scale * base * np.clip((epochs - epoch) / (epochs - decay), 0, 1)

    return {"recipe": linear(100 / 150, 1.0), "late": linear(0.85, 1.0), "recipe1.5x": linear(100 / 150, 1.5),
            "recipe2x": linear(100 / 150, 2.0), "late1.5x": linear(0.85, 1.5)}


@app.function(gpu="A100-80GB", timeout=1200, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import torch, mnist, ladder_t

    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {}
    for steps in (5000,):
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
