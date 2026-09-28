"""One Modal A100-80GB container: are the fused Triton kernels right, and how fast is the step?

    modal run submissions/ladder-triton-20260928/verify_kernels.py    # from mnist-a100/; ~4-5 min, ~$0.20

The reference is the plain-PyTorch port in ../ladder-graph-20260928/ladder.py.

1. Decode and Encode (noise off) against the PyTorch reference: outputs and every gradient.
2. Encode's in-kernel noise: mean ~0, std ~0.3, different at different step counts.
3. ms per step (2,000-step fit), then fits at 9,000 and 24,000 steps on fresh draws.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # mnist-a100/
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(ROOT / "mnist.py", "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")
    .add_local_file(HERE.parent / "ladder-graph-20260928" / "ladder.py", "/root/ladder.py")
    .add_local_file(HERE / "ladder_d5.py", "/root/ladder_t.py")
)
app = modal.App("sutro-ladder-probe", image=image)


@app.function(gpu="A100-80GB", timeout=900, single_use_containers=True)
def probe():
    import sys, time
    sys.path.insert(0, "/root")
    import torch, mnist, ladder, ladder_t
    import torch.nn.functional as F

    dev = torch.device("cuda")
    torch.manual_seed(0)
    out = {"gpu": torch.cuda.get_device_name()}

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

    # 1a. Decode vs reference combinator(normalize(u))
    worst = 0.0
    for s in (1000, 250, 60, 10):
        u = torch.randn(250, s, device=dev, requires_grad=True)
        lat = torch.randn(250, s, device=dev, requires_grad=True)
        p = (torch.randn(17, s, device=dev) * 0.5).requires_grad_()
        g = torch.randn(250, s, device=dev)
        y = ladder_t.Decode.apply(u, lat, p)
        grads = torch.autograd.grad(y, (u, lat, p), g)
        W1 = p[0:6].T.reshape(s, 3, 2); b1 = p[6:8].T
        W2 = p[8:12].T.reshape(s, 2, 2); b2 = p[12:14].T
        W3 = p[14:16].T.reshape(s, 2, 1); b3 = p[16:17].T
        yr = ladder.combinator([W1, b1, W2, b2, W3, b3], lat, ladder.normalize(u))
        grads_r = torch.autograd.grad(yr, (u, lat, p), g)
        errs = [rel(y, yr)] + [rel(a, b) for a, b in zip(grads, grads_r)]
        worst = max(worst, *errs)
        print(f"decode s={s}: rel err out/du/dlat/dp = {[f'{e:.1e}' for e in errs]}", flush=True)
    out["decode_worst_rel"] = worst

    # 1b. Encode with noise off vs reference
    saved = ladder_t.NOISE
    ladder_t.NOISE = 0.0
    counter = torch.zeros(1, dtype=torch.long, device=dev)
    worst = 0.0
    for d, top in ((1000, False), (250, False), (10, True)):
        raw = torch.randn(2, 250, d, device=dev, requires_grad=True)
        beta = torch.randn(d, device=dev, requires_grad=True)
        gamma = torch.randn(d, device=dev, requires_grad=True) if top else None
        gz, gh = torch.randn(2, 250, d, device=dev), torch.randn(2, 250, d, device=dev)
        z, h = ladder_t.Encode.apply(raw, beta, gamma, counter, 0)
        inputs = (raw, beta) + ((gamma,) if top else ())
        grads = torch.autograd.grad((z, h), inputs, (gz, gh))
        zr = ladder.normalize(raw)
        hr = (zr + beta) * gamma if top else F.relu(zr + beta)
        grads_r = torch.autograd.grad((zr, hr), inputs, (gz, gh))
        errs = [rel(z, zr), rel(h, hr)] + [rel(a, b) for a, b in zip(grads, grads_r)]
        worst = max(worst, *errs)
        print(f"encode d={d} top={top}: rel err z/h/draw/dbeta[/dgamma] = {[f'{e:.1e}' for e in errs]}", flush=True)
    out["encode_worst_rel"] = worst
    ladder_t.NOISE = saved

    # 2. noise statistics
    raw = torch.randn(2, 250, 1000, device=dev)
    beta = torch.zeros(1000, device=dev)
    z0, _ = ladder_t.Encode.apply(raw, beta, None, counter, 0)
    counter.add_(1)
    z1, _ = ladder_t.Encode.apply(raw, beta, None, counter, 0)
    noise = z0 - ladder.normalize(raw)
    out["noise"] = (noise.mean().item(), noise.std().item(), (z0 - z1).abs().mean().item())
    print(f"noise mean {out['noise'][0]:.4f} std {out['noise'][1]:.4f}; step-to-step change {out['noise'][2]:.3f}", flush=True)

    # 3. speed and accuracy
    def fit(steps, seed):
        ladder_t.STEPS = steps
        tx, ty, qx, qy = [torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", seed)]
        t = time.perf_counter()
        ladder_t.ladder(tx, ty, qx)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        labels = ladder_t.ladder(tx, ty, qx)
        end.record(); torch.cuda.synchronize()
        err, ms = (labels != qy).float().mean().item() * 100, start.elapsed_time(end)
        print(f"{steps} steps: {err:.2f}% in {ms / 1000:.2f} s ({ms / steps:.3f} ms/step), warm-up {warm:.1f} s", flush=True)
        return err, ms

    out["fit_2000"] = fit(2000, 101)
    out["fit_9000"] = fit(9000, 103)
    out["fit_24000"] = fit(24000, 104)
    return out


@app.local_entrypoint()
def main():
    print(probe.remote())
