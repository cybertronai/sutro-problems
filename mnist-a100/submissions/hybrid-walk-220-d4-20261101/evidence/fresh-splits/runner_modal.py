"""d4 fresh-split runner on Modal A100-80GB (official image recipe). Every arm of a batch runs in the SAME container on
the same splits, each in a fresh subprocess with a cold cache, so its ms are paired by host and draw. Only our own
method files in bio/d4h/modal/ run (plus the scorer's mnist.py for data). Triton files run here, never on the 4090.

    cd upstream/sutro_bench && uv run --with modal modal run bio/d4h/d4_modal.py --specs 'f.py:fn|{json}=name;...' \
        --seed 1030000 --splits 0:12 --per 4 --out bio/d4h/d4_modal.jsonl
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MNIST = HERE.parent.parent.parent / "sutro-problems" / "mnist-a100" / "mnist.py"
FILES = sorted(p.name for p in (HERE / "modal").glob("*.py"))

image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3", "ninja~=1.11")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(MNIST, "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")
)
for f in FILES:
    image = image.add_local_file(HERE / "modal" / f, f"/root/subs/{f}", copy=True)
app = modal.App("d4-splits", image=image)

CHILD = r"""
import importlib.util, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/root")
import mnist
path, fn = @FILE@.split(":")
s = importlib.util.spec_from_file_location("cand", "/root/subs/" + path)
mod = importlib.util.module_from_spec(s); sys.modules["cand"] = mod; s.loader.exec_module(mod)
for k, v in json.loads(@CFG@).items():
    setattr(mod, k, tuple(v) if isinstance(v, list) else v)
m = getattr(mod, fn)
cu = lambda a: torch.from_numpy(a).cuda()
w = [cu(a) for a in mnist.draw("kmnist", seed=10999)]
t0 = time.time(); m(*w[:3]); torch.cuda.synchronize(); warm = time.time() - t0
pix, lab = None, None
out = []
for sp in @SPLITS@:
    f = f"/tmp/dr_@SEED@_{sp}.npz"
    if not os.path.exists(f):
        if pix is None:
            pix, lab = mnist.load_pool("mnist")
        rng = np.random.default_rng(@SEED@ + sp)
        pool = mnist.Pool("mnist", pix, lab, rng)
        ds = [pool.draw(rng) for _ in range(11)]
        np.savez(f, **{k: np.stack([d[k] for d in ds]) for k in ("train_x", "train_y", "test_x", "test_y")})
    d = np.load(f)
    wrong, ms = [], []
    for j in range(11):
        tx, ty, vx = (cu(d[k][j]) for k in ("train_x", "train_y", "test_x"))
        torch.cuda.synchronize(); t = time.perf_counter(); p = m(tx, ty, vx); torch.cuda.synchronize()
        ms.append(round((time.perf_counter() - t) * 1e3, 3))
        wrong.append(int((p.view(-1).long().cpu().numpy() != d["test_y"][j]).sum()))
    out.append(dict(split=sp, wrong=wrong, ms=ms))
print("RESULT", json.dumps(dict(warm=round(warm, 1), out=out)), flush=True)
"""


@app.function(gpu="A100-80GB", timeout=1500, max_containers=3)
def job(g: dict):
    import os, subprocess, sys, tempfile
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    rs = []
    for arm in g["arms"]:
        d = tempfile.mkdtemp()
        env = dict(os.environ, TRITON_CACHE_DIR=f"{d}/t", TORCHINDUCTOR_CACHE_DIR=f"{d}/i")
        code = (CHILD.replace("@FILE@", repr(arm["file"])).replace("@CFG@", repr(json.dumps(arm["cfg"])))
                .replace("@SPLITS@", repr(g["splits"])).replace("@SEED@", str(g["seed"])))
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=d, timeout=400)
        res = next((json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        rs.append(dict(name=arm["name"], gpu=gpu, res=res, err=None if res else (r.stdout[-1500:] + r.stderr[-3000:])))
    return rs


@app.local_entrypoint()
def main(specs: str, seed: int = 1030000, splits: str = "0:12", per: int = 4, out: str = "bio/d4h/d4_modal.jsonl"):
    lo, hi = map(int, splits.split(":"))
    arms = []
    for sp in specs.split(";"):
        fs, name = sp.rsplit("=", 1)
        f, cfg = (fs.split("|", 1) + ["{}"])[:2]
        arms.append(dict(name=name, file=f, cfg=json.loads(cfg)))
    groups = [dict(arms=arms, seed=seed, splits=list(range(a, min(hi, a + per)))) for a in range(lo, hi, per)]
    with open(out, "a") as fh:
        for rs in job.map(groups, order_outputs=False):
            for r in rs:
                if r["res"] is None:
                    print("FAILED", r["name"], r["err"])
                    continue
                for o in r["res"]["out"]:
                    fh.write(json.dumps(dict(name=r["name"], seed=seed, gpu=r["gpu"], warm=r["res"]["warm"], **o)) + "\n")
                    print(r["name"], o["split"], f"{sum(o['wrong']) / 1100:.3f}%", f"med {sorted(o['ms'])[5]:.2f} ms", r["gpu"], f"warm {r['res']['warm']}", flush=True)
