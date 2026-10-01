"""Paired public development draws, not an official sandboxed score."""
import json
from pathlib import Path
import sys

import modal

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] if modal.is_local() else Path("/root")
if modal.is_local():
    sys.path.insert(0, str(BASE))
    from run_modal import image
else:
    image = None

app = modal.App("mnist-a100-shorter-ladder-dev", image=image)


@app.function(gpu="A100-80GB", timeout=1200, single_use_containers=True)
def probe(difficulty, original, candidate):
    import importlib.util
    import time
    import torch
    import mnist

    methods = {}
    for name, source in (("original", original), ("candidate", candidate)):
        path = Path(f"/tmp/{name}.py")
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        methods[name] = module

    def draw(seed):
        return tuple(torch.as_tensor(a, device="cuda") for a in mnist.draw(seed=seed))

    warm = draw(300)
    for module in methods.values():
        module.ladder(*warm[:3])
    torch.cuda.synchronize()
    rows = []
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for seed in (301, 302, 303):
        data = draw(seed)
        for name, module in methods.items():
            flush.zero_()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            labels = module.ladder(*data[:3])
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall) * 1000
            correct = int((labels == data[3]).sum())
            rows.append(dict(seed=seed, variant=name, steps=module.STEPS,
                             total_updates=next(iter(module.CACHE.values())).total,
                             correct=correct, total=len(labels), cuda_ms=start.elapsed_time(end),
                             wall_ms=wall_ms))
            print(json.dumps(dict(difficulty=difficulty, **rows[-1])), flush=True)
    return dict(mode="paired public development draws; unsandboxed, not an official score",
                difficulty=difficulty, device=torch.cuda.get_device_name(), torch=str(torch.__version__), rows=rows)


if __name__ == "__main__":
    jobs = [(d, (BASE / f"submissions/ladder-fast-20260929/ladder_d{d}.py").read_text(),
             (HERE.parent / f"ladder_d{d}.py").read_text()) for d in (3, 4, 5)]
    with modal.enable_output(), app.run():
        for result in probe.starmap(jobs):
            (HERE / f"dev-ladder-d{result['difficulty']}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
