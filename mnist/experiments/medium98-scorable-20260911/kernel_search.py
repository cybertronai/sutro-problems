"""Training-only kernel feasibility search; never a formal submission score.

The bounded grid is frozen with source/input/split hashes before dispatch.
Uses native GPU linear algebra for screening only. An eligible implementation
would separately need a fixed arithmetic solver and a fresh eleven-draw audit.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).pip_install('numpy==2.2.6')
app = modal.App('sutro-mnist-medium-kernel-screen')


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=8192, timeout=1800,
              max_containers=1, scaledown_window=2, retries=0)
def search(payload, split, protocol):
    import time
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert set(payload) == {'train_images', 'train_labels'}
    arrays = {k: np.frombuffer(v['bytes'], dtype=v['dtype']).reshape(v['shape']).copy()
              for k, v in payload.items()}
    for k, a in arrays.items():
        assert hashlib.sha256(a.tobytes()).hexdigest() == protocol['input_sha256'][k]
    images = torch.tensor(arrays['train_images'], device='cuda', dtype=torch.float64)
    labels = torch.tensor(arrays['train_labels'], device='cuda')
    fit = torch.tensor(split['fit_positions'], device='cuda')
    val = torch.tensor(split['validation_positions'], device='cuda')
    truth = labels[val]
    targets = torch.nn.functional.one_hot(labels[fit], 10).double()
    rows = []
    start = time.perf_counter()
    for transform in protocol['transforms']:
        x = images
        if transform == 'deskew':
            coord = torch.arange(9, device='cuda', dtype=torch.float64) - 4
            yy, xx = torch.meshgrid(coord, coord, indexing='ij')
            mass = x.sum((1,2,3)).clamp_min(1e-8)
            mx = (x[:,0]*xx).sum((1,2))/mass
            my = (x[:,0]*yy).sum((1,2))/mass
            dx, dy = xx[None]-mx[:,None,None], yy[None]-my[:,None,None]
            xy = (x[:,0]*dx*dy).sum((1,2))
            y2 = (x[:,0]*dy*dy).sum((1,2)).clamp_min(1e-8)
            shear = (xy/y2).clamp(-1,1)
            # Output-centered inverse warp: eliminate covariance in x/y.
            gx = xx[None]+mx[:,None,None]+shear[:,None,None]*yy[None]
            gy = yy[None]+my[:,None,None]
            grid = torch.stack((gx.expand(-1,9,9),gy.expand(-1,9,9)),dim=-1)/4
            x = torch.nn.functional.grid_sample(x, grid, align_corners=True)
        x = x.reshape(len(x),81)
        xf, xv = x[fit], x[val]
        df = (xf.square().sum(1)[:,None]+xf.square().sum(1)[None]-2*(xf@xf.T)).clamp_min(0)
        dv = (xv.square().sum(1)[:,None]+xf.square().sum(1)[None]-2*(xv@xf.T)).clamp_min(0)
        for gamma in protocol['gammas']:
            k, q = torch.exp(-gamma*df), torch.exp(-gamma*dv)
            for ridge in protocol['ridges']:
                weights = torch.linalg.solve(k+ridge*torch.eye(len(fit),device='cuda'), targets)
                scores = q@weights
                pred = scores.argmax(1)
                row = dict(transform=transform,gamma=gamma,ridge=ridge,
                           correct=int((pred==truth).sum()),total=len(val),
                           elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
                print(json.dumps(row),flush=True)
    return dict(rows=rows,protocol=protocol,device=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__),elapsed_seconds=time.perf_counter()-start,
                formal_implementation=False,test_arrays_supplied=False)


@app.local_entrypoint()
def main(data: str='/tmp/sutro-mnist-medium-convnet/mnist/data/medium-train-only.npz'):
    import numpy as np
    split = json.loads((HERE.parent/'medium-convnet-20260911'/'validation_split.json').read_text())
    with np.load(data,allow_pickle=False) as archive:
        assert {'train_images','train_labels'} <= set(archive.files) <= {'train_images','train_labels','train_indices'}
        arrays = {k: np.ascontiguousarray(archive[k]) for k in ('train_images','train_labels')}
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest(),
        input_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()},
        transforms=['raw','deskew'],gammas=[.25,.5,1.,2.],ridges=[.001,.01,.1,1.],
        scope='4800/1200 split of canonical training arrays only; exploratory native FP64 solve',
        image=IMAGE_REF)
    target = HERE/'kernel_protocol.json'
    if target.exists():
        raise RuntimeError('Protocol already frozen; do not overwrite')
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    payload = {k:dict(bytes=a.tobytes(),shape=a.shape,dtype=str(a.dtype)) for k,a in arrays.items()}
    result = search.remote(payload,split,protocol)
    (HERE/'kernel_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(sorted(result['rows'],key=lambda r:-r['correct'])[:6],indent=2))
