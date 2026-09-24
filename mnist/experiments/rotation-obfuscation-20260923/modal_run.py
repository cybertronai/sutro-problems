#!/usr/bin/env python
"""Fan the study's stages out to Modal containers.

    python -m modal run modal_run.py --stage blind    --seeds 1,2,3            # CPU containers, one per (seed, variant)
    python -m modal run modal_run.py --stage blindcnn --seeds 1,2,3 --gpu      # T4 containers, at most 2 at a time
    python -m modal run modal_run.py --stage smuggle  --seeds 1,2,3 --gpu --per-seed

Result files come back into results/ next to this script; existing files are skipped.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import modal

HERE = Path(__file__).resolve().parent
PMNIST = Path('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
RES = HERE / 'results'
REMOTE = '/root/exp'

app = modal.App('rotation-obfuscation')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('numpy==1.26.4', 'scipy==1.11.4', 'scikit-learn==1.4.2', 'torch==2.2.2')
         .add_local_file(str(PMNIST / 'topology.py'), f'{REMOTE}/pmnist/topology.py')
         .add_local_file(str(HERE / 'pool9.npz'), f'{REMOTE}/pool9.npz')
         .add_local_file(str(HERE / 'common.py'), f'{REMOTE}/common.py')
         .add_local_file(str(HERE / 'models.py'), f'{REMOTE}/models.py')
         .add_local_file(str(HERE / 'attacks.py'), f'{REMOTE}/attacks.py')
         .add_local_file(str(HERE / 'run.py'), f'{REMOTE}/run.py'))

DEFAULT_VARIANTS = {
    'utility': ['raw', 'perm', 'rot', 'zca1e-3', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'separate': ['zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'blind': ['perm', 'rot', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'blind2': ['perm', 'rot', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'blindcnn': ['perm', 'rot', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'informed': ['perm', 'rot', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'smuggle': ['perm', 'rot', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot'],
    'exact': ['rot', 'zca1e-3+rot', 'pca60+rot'],
}
INPUTS = {'blind2': lambda s, v: [f'blind-{v}-s{s}.npz'], 'blindcnn': lambda s, v: [f'blind-{v}-s{s}.npz', f'blind2-{v}-s{s}.npz'],
          'smuggle': lambda s, v: [f'informed-{v}-s{s}.json']}


def _run(stage, seeds, variants, inputs):
    import sys
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    res = Path(REMOTE) / 'results'
    res.mkdir(exist_ok=True)
    for name, blob in (inputs or {}).items():
        (res / name).write_bytes(blob)
    before = set(os.listdir(res))
    t0 = time.time()
    import run
    run.STAGES[stage](seeds, variants)
    out = {n: (res / n).read_bytes() for n in os.listdir(res) if n not in before}
    out['__meta__'] = f'{stage} seeds={seeds} variants={variants} seconds={time.time() - t0:.0f}'.encode()
    return out


@app.function(image=image, cpu=4.0, memory=8192, timeout=3 * 3600, max_containers=12)
def run_cpu(stage, seeds, variants, inputs=None):
    return _run(stage, seeds, variants, inputs)


@app.function(image=image, gpu='T4', cpu=4.0, memory=8192, timeout=3 * 3600, max_containers=2)
def run_gpu(stage, seeds, variants, inputs=None):
    return _run(stage, seeds, variants, inputs)


@app.local_entrypoint()
def main(stage: str, seeds: str = '1,2,3', variants: str = '', gpu: bool = False, per_seed: bool = False):
    seed_list = [int(s) for s in seeds.split(',')]
    variant_list = variants.split(',') if variants else DEFAULT_VARIANTS[stage]
    RES.mkdir(exist_ok=True)
    tasks = []
    for s in seed_list:
        groups = [variant_list] if per_seed else [[v] for v in variant_list]
        for vs in groups:
            todo = [v for v in vs if not (RES / f'{stage}-{v}-s{s}.json').exists()]
            if not todo:
                continue
            inputs = {}
            for v in todo:
                for name in INPUTS.get(stage, lambda s, v: [])(s, v):
                    if (RES / name).exists():
                        inputs[name] = (RES / name).read_bytes()
            tasks.append((s, todo, inputs))
    fn = run_gpu if gpu else run_cpu
    print(f'{stage}: {len(tasks)} tasks on {"T4 GPU" if gpu else "CPU"} containers', flush=True)
    t0 = time.time()
    handles = [fn.spawn(stage, [s], vs, inputs) for s, vs, inputs in tasks]
    total_container_seconds = 0.0
    for (s, vs, _), h in zip(tasks, handles):
        try:
            out = h.get()
        except Exception as err:
            print(f'  FAILED seed {s} {vs}: {type(err).__name__}: {err}', flush=True)
            continue
        meta = out.pop('__meta__', b'').decode()
        total_container_seconds += float(meta.split('seconds=')[-1] or 0)
        for name, blob in out.items():
            (RES / name).write_bytes(blob)
        print(f'  done seed {s} {vs}: {len(out)} files; {meta}', flush=True)
    print(f'{stage}: wall {time.time() - t0:.0f}s, container-seconds {total_container_seconds:.0f}', flush=True)
