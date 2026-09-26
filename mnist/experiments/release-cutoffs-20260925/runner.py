"""Bounded Modal controller (and free local-CPU runner) for the release-cutoffs study
(copied from release-ladder-20260924; only the app name differs).

Run as a PLAIN SCRIPT.  Never through ``modal run``::

    /tmp/penv/bin/python runner.py --plan plans/smoke.json --validate-only
    /tmp/penv/bin/python runner.py --plan plans/dev.json --gpu a100 --gpus 4 --minutes 25
    /tmp/penv/bin/python runner.py --local --plan plans/kernels.json

``--validate-only`` never creates a Modal app and never touches the ledger:
``main()`` writes the preflight receipt and returns *before* the ``with app.run()``
block is reached, so no app object is ever registered with the provider.  There
is no ``@app.local_entrypoint`` on purpose -- ``modal run`` would create an
ephemeral app before any of our code ran, which is exactly what validation must
avoid.

``--local`` runs the plan sequentially on this machine's CPU (``device='cpu'``)
in the same result format, for kernel candidates that do not need a GPU.  It
creates no app and touches no ledger, so it is free.

WHAT THE CONTAINER GETS
-----------------------
Only the three released arrays of ONE job (``train_z``, ``train_y``, ``query_z``,
zlib-compressed, ~4.3 MB at n=10000) and the learner modules.  The controller
builds the arrays with ``study.job_arrays``; the container asserts their hashes
against the job dict, so what a learner sees is bit-identical to what
``score.verify_record`` re-derives.

The container does NOT recompute the release from pixels and the seed.
``release_map`` is an eigendecomposition and LAPACK is not bit-reproducible
across BLAS builds or CPU targets -- on this controller, changing
``OPENBLAS_NUM_THREADS`` alone already changes ``W``, and a float32 released
array flips entries -- so every input-hash assertion would fail remotely.  The
pixel pool is therefore not in the image either, which also means no container
can refit ``(mu, W, Q)`` and invert the obfuscation back to 9x9 pixels.

Query labels never exist inside a container.  Network access is NOT blocked:
Modal moves inputs and outputs above 2 MiB through its blob store, which the
container must reach.

The ledger reserves and charges exactly ``--gpus`` fully occupied workers of the
selected GPU type for the whole app lifetime (upper-envelope accounting), and
the app is stopped at the reserved work deadline.  No accuracy is ever computed
or printed here: query labels never leave ``score.py``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from threading import Event, Lock, Timer
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
APP_NAME = 'release-cutoffs-20260925'
IMAGE = ('ghcr.io/ab-10/wikitext-bench@'
         'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
# study.py is deliberately NOT mounted: it carries query_labels() and reads
# raw/pool_labels.npy.  The container reconstructs the draw from release.py.
SOURCE_FILES = ['release.py', 'kernels.py', 'neural.py',
                'pmnist_learners.py', 'ladder_model.py']
POOL_PIXELS = ROOT / 'raw' / 'pool_pixels.npy'   # controller only; never in the image
MAX_GPUS = 8
# Must cover the post-run shutdown verification (one `modal app stop`, up to
# eight `modal app list` calls and 107 s of retry sleeps), because budget.py
# charges wall clock from the reservation to the moment the app is verified
# stopped and blocks the ledger when that exceeds the reservation.
SHUTDOWN_GRACE_SECONDS = 240
GPU_KINDS = {'a100': 'A100-40GB', 't4': 'T4'}
NEURAL_FAMILIES = ('mlp', 'ladder', 'vat')
POOL_COUNT = 60000
PIXEL_COUNT = 81
RELEASE_DIMS = 60
QUERY_COUNT = 10000
ARRAY_NAMES = ('train_z', 'train_y', 'query_z')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ahash(array):
    array = np.asarray(array)
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def learner_module_name(config):
    """Which learner module owns a config: explicit 'module', else family."""
    name = config.get('module')
    if name is not None:
        if name not in ('kernels', 'neural'):
            raise ValueError(f"config['module'] must be 'kernels' or 'neural', got {name!r}")
        return name
    return 'neural' if config.get('family') in NEURAL_FAMILIES else 'kernels'


def call_fit_predict(module, arrays, config, seed, device, deadline_unix):
    """Call ``module.fit_predict`` by PARAMETER NAME where possible.

    The study contract is
    ``fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix)``
    but the ported pmnist learners use ``(..., config, seed, deadline_unix, device)``.
    Binding by name makes either ordering correct and fails loudly on anything else.
    """
    function = module.fit_predict
    names = [p.name for p in inspect.signature(function).parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    supplied = {
        'train_z': arrays['train_z'], 'train_x': arrays['train_z'],
        'train_y': arrays['train_y'],
        'query_z': arrays['query_z'], 'query_x': arrays['query_z'],
        'config': config, 'seed': seed, 'device': device,
        'deadline_unix': deadline_unix, 'deadline': deadline_unix,
    }
    if any(name.startswith('*') for name in names) or not set(names) <= set(supplied):
        # Unknown parameter names: fall back to the documented positional order.
        return function(arrays['train_z'], arrays['train_y'], arrays['query_z'],
                        config, seed, device, deadline_unix)
    return function(**{name: supplied[name] for name in names})


def rebuild_arrays(pool_pixels, seed, n, train_y):
    """The DRAW PROTOCOL, from pixels + seed only.  CONTROLLER ONLY.

    An independent re-implementation of ``study.job_arrays``: ``check_arrays``
    runs both in the same process and compares them against the job's
    ``input_sha256``, so the two can never silently diverge.  It is NOT run in a
    container -- the eigendecomposition inside ``release_map`` is not
    bit-reproducible across BLAS builds, and the container has no pixels.
    """
    import release

    seed, n = int(seed), int(n)
    train_universe, test_universe = release.split_universes(
        POOL_COUNT, seed, release.UNIVERSE_SALT)
    order = np.random.default_rng([seed, release.DRAW_SALT]).permutation(train_universe)
    train_rows = np.ascontiguousarray(order)[:n]
    query_rows = np.random.default_rng([seed, release.DRAW_SALT + 1]).choice(
        test_universe, QUERY_COUNT, replace=False)
    if np.intersect1d(train_rows, query_rows).size:
        raise AssertionError('train and query rows overlap')
    mu, whitener, rotation = release.release_map(pool_pixels[train_rows], seed, RELEASE_DIMS)
    transform = rotation @ whitener
    arrays = {'train_z': release.apply_release(pool_pixels[train_rows], mu, transform),
              'train_y': np.ascontiguousarray(train_y, dtype=np.uint8),
              'query_z': release.apply_release(pool_pixels[query_rows], mu, transform)}
    del mu, whitener, rotation, transform
    return arrays, np.ascontiguousarray(train_rows), np.ascontiguousarray(query_rows)


def run_fit(arrays, job, provenance, deadline_unix, device):
    """Verify the job's arrays, run the learner and build the result record.

    Shared by the Modal worker and ``--local``.  ``arrays`` are the controller's
    canonical released arrays (``study.job_arrays``); this function never
    receives, derives or returns a query label, and never sees a pixel.
    """
    started = time.monotonic()
    if time.time() > deadline_unix:
        raise TimeoutError('Expired study deadline before startup')
    for name in ARRAY_NAMES:
        if ahash(arrays[name]) != job['input_sha256'][name]:
            raise AssertionError('Input hash mismatch: ' + name)
    n = int(job['n'])
    assert arrays['train_z'].shape == (n, RELEASE_DIMS) and arrays['train_z'].dtype == np.float32
    assert arrays['train_y'].shape == (n,) and arrays['train_y'].dtype == np.uint8
    assert arrays['query_z'].shape == (QUERY_COUNT, RELEASE_DIMS)
    assert arrays['query_z'].dtype == np.float32
    assert int(arrays['train_y'].max()) < 10
    assert np.isfinite(arrays['train_z']).all() and np.isfinite(arrays['query_z']).all()

    module_name = learner_module_name(job['config'])
    module = __import__(module_name)
    time_budget = float(job['time_budget_seconds'])
    deadline = min(float(deadline_unix), time.time() + time_budget)
    output = call_fit_predict(module, arrays, job['config'], int(job['learner_seed']),
                              device, deadline)
    # The per-fit budget is a protocol claim, not a suggestion: a learner that
    # only checks the clock between coarse iterations must not have an
    # over-budget result recorded as if it were legal.
    measured_wall_seconds = time.monotonic() - started
    if measured_wall_seconds > time_budget + 5.0:
        raise TimeoutError(
            f'fit used {measured_wall_seconds:.1f}s > '
            f"time_budget_seconds={job['time_budget_seconds']}")
    logits = np.ascontiguousarray(output['logits'], dtype=np.float32)
    labels = np.ascontiguousarray(output['labels'], dtype=np.uint8)
    assert logits.shape == (QUERY_COUNT, 10) and labels.shape == (QUERY_COUNT,)
    assert np.isfinite(logits).all()
    assert np.array_equal(labels, np.argmax(logits, axis=1).astype(np.uint8))
    software = {'python': platform.python_version(), 'numpy': np.__version__}
    hardware = platform.processor() or platform.machine()
    try:
        import torch
        software.update({'torch': str(torch.__version__), 'cuda': torch.version.cuda})
        if device == 'cuda':
            hardware = torch.cuda.get_device_name(0)
            software['cudnn'] = torch.backends.cudnn.version()
    except Exception:                                          # pragma: no cover
        pass
    metrics = dict(output['metrics'])
    metrics['measured_wall_seconds'] = measured_wall_seconds
    record = {
        'job': job, 'metrics': metrics, 'provenance': provenance,
        'completed_at': utc(), 'job_wall_seconds': time.monotonic() - started,
        'device': device, 'hardware': hardware, 'learner_module': module_name,
        'software': software,
        'output_sha256': {'logits': ahash(logits), 'labels': ahash(labels)},
        'output_specs': {'logits': {'shape': list(logits.shape), 'dtype': 'float32'},
                         'labels': {'shape': list(labels.shape), 'dtype': 'uint8'}},
        'query_labels_supplied': False,
    }
    return record, logits, labels


# --------------------------------------------------------------------------
# Modal image and workers
# --------------------------------------------------------------------------
def _cli_max_containers(argv=None):
    """Read ``--gpus`` before the decorators run; Modal fixes the pool at import."""
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(argv):
        value = None
        if token == '--gpus' and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith('--gpus='):
            value = token.split('=', 1)[1]
        if value is not None:
            try:
                parsed = int(value)
            except ValueError:
                break
            if 1 <= parsed <= MAX_GPUS:
                return parsed
            break
    return MAX_GPUS


MAX_CONTAINERS = _cli_max_containers()
ADDED_FILES = [name for name in SOURCE_FILES if (ROOT / name).exists()]

if os.environ.get('RELEASE_LADDER_NO_MODAL') == '1':
    modal = None
    app = None
    fit_a100 = fit_t4 = None
else:
    import modal

    image = modal.Image.from_registry(IMAGE).pip_install('numpy==2.2.6', 'scipy==1.15.3')
    for _name in ADDED_FILES:
        image = image.add_local_file(str(ROOT / _name), remote_path='/root/' + _name)
    # The pixel pool is deliberately NOT added: with it and the job's seed a
    # container could refit the secret release map and invert z back to pixels.
    app = modal.App(APP_NAME)

    def _remote_body(job, payload, provenance, deadline_unix):
        """One learner fit on one GPU.  Receives this job's released arrays only."""
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        sys.path.insert(0, '/root')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for name, digest in provenance['image_source_sha256'].items():
            assert hashlib.sha256(Path('/root', name).read_bytes()).hexdigest() == digest, (
                'Source changed in flight: ' + name)
        record, logits, labels = run_fit(unpack_arrays(payload), job, provenance,
                                         deadline_unix, 'cuda')
        return record, logits.tobytes(), labels.tobytes()

    # Two distinctly NAMED wrappers: Modal tags a function by its qualified name,
    # so registering one object twice would collide.
    def fit_a100(job, payload, provenance, deadline_unix):
        """A100-40GB worker."""
        return _remote_body(job, payload, provenance, deadline_unix)

    def fit_t4(job, payload, provenance, deadline_unix):
        """T4 worker."""
        return _remote_body(job, payload, provenance, deadline_unix)

    _WORKER = dict(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1260,
                   startup_timeout=300, retries=0, min_containers=0,
                   max_containers=MAX_CONTAINERS, buffer_containers=0,
                   scaledown_window=30)
    fit_a100 = app.function(gpu='A100-40GB', **_WORKER)(fit_a100)
    fit_t4 = app.function(gpu='T4', **_WORKER)(fit_t4)


# --------------------------------------------------------------------------
# provider evidence helpers
# --------------------------------------------------------------------------
def list_apps():
    completed = subprocess.run([sys.executable, '-m', 'modal', 'app', 'list', '--json'],
                               capture_output=True, text=True, timeout=60, check=True)
    return json.loads(completed.stdout)


def stopped_row(app_id):
    for row in list_apps():
        normalized = {str(k).lower().replace(' ', '_'): v for k, v in row.items()}
        if normalized.get('app_id') == app_id:
            state = str(normalized.get('state', '')).lower()
            tasks = normalized.get('tasks', -1)
            try:
                tasks = int(tasks)
            except (TypeError, ValueError):
                tasks = -1
            if state.endswith('stopped') and tasks == 0:
                return row
    return None


# --------------------------------------------------------------------------
# plan handling, resume and scheduling
# --------------------------------------------------------------------------
REQUIRED_JOB_KEYS = {'id', 'stage', 'seed', 'n', 'candidate_id', 'config', 'learner_seed',
                     'time_budget_seconds', 'device_kind', 'train_rows_sha256',
                     'query_rows_sha256', 'input_sha256', 'estimated_seconds'}


def load_plan(plan_path):
    jobs = json.loads(Path(plan_path).read_text())
    if isinstance(jobs, dict):
        jobs = jobs['jobs']
    assert isinstance(jobs, list) and jobs, 'Plan must be a nonempty list of jobs'
    assert len({job['id'] for job in jobs}) == len(jobs), 'Duplicate job ids'
    for job in jobs:
        missing = REQUIRED_JOB_KEYS - set(job)
        assert not missing, f"Job {job['id']} missing keys {sorted(missing)}"
        assert set(job['input_sha256']) == set(ARRAY_NAMES)
        assert 0 < int(job['time_budget_seconds']) <= 1200
        assert job['device_kind'] in ('gpu', 'cpu')
        assert float(job['estimated_seconds']) > 0
    return jobs


def verify_existing(job, sources):
    """True when a complete, hash-verified result for this exact job exists."""
    path = ROOT / 'results' / f"{job['id']}.json"
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except Exception:
        return False
    if record.get('job') != job:
        print(json.dumps({'id': job['id'], 'status': 'rerun',
                          'reason': 'existing result used a different job definition'}),
              flush=True)
        return False
    recorded_sources = (record.get('provenance') or {}).get('source_sha256') or {}
    for name in SOURCE_FILES:
        if name not in sources:
            continue
        if recorded_sources.get(name) != sources[name]:
            print(json.dumps({'id': job['id'], 'status': 'rerun',
                              'reason': f'{name} changed since this result was produced'}),
                  flush=True)
            return False
    predictions = ROOT / record['predictions_path']
    if not predictions.exists() or sha(predictions) != record['predictions_sha256']:
        print(json.dumps({'id': job['id'], 'status': 'rerun',
                          'reason': 'prediction file missing or changed'}), flush=True)
        return False
    with np.load(predictions, allow_pickle=False) as bundle:
        if (ahash(bundle['logits']) != record['output_sha256']['logits']
                or ahash(bundle['labels']) != record['output_sha256']['labels']):
            print(json.dumps({'id': job['id'], 'status': 'rerun',
                              'reason': 'prediction contents changed'}), flush=True)
            return False
    return True


def lpt_makespan(durations, workers):
    """Longest-processing-time-first makespan for ``workers`` identical machines."""
    workers = max(1, int(workers))
    loads = [0.0] * workers
    for duration in sorted((float(d) for d in durations), reverse=True):
        index = min(range(workers), key=lambda i: loads[i])
        loads[index] += duration
    return max(loads) if loads else 0.0


def pack_arrays(arrays):
    """Compress the three released arrays for shipping to a container."""
    packed = {}
    for name in ARRAY_NAMES:
        array = np.ascontiguousarray(arrays[name])
        packed[name] = {'bytes': zlib.compress(array.tobytes(), 1),
                        'dtype': str(array.dtype), 'shape': list(array.shape),
                        'sha256': ahash(array)}
    return packed


def unpack_arrays(payload):
    """Inverse of :func:`pack_arrays`; ``run_fit`` re-checks every hash."""
    arrays = {}
    for name in ARRAY_NAMES:
        blob = payload[name]
        flat = np.frombuffer(zlib.decompress(blob['bytes']), dtype=np.dtype(blob['dtype']))
        arrays[name] = np.ascontiguousarray(flat.reshape(tuple(blob['shape'])))
    return arrays


def job_payload(job, study):
    """The controller's canonical arrays for one job, plus their shipping form.

    ``study.job_arrays`` is the single source of truth: the hashes in the job
    dict, the arrays the learner is handed and the arrays ``score.verify_record``
    re-derives all come from it, so no cross-machine numerical agreement is
    assumed anywhere.
    """
    arrays = study.job_arrays(job['seed'], job['n'])
    for name in ARRAY_NAMES:
        assert ahash(arrays[name]) == job['input_sha256'][name], (
            f"Regenerated {name} does not match job {job['id']}")
    return pack_arrays(arrays), arrays


def resolve_configs(jobs):
    """Resolve every config through its learner module BEFORE any paid dispatch.

    Returns ``(resolved_module_names, missing_module_names)``.  A module not yet on
    disk (kernels.py / neural.py are written concurrently) is reported as missing
    rather than raising, so ``--validate-only`` still works; ``main`` refuses to
    dispatch while anything is missing.
    """
    sys.path.insert(0, str(ROOT))
    resolved, missing = {}, set()
    for job in jobs:
        name = learner_module_name(job['config'])
        if name in missing:
            continue
        if name not in resolved:
            if not (ROOT / (name + '.py')).exists():
                missing.add(name)
                continue
            resolved[name] = __import__(name)
        resolved[name].resolve_config(job['config'])
    return sorted(resolved), sorted(missing)


def check_arrays(job, study, pool=None):
    """Regenerate the job's arrays two independent ways and check both hashes.

    Controller-side only, and both ways run in THIS process: the comparison is of
    two implementations, not of two machines.
    """
    payload, arrays = job_payload(job, study)
    pool = np.load(POOL_PIXELS) if pool is None else pool
    rebuilt, train_rows, query_rows = rebuild_arrays(
        pool, job['seed'], job['n'], arrays['train_y'])
    for name in ARRAY_NAMES:
        assert ahash(rebuilt[name]) == job['input_sha256'][name], (
            f"runner.rebuild_arrays disagrees with study.job_arrays on {name} "
            f"for job {job['id']}")
    assert ahash(train_rows) == job['train_rows_sha256']
    assert ahash(query_rows) == job['query_rows_sha256']
    return payload, arrays


def preflight(jobs, pending, sources, plan_path, authorization, work_seconds, study,
              missing, gpus, gpu_name, ledger_path):
    import budget
    from budget import read_ledger

    summary = {'charged_upper_usd': 0.0,
               'available_worker_usd': float(authorization['total_modal_cap_usd']
                                             - authorization['contingency_usd']),
               'active_ids': [], 'blocked_reason': None}
    if Path(ledger_path).exists():
        before = sha(ledger_path)
        summary = read_ledger(ledger_path)['summary']
        assert sha(ledger_path) == before, 'Reading the ledger changed it'
    assert not summary['active_ids'] and not summary['blocked_reason'], (
        'A prior app is unresolved or the ledger is blocked')
    rate = float(budget.WORKER_RATES[gpu_name])
    projected = (work_seconds + SHUTDOWN_GRACE_SECONDS) * gpus * rate
    assert projected <= summary['available_worker_usd'] + 1e-9, (
        'Reservation exceeds the remaining authorized allowance')
    modules, unresolved = resolve_configs(jobs)
    missing = sorted(set(missing) | {name + '.py' for name in unresolved})
    pool = np.load(POOL_PIXELS)
    for job in pending:
        check_arrays(job, study, pool)
    del pool
    estimates = [float(job['estimated_seconds']) for job in pending]
    makespan = lpt_makespan(estimates, gpus)
    manifest = json.loads((ROOT / 'raw/data_manifest.json').read_text())
    return {
        'passed': not missing, 'missing_sources': missing,
        'checked_at': utc(), 'plan': str(Path(plan_path).name),
        'plan_sha256': sha(plan_path), 'jobs': len(jobs), 'pending': len(pending),
        'pending_ids': [job['id'] for job in pending],
        'learner_modules': modules, 'unresolved_modules': unresolved,
        'source_sha256': sources,
        'data_manifest_sha256': sha(ROOT / 'raw/data_manifest.json'),
        'data_manifest_created_at': manifest.get('created_at_utc'),
        'gpu': gpu_name, 'max_concurrent_gpus': gpus,
        'worker_usd_per_second': rate,
        'work_seconds': work_seconds, 'shutdown_grace_seconds': SHUTDOWN_GRACE_SECONDS,
        'estimated_total_gpu_seconds': round(sum(estimates), 1),
        'estimated_lpt_makespan_seconds': round(makespan, 1),
        'estimated_lpt_makespan_minutes': round(makespan / 60.0, 2),
        'reservation_upper_usd': round(projected, 6),
        'estimated_cost_at_makespan_usd': round(
            (makespan + SHUTDOWN_GRACE_SECONDS) * gpus * rate, 6),
        'available_worker_usd': summary['available_worker_usd'],
        'new_reservation_created': False, 'paid_compute_started': False,
        'query_labels_opened': False,
    }


def save_outputs(job, record, logits, labels):
    predictions = ROOT / 'predictions' / f"{job['id']}.npz"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions.with_suffix('.npz.part')
    with temporary.open('wb') as handle:
        np.savez(handle, logits=logits, labels=labels)
    temporary.replace(predictions)
    record['predictions_path'] = str(predictions.relative_to(ROOT))
    record['predictions_sha256'] = sha(predictions)
    write_json(ROOT / 'results' / f"{job['id']}.json", record)
    metrics = record.get('metrics') or {}
    print(json.dumps({'id': job['id'], 'status': 'completed',
                      'device': record.get('device'),
                      'fit_wall_seconds': round(float(metrics.get('fit_wall_seconds', 0)), 2),
                      'truncated': metrics.get('truncated')}), flush=True)


# --------------------------------------------------------------------------
# local CPU runner (free)
# --------------------------------------------------------------------------
def run_local(jobs, pending, sources, plan_path, study):
    pool = np.load(POOL_PIXELS)
    provenance_base = {
        'source_sha256': sources,
        'image_source_sha256': {},
        'plan_sha256': sha(plan_path),
        'pool_pixels_sha256': ahash(pool),
        'data_manifest_sha256': sha(ROOT / 'raw/data_manifest.json'),
        'authorization_sha256': sha(ROOT / 'authorization.json'),
        'budget_reservation': None, 'app_id': None, 'image': None,
        'runner_mode': 'local-cpu',
        'controller_python': platform.python_version(),
    }
    failures = 0
    for job in pending:
        provenance = dict(provenance_base, dispatched_at=utc())
        deadline = time.time() + float(job['time_budget_seconds'])
        try:
            _, arrays = check_arrays(job, study, pool)
            record, logits, predicted = run_fit(arrays, job, provenance,
                                                deadline, 'cpu')
            save_outputs(job, record, logits, predicted)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            failures += 1
            write_json(ROOT / 'results' / f"{job['id']}-failed.json",
                       {'id': job['id'], 'status': 'failed', 'job': job,
                        'error': f'{type(error).__name__}: {error}', 'failed_at': utc(),
                        'runner_mode': 'local-cpu'})
            print(json.dumps({'id': job['id'], 'status': 'failed',
                              'error': f'{type(error).__name__}: {error}'}), flush=True)
    print(json.dumps({'local_jobs': len(pending), 'failed': failures}), flush=True)
    return 1 if failures else 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpu', choices=sorted(GPU_KINDS), default='a100')
    parser.add_argument('--gpus', type=int, default=MAX_GPUS,
                        help='concurrent fits to dispatch and to charge for (1..8)')
    parser.add_argument('--minutes', type=float, default=20.0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--local', action='store_true',
                        help='run the plan sequentially on this CPU; no app, no ledger')
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    import study

    authorization = json.loads((ROOT / 'authorization.json').read_text())
    assert authorization['status'] == 'frozen'
    gpu_name = GPU_KINDS[args.gpu]
    assert gpu_name in authorization['allowed_gpus'], f'{gpu_name} is not authorized'
    gpus = int(args.gpus)
    assert 1 <= gpus <= MAX_GPUS, f'--gpus must be in 1..{MAX_GPUS}'
    assert gpus <= int(authorization['max_concurrent_gpus'])
    assert gpus == MAX_CONTAINERS, (
        'The container pool is fixed at import time from --gpus; pass --gpus once')

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = (ROOT / plan_path)
    plan_path = plan_path.resolve()
    assert plan_path.is_relative_to(ROOT), 'Plan must belong to this study'
    jobs = load_plan(plan_path)
    for job in jobs:
        assert int(job['time_budget_seconds']) <= int(
            authorization['per_fit_time_budget_seconds'])
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES if (ROOT / name).exists()}
    sources.update({name: sha(ROOT / name) for name in ('runner.py', 'budget.py', 'study.py')})
    needed = sorted({learner_module_name(job['config']) + '.py' for job in jobs}
                    | {'release.py'})
    missing = [name for name in needed if not (ROOT / name).exists()]
    for name in ('results', 'predictions', 'plans', 'logs'):
        (ROOT / name).mkdir(exist_ok=True)
    pending = [job for job in jobs if not verify_existing(job, sources)]

    if args.local:
        assert not missing, f'Missing required sources: {missing}'
        assert all(job['device_kind'] == 'cpu' for job in pending), (
            '--local only runs jobs planned with device_kind="cpu"')
        _, unresolved = resolve_configs(jobs)
        assert not unresolved, f'Missing learner modules: {unresolved}'
        if not pending:
            print('All planned jobs already complete.')
            return 0
        return run_local(jobs, pending, sources, plan_path, study)

    assert modal is not None, 'Modal is disabled in this process (RELEASE_LADDER_NO_MODAL=1)'
    ledger_path = ROOT / 'budget-ledger.json'
    global_deadline = datetime.fromisoformat(
        authorization['deadline_utc'].replace('Z', '+00:00')).timestamp()
    work_seconds = min(int(math.ceil(args.minutes * 60)),
                       int(global_deadline - time.time()) - SHUTDOWN_GRACE_SECONDS)
    assert work_seconds > 0, 'Authorization deadline reached; no further launches'
    receipt = preflight(jobs, pending, sources, plan_path, authorization, work_seconds,
                        study, missing, gpus, gpu_name, ledger_path)
    if args.validate_only:
        write_json(ROOT / 'logs' / f'{plan_path.stem}-preflight.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'},
                         indent=2, sort_keys=True))
        return 0
    missing = receipt['missing_sources']
    assert not missing, f'Missing required sources: {missing}'
    unmounted = [name for name in needed if name not in ADDED_FILES]
    assert not unmounted, f'Required sources are not in the image: {unmounted}'
    assert all(job['device_kind'] == 'gpu' for job in pending), (
        'A GPU dispatch may only contain device_kind="gpu" jobs; use --local for the rest')
    if not pending:
        print('All planned jobs already complete.')
        return 0

    from budget import reserve_app, attach_app, finish_app

    reservation_id = f"{plan_path.stem}-{args.gpu}-" + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    reservation = reserve_app(ledger_path, reservation_id,
                              total_budget_usd=float(authorization['total_modal_cap_usd']),
                              work_seconds=work_seconds,
                              shutdown_grace_seconds=SHUTDOWN_GRACE_SECONDS,
                              containers=gpus, gpu=gpu_name)
    deadline = float(reservation['work_deadline_unix'])
    pool_sha = ahash(np.load(POOL_PIXELS))
    write_json(ROOT / 'logs' / f'{reservation_id}-plan.json',
               {'pending_ids': [job['id'] for job in pending], 'source_sha256': sources,
                'reservation': dict(reservation), 'preflight': receipt,
                'plan_sha256': sha(plan_path), 'created_at': utc()})

    worker = fit_a100 if gpu_name == 'A100-40GB' else fit_t4
    calls, lock, aborted = [], Lock(), Event()
    outcomes = []
    app_id, failure, timer = None, None, None

    def stop_own_app():
        aborted.set()
        active = app_id or getattr(app, 'app_id', None)
        if active:
            try:
                subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', active, '--yes'],
                               timeout=60, check=True)
            except Exception as error:                        # pragma: no cover
                print('STOP ERROR ' + str(error), flush=True)
        with lock:
            known = list(calls)
        for call in known:
            try:
                call.cancel(terminate_containers=True)
            except Exception as error:                        # pragma: no cover
                print('CANCEL ERROR ' + str(error), flush=True)

    def skip(job, reason):
        print(json.dumps({'id': job['id'], 'status': 'skipped', 'reason': reason}), flush=True)
        return {'id': job['id'], 'status': 'skipped', 'reason': reason}

    def run_one(job):
        # Never raise for a job-level problem: an exception here would abort the
        # pool and cancel a sibling container, destroying a nearly finished fit.
        try:
            return _run_one(job)
        except (KeyboardInterrupt, SystemExit):
            aborted.set()
            stop_own_app()
            raise
        except Exception as error:
            record = {'id': job['id'], 'status': 'failed',
                      'error': f'{type(error).__name__}: {error}', 'job': job,
                      'failed_at': utc(), 'budget_reservation': reservation_id,
                      'app_id': app_id}
            write_json(ROOT / 'results' / f"{job['id']}-failed.json", record)
            print(json.dumps({'id': job['id'], 'status': 'failed',
                              'error': record['error']}), flush=True)
            return record

    def _run_one(job):
        if aborted.is_set():
            return skip(job, 'run aborted')
        if time.time() > deadline:
            return skip(job, 'work deadline reached before dispatch')
        payload, _ = job_payload(job, study)
        provenance = {
            'source_sha256': sources,
            'image_source_sha256': {name: sources[name] for name in ADDED_FILES},
            'plan_sha256': sha(plan_path),
            'pool_pixels_sha256': pool_sha,
            'data_manifest_sha256': sha(ROOT / 'raw/data_manifest.json'),
            'authorization_sha256': sha(ROOT / 'authorization.json'),
            'budget_reservation': reservation_id, 'app_id': app_id,
            'global_deadline_unix': global_deadline, 'image': IMAGE, 'gpu': gpu_name,
            'runner_mode': 'modal-' + args.gpu,
            'controller_python': platform.python_version(),
            'dispatched_at': utc(),
        }
        if time.time() > deadline - 30:
            return skip(job, 'no time left to dispatch')
        call = worker.spawn(job, payload, provenance, deadline - 10)
        with lock:
            calls.append(call)
        try:
            record, logits_bytes, labels_bytes = call.get(
                timeout=max(1, int(deadline - time.time())))
            logits = np.frombuffer(logits_bytes, dtype=np.float32).reshape(
                record['output_specs']['logits']['shape'])
            labels = np.frombuffer(labels_bytes, dtype=np.uint8).reshape(
                record['output_specs']['labels']['shape'])
            assert ahash(logits) == record['output_sha256']['logits']
            assert ahash(labels) == record['output_sha256']['labels']
            assert record['job'] == job, 'The container returned a different job dict'
            save_outputs(job, record, logits, labels)
            return {'id': job['id'], 'status': 'completed'}
        except BaseException:
            try:
                call.cancel(terminate_containers=True)
            except Exception as error:                        # pragma: no cover
                print('CANCEL ERROR ' + str(error), flush=True)
            raise
        finally:
            with lock:
                if call in calls:
                    calls.remove(call)

    # Longest first: the LPT order the makespan estimate assumes.
    ordered = sorted(pending, key=lambda job: -float(job['estimated_seconds']))
    try:
        timer = Timer(max(0.0, deadline - time.time()), stop_own_app)
        timer.daemon = True
        timer.start()
        with modal.enable_output(), app.run():
            app_id = app.app_id
            attach_app(ledger_path, reservation_id, app_id)
            if aborted.is_set() or time.time() > deadline:
                raise TimeoutError('App startup exceeded the work deadline')
            print(json.dumps({'app_id': app_id, 'pending': len(ordered), 'gpu': gpu_name,
                              'deadline_unix': deadline, 'containers': gpus}), flush=True)
            with ThreadPoolExecutor(max_workers=gpus) as pool_executor:
                futures = [pool_executor.submit(run_one, job) for job in ordered]
                try:
                    for future in as_completed(futures):
                        outcomes.append(future.result())
                except BaseException:
                    aborted.set()
                    for future in futures:
                        future.cancel()
                    stop_own_app()
                    raise
    except BaseException as error:
        failure = error
    finally:
        if timer is not None:
            timer.cancel()
        if app_id:
            if failure is not None:
                stop_own_app()
            row = None
            # budget.finish_app charges wall clock to the moment the stop is
            # verified, so this loop must finish inside SHUTDOWN_GRACE_SECONDS
            # (2 + 7*15 = 107 s of sleeps in the normal case) or the ledger
            # blocks itself on an overrun it caused.
            verify_until = deadline + SHUTDOWN_GRACE_SECONDS - 30
            for attempt in range(8):
                try:
                    row = stopped_row(app_id)
                except Exception as error:                    # pragma: no cover
                    print('LIST ERROR ' + str(error), flush=True)
                if row:
                    break
                if attempt == 0:
                    stop_own_app()
                if time.time() > verify_until:                # pragma: no cover
                    print('SHUTDOWN VERIFICATION OUT OF GRACE', flush=True)
                    break
                time.sleep(15 if attempt else 2)
            unfinished_jobs = [o for o in outcomes if o.get('status') != 'completed']
            if failure is not None:
                outcome = 'failed'
            elif unfinished_jobs or len(outcomes) < len(ordered):
                outcome = 'partial'
            else:
                outcome = 'completed'
            evidence = ROOT / 'logs' / f'{reservation_id}-stopped.json'
            write_json(evidence, {'app_id': app_id, 'provider_row': row,
                                  'checked_at': utc(), 'job_outcomes': outcomes,
                                  'outcome': outcome})
            if row:
                settled = finish_app(ledger_path, reservation_id, verified_stopped=True,
                                     remaining_tasks=0, app_state='stopped',
                                     evidence=str(evidence), outcome=outcome)
                print(json.dumps({'budget': settled['summary']}), flush=True)
            else:
                print('UNVERIFIED SHUTDOWN: reservation retained', flush=True)
        if failure is not None:
            raise failure
    unfinished = [o for o in outcomes if o.get('status') != 'completed']
    if unfinished:
        print(json.dumps({'unfinished': unfinished}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
