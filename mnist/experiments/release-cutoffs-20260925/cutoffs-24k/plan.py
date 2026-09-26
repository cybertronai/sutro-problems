"""Build plan files (lists of ``study.make_job`` dicts) for runner.py.

    python plan.py smoke
    python plan.py dev --candidates plans/candidates.json --levels 500,10000
    python plan.py final --selection selection.json

Every job is produced by ``study.make_job``, so it carries the hashes of the
exact arrays the learner will see.  On top of the study contract each job also
carries the two scheduling fields the runner needs:

  ``device_kind``        'gpu' (a Modal A100/T4 fit) or 'cpu' (free, local)
  ``estimated_seconds``  the planner's per-candidate estimate for that level,
                         used for the LPT makespan and the dollar estimate that
                         ``runner.py --validate-only`` prints BEFORE any
                         reservation is made.  It is an estimate only: the
                         binding limit is ``time_budget_seconds``.

A candidate entry is ``{"id", "config"}`` plus optionally
``"device_kind"``, ``"levels"``, ``"learner_seed"``, ``"time_budget_seconds"``
and ``"estimated_seconds"``.  ``estimated_seconds`` is either a number (the same
estimate at every level) or a table keyed by level, e.g.
``{"500": 40, "10000": 300}``; missing levels are interpolated from the nearest
supplied ones by the cost model ``seconds(n) = a + b * n`` fitted on the table
(one entry -> constant), and the result is clipped to ``time_budget_seconds``.

plan.py never touches a GPU, never reads a query label and never creates a
Modal app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PLANS = ROOT / 'plans'
DEFAULT_LEARNER_SEED = 11
DEFAULT_TIME_BUDGET = 1200
NEURAL_FAMILIES = ('mlp', 'ladder', 'vat')
DEFAULT_ESTIMATE_SECONDS = 120.0

# One tiny job per learner FAMILY, at the smallest level on a dev seed.  The
# smoke plan is the integration test of the whole pipeline (plan -> runner
# --local -> score -> score --ensemble), so every candidate is
# ``device_kind='cpu'``: ``runner.py --local`` refuses a job planned for a GPU,
# and paying an A100 to discover that a config key is misspelled is the one
# thing the smoke stage exists to prevent.  The grids are cut to a single point
# and the epoch counts to a handful so the whole plan finishes in minutes on a
# laptop CPU -- these configs are NOT meant to be accurate.
SMOKE_CANDIDATES = [
    # kernels.py
    {'id': 'krr', 'device_kind': 'cpu', 'estimated_seconds': 20,
     'config': {'family': 'krr', 'kernel': 'laplace', 'bandwidth_scales': [1.0],
                'ridge_grid': [1e-6], 'cv_folds': 2, 'cv_max_rows': 250,
                'median_sample': 250}},
    {'id': 'rfm', 'device_kind': 'cpu', 'estimated_seconds': 40,
     'config': {'family': 'rfm', 'iters': 1, 'bandwidth_grid': [10.0],
                'ridge_grid': [1e-6], 'holdout_max_rows': 400}},
    {'id': 'labelprop', 'device_kind': 'cpu', 'estimated_seconds': 60,
     'config': {'family': 'labelprop', 'k_grid': [10], 'alpha_grid': [0.9],
                'cv_folds': 2, 'spread_iters': 10}},
    # neural.py
    {'id': 'mlp', 'device_kind': 'cpu', 'estimated_seconds': 30,
     'config': {'family': 'mlp', 'widths': [256], 'epochs': 3, 'members': 2,
                'batch_size': 128, 'lr': 0.001}},
    {'id': 'vat', 'device_kind': 'cpu', 'estimated_seconds': 60,
     'config': {'family': 'vat', 'widths': [256], 'epochs': 3, 'members': 1,
                'batch_size': 128, 'lr': 0.001,
                'vat': {'rampup_fraction': 0.5, 'power_iterations': 1}}},
    {'id': 'ladder', 'device_kind': 'cpu', 'estimated_seconds': 60,
     'config': {'family': 'ladder', 'hidden_dims': [100, 50], 'epochs': 2,
                'decay_start_epoch': 1, 'members': 1}},
]
SMOKE_SEED = 2026092491
SMOKE_N = 500
SMOKE_TIME_BUDGET = 300


def _study():
    sys.path.insert(0, str(ROOT))
    import study
    return study


def _module_for(config):
    """Same rule as runner.learner_module_name, without importing modal."""
    name = config.get('module')
    if name is not None:
        if name not in ('kernels', 'neural'):
            raise ValueError(f"config['module'] must be 'kernels' or 'neural', got {name!r}")
        return name
    return 'neural' if config.get('family') in NEURAL_FAMILIES else 'kernels'


def _resolve_config(config, skip=False):
    """Reject a bad candidate config here, before any paid dispatch."""
    name = _module_for(config)
    if not (ROOT / f'{name}.py').exists():
        if skip:
            print(json.dumps({'warning': f'{name}.py absent; config not resolved',
                              'family': config.get('family')}))
            return config
        raise FileNotFoundError(
            f'{name}.py does not exist yet; pass --skip-resolve only while the learner '
            'modules are still being written')
    sys.path.insert(0, str(ROOT))
    return __import__(name).resolve_config(config)


def estimate_seconds(entry, n, time_budget):
    """Per-level estimate from the candidate's table, linear in ``n``, clipped."""
    table = entry.get('estimated_seconds', DEFAULT_ESTIMATE_SECONDS)
    if isinstance(table, dict):
        points = sorted((int(key), float(value)) for key, value in table.items())
        if not points:
            raise ValueError(f"candidate {entry['id']}: empty estimated_seconds table")
        if len(points) == 1:
            value = points[0][1]
        else:
            exact = dict(points)
            if int(n) in exact:
                value = exact[int(n)]
            else:
                # fit seconds(n) = a + b*n on the two bracketing (or nearest) points
                low = max([p for p in points if p[0] <= n], default=points[0])
                high = min([p for p in points if p[0] >= n], default=points[-1])
                if low[0] == high[0]:
                    value = low[1]
                else:
                    slope = (high[1] - low[1]) / (high[0] - low[0])
                    value = low[1] + slope * (int(n) - low[0])
    else:
        value = float(table)
    return float(max(1.0, min(float(time_budget), value)))


def _candidates(payload):
    entries = payload['candidates'] if isinstance(payload, dict) else payload
    out = []
    for entry in entries:
        if 'id' not in entry or 'config' not in entry:
            raise ValueError('each candidate needs "id" and "config"')
        entry = dict(entry)
        entry.setdefault('device_kind', 'gpu')
        if entry['device_kind'] not in ('gpu', 'cpu'):
            raise ValueError(f"candidate {entry['id']}: device_kind must be gpu or cpu")
        out.append(entry)
    ids = [entry['id'] for entry in out]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate candidate ids')
    return out


def _jobs_for(study, stage, entries, seeds, levels, skip_resolve, default_budget):
    jobs = []
    for entry in entries:
        _resolve_config(entry['config'], skip=skip_resolve)
        budget = int(entry.get('time_budget_seconds', default_budget))
        entry_levels = [int(v) for v in entry.get('levels', levels)]
        learner_seed = int(entry.get('learner_seed', DEFAULT_LEARNER_SEED))
        for n in entry_levels:
            if n not in study.LEVELS:
                raise ValueError(f'{n} is not a study level')
            estimate = estimate_seconds(entry, n, budget)
            for seed in seeds:
                jobs.append(study.make_job(
                    stage, seed, n, entry['id'], entry['config'],
                    learner_seed=learner_seed, time_budget_seconds=budget,
                    device_kind=entry['device_kind'],
                    extra={'estimated_seconds': estimate,
                           'learner_module': _module_for(entry['config'])}))
    if not jobs:
        raise ValueError('the candidate list produced no jobs')
    return jobs


def _write(path, jobs):
    PLANS.mkdir(exist_ok=True)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(jobs, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)
    gpu_seconds = sum(job['estimated_seconds'] for job in jobs if job['device_kind'] == 'gpu')
    cpu_seconds = sum(job['estimated_seconds'] for job in jobs if job['device_kind'] == 'cpu')
    print(json.dumps({'plan': str(path.relative_to(ROOT)), 'jobs': len(jobs),
                      'gpu_jobs': sum(1 for job in jobs if job['device_kind'] == 'gpu'),
                      'cpu_jobs': sum(1 for job in jobs if job['device_kind'] == 'cpu'),
                      'estimated_gpu_seconds': round(gpu_seconds, 1),
                      'estimated_cpu_seconds': round(cpu_seconds, 1)}, indent=2))
    return path


def build_smoke(args):
    study = _study()
    jobs = _jobs_for(study, 'smoke', _candidates(SMOKE_CANDIDATES), [SMOKE_SEED],
                     [SMOKE_N], args.skip_resolve, SMOKE_TIME_BUDGET)
    return _write(PLANS / args.out, jobs)


def build_dev(args):
    study = _study()
    entries = _candidates(json.loads(Path(args.candidates).read_text()))
    levels = ([int(v) for v in args.levels.split(',')] if args.levels
              else [study.LEVELS[0], study.LEVELS[-1]])
    seeds = ([int(v) for v in args.seeds.split(',')] if args.seeds else list(study.DEV_SEEDS))
    for seed in seeds:
        if seed not in study.DEV_SEEDS:
            raise ValueError(f'{seed} is not a dev seed')
    jobs = _jobs_for(study, 'dev', entries, seeds, levels, args.skip_resolve,
                     args.time_budget)
    return _write(PLANS / args.out, jobs)


def build_final(args):
    study = _study()
    selection = json.loads(Path(args.selection).read_text())
    entries = _candidates(selection)
    jobs = _jobs_for(study, 'final', entries, list(study.FINAL_SEEDS),
                     list(study.LEVELS), args.skip_resolve, args.time_budget)
    return _write(PLANS / args.out, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--skip-resolve', action='store_true',
                        help='allow planning before kernels.py/neural.py exist')
    sub = parser.add_subparsers(dest='command', required=True)

    smoke = sub.add_parser('smoke', help='two tiny jobs, one per learner module')
    smoke.add_argument('--out', default='smoke.json')
    smoke.set_defaults(handler=build_smoke)

    dev = sub.add_parser('dev', help='model selection over the dev seeds')
    dev.add_argument('--candidates', required=True)
    dev.add_argument('--levels', default=None, help='default: the two endpoint levels')
    dev.add_argument('--seeds', default=None)
    dev.add_argument('--time-budget', type=int, default=DEFAULT_TIME_BUDGET)
    dev.add_argument('--out', default='dev.json')
    dev.set_defaults(handler=build_dev)

    final = sub.add_parser('final', help='finalists at every level on every final seed')
    final.add_argument('--selection', default='selection.json')
    final.add_argument('--time-budget', type=int, default=DEFAULT_TIME_BUDGET)
    final.add_argument('--out', default='final.json')
    final.set_defaults(handler=build_final)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
