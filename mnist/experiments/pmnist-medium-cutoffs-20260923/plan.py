"""Build plan files (lists of study.make_job dicts) for runner.py.

    python plan.py smoke
    python plan.py dev --candidates plans/candidates.json \
        --levels 1000,10000 --seeds 2026092301,2026092302
    python plan.py final --selection selection.json

Every job is produced by ``study.make_job``; plan.py never touches the GPU,
never reads labels for the query rows and never creates a Modal app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PLANS = ROOT / 'plans'
DEFAULT_LEARNER_SEED = 11


def _resolve_config(config):
    """Reject a bad candidate config here, before any paid dispatch."""
    sys.path.insert(0, str(ROOT))
    import learners
    return learners.resolve_config(config)


SMOKE_CANDIDATES = [
    {'id': 'smoke-mlp',
     'config': {'family': 'mlp', 'widths': [256], 'epochs': 3, 'members': 2,
                'batch_size': 128, 'lr': 0.001}},
    {'id': 'smoke-ladder',
     'config': {'family': 'ladder', 'hidden_dims': [250, 250], 'epochs': 2,
                'decay_start_epoch': 1, 'batch_size': 100, 'members': 1}},
    {'id': 'smoke-topo',
     'config': {'family': 'topo_cnn', 'epochs': 2, 'member_seeds': [101]}},
]
SMOKE_SEED = 2026092301
SMOKE_N = 1000
SMOKE_TIME_BUDGET = 300


def _study():
    sys.path.insert(0, str(ROOT))
    import study
    return study


def _write(path, jobs):
    PLANS.mkdir(exist_ok=True)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(jobs, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)
    print(json.dumps({'plan': str(path.relative_to(ROOT)), 'jobs': len(jobs),
                      'ids': [job['id'] for job in jobs]}, indent=2))
    return path


def _candidates(payload):
    """Accept either a bare list or {'candidates': [...]} with id/config keys."""
    entries = payload['candidates'] if isinstance(payload, dict) else payload
    out = []
    for entry in entries:
        assert 'id' in entry and 'config' in entry, 'Each candidate needs id and config'
        if entry.get('runner', 'gpu') != 'gpu':
            # Classical (CPU) candidates live in the same committed selection but
            # are planned by run_classical.py; they have no GPU learner family.
            print(json.dumps({'candidate': entry['id'], 'skipped': 'classical runner'}))
            continue
        _resolve_config(entry['config'])
        out.append(entry)
    ids = [entry['id'] for entry in out]
    assert len(set(ids)) == len(ids), 'Duplicate candidate ids'
    return out


def build_smoke(_args):
    study = _study()
    for candidate in SMOKE_CANDIDATES:
        _resolve_config(candidate['config'])
    jobs = [study.make_job('smoke', SMOKE_SEED, SMOKE_N, candidate['id'],
                           candidate['config'], DEFAULT_LEARNER_SEED,
                           time_budget_seconds=SMOKE_TIME_BUDGET)
            for candidate in SMOKE_CANDIDATES]
    return _write(PLANS / 'smoke.json', jobs)


def build_dev(args):
    study = _study()
    candidates = _candidates(json.loads(Path(args.candidates).read_text()))
    levels = ([int(value) for value in args.levels.split(',')] if args.levels
              else list(study.LEVELS))
    seeds = ([int(value) for value in args.seeds.split(',')] if args.seeds
             else list(study.DEV_SEEDS))
    for n in levels:
        assert n in study.LEVELS, f'{n} is not a study level'
    for seed in seeds:
        assert seed in study.DEV_SEEDS, f'{seed} is not a dev seed'
    jobs = [study.make_job('dev', seed, n, candidate['id'], candidate['config'],
                           int(args.learner_seed),
                           time_budget_seconds=int(args.time_budget))
            for candidate in candidates for seed in seeds for n in levels]
    return _write(PLANS / args.out, jobs)


def build_final(args):
    study = _study()
    selection = json.loads(Path(args.selection).read_text())
    learner_seed = int(selection.get('learner_seed', DEFAULT_LEARNER_SEED))
    jobs = []
    for candidate in _candidates(selection):
        levels = [int(value) for value in candidate.get('levels', study.LEVELS)]
        for n in levels:
            assert n in study.LEVELS, f'{n} is not a study level'
            for seed in study.FINAL_SEEDS:
                jobs.append(study.make_job('final', seed, n, candidate['id'],
                                           candidate['config'], learner_seed,
                                           time_budget_seconds=int(args.time_budget)))
    assert jobs, 'selection.json produced no jobs'
    return _write(PLANS / args.out, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    smoke = sub.add_parser('smoke', help='three tiny jobs, one per family')
    smoke.set_defaults(handler=build_smoke)

    dev = sub.add_parser('dev', help='model selection over dev seeds')
    dev.add_argument('--candidates', required=True)
    dev.add_argument('--levels', default=None)
    dev.add_argument('--seeds', default=None)
    dev.add_argument('--learner-seed', type=int, default=DEFAULT_LEARNER_SEED)
    dev.add_argument('--time-budget', type=int, default=1200)
    dev.add_argument('--out', default='dev.json')
    dev.set_defaults(handler=build_dev)

    final = sub.add_parser('final', help='selected candidates over every final seed')
    final.add_argument('--selection', default='selection.json')
    final.add_argument('--time-budget', type=int, default=1200)
    final.add_argument('--out', default='final.json')
    final.set_defaults(handler=build_final)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
