"""Summarise the Ladder timing sweep: per step budget, and the time to reach each cutoff.

    /tmp/penv/bin/python summarize.py        # writes results/summary.json

Per step budget: the error pooled over the five dev draws and the median, fastest and
slowest per-call time over the five hosts. Time to reach a cutoff c: where the Ladder's
pooled error curve reaches c, interpolated between the two measured budgets that bracket
it (error linear, time and steps logarithmic); "at most" the smallest budget when even it
is below c. The budgets double, so a cutoff that falls just short of a measured budget
would otherwise jump a whole doubling on a 0.01-point difference.
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUTOFFS = HERE.parent / 'analysis' / 'cutoffs.json'


def summarise(sweep, cutoffs):
    containers = sweep['containers']
    steps = sorted({r['steps'] for c in containers for r in c['rows']})
    curve = []
    for s in steps:
        rows = [r for c in containers for r in c['rows'] if r['steps'] == s]
        secs = [r['seconds'] for r in rows]
        curve.append({'steps': s, 'draws': len(rows),
                      'error_pct': 100.0 * (1 - sum(r['correct'] for r in rows) / (10000.0 * len(rows))),
                      'ms': 1000.0 * statistics.median(secs), 'ms_min': 1000.0 * min(secs),
                      'ms_max': 1000.0 * max(secs)})

    def reach(c):
        if curve[0]['error_pct'] <= c:
            return {'kind': 'at most', 'ms': curve[0]['ms'], 'steps': curve[0]['steps']}
        for a, b in zip(curve, curve[1:]):
            if a['error_pct'] > c >= b['error_pct']:
                f = (a['error_pct'] - c) / (a['error_pct'] - b['error_pct'])
                return {'kind': 'about', 'ms': a['ms'] * (b['ms'] / a['ms']) ** f,
                        'steps': a['steps'] * (b['steps'] / a['steps']) ** f, 'between': [a['steps'], b['steps']]}
        return None

    reached = {str(r['n']): reach(r['pooled_error_pct']) for r in cutoffs}
    return {'curve': curve, 'reach': reached,
            'hosts': [{'draw': c['draw_index'], 'device': c['device'], 'host': c['host'], 'gpu_clock': c['gpu_clock'],
                       'reference_mlp_ms': c['reference_mlp']['ms']} for c in containers]}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoffs', default=str(CUTOFFS))
    parser.add_argument('--out', default=str(HERE / 'results' / 'summary.json'))
    args = parser.parse_args()
    sweep = json.loads((HERE / 'results' / 'ladder-sweep.json').read_text())
    cutoffs = json.loads(Path(args.cutoffs).read_text())['cutoffs']
    out = summarise(sweep, cutoffs)
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    for p in out['curve']:
        print(f"{p['steps']:>6} steps: {p['error_pct']:.2f}% error, median {p['ms'] / 1000:.1f} s "
              f"({p['ms_min'] / 1000:.1f}-{p['ms_max'] / 1000:.1f} s)")
    for n, r in out['reach'].items():
        print(n, r and f"{r['kind']} {r['ms'] / 1000:.1f} s, {r['steps']:.0f} steps")
