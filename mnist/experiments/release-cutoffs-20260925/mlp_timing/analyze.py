"""The fastest MLP per cutoff, by the rule frozen in ../protocol.draft.json (mlp_timing_rule).

    /tmp/penv/bin/python analyze.py            # after both sweeps and ../cutoffs.py

For each cutoff c (../analysis/cutoffs.json) and each family -- eager (results/sweep.json,
the registered family) and graph-captured (results/sweep-graphed.json, the same MLPs
with each training step replayed from a CUDA graph; added after the eager sweep had
started, before any cutoff was known) -- among configurations whose mean per-call time
is within the harness's 60 s limit and whose pooled dev error is at most c - 0.15
points, the one with the lowest mean time. The two sweeps ran in different containers;
the eager configurations both sweeps share give the ratio of the two hosts' speeds.
Writes results/picks.json, results/picks.md and one harness-ready file per pick under
submissions/, for confirmation in the popcorn3 harness.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARGIN_PP = 0.15
LIMIT_MS = 60000.0
sys.path.insert(0, str(HERE))
import mlp_family  # noqa: E402


def frontier(rows):
    best, out = float('inf'), []
    for row in sorted(rows, key=lambda r: (r['mean_ms'], r['error_pct'])):
        if row['error_pct'] < best:
            best = row['error_pct']
            out.append(row)
    return out


def pick(rows, cutoff):
    ok = sorted((r for r in rows if r['error_pct'] <= cutoff - MARGIN_PP and r['mean_ms'] <= LIMIT_MS),
                key=lambda r: r['mean_ms'])
    return ok


def main(cutoffs_path=ROOT / 'analysis' / 'cutoffs.json', suffix=''):
    eager = json.loads((HERE / 'results' / 'sweep.json').read_text())
    graphed = json.loads((HERE / 'results' / 'sweep-graphed.json').read_text())
    families = {'eager': [r for r in eager['rows'] if not r.get('graphed')],
                'graphed': [r for r in graphed['rows'] if r.get('graphed')]}
    shared = {r['name']: r for r in graphed['rows'] if not r.get('graphed')}
    ratios = [shared[r['name']]['mean_ms'] / r['mean_ms'] for r in families['eager'] if r['name'] in shared]
    host_ratio = statistics.median(ratios) if ratios else None
    cutoffs = json.loads(Path(cutoffs_path).read_text())['cutoffs']
    picks = []
    for level in cutoffs:
        entry = {'level': level['level'], 'n': level['n'], 'cutoff_pct': level['pooled_error_pct'],
                 'band_bp': level['bp_ceiling']}
        for family, rows in families.items():
            ok = pick(rows, level['pooled_error_pct'])
            entry[family] = ({'choice': ok[0]['name'], 'dev_error_pct': ok[0]['error_pct'],
                              'dev_mean_ms': ok[0]['mean_ms'], 'runner_up': ok[1]['name'] if len(ok) > 1 else None}
                             if ok else None)
        picks.append(entry)
    best = {family: min((r for r in rows if r['mean_ms'] <= LIMIT_MS), key=lambda r: r['error_pct'])
            for family, rows in families.items()}
    out = {'margin_pp': MARGIN_PP, 'limit_ms': LIMIT_MS, 'picks': picks,
           'host_ratio_graphed_container_over_eager_container': host_ratio, 'host_ratio_samples': ratios,
           'containers': {name: {k: sweep[k] for k in ('device', 'host', 'gpu_clock', 'draws', 'dev_seed')}
                          for name, sweep in (('eager', eager), ('graphed', graphed))},
           'best_under_limit': {f: {k: r[k] for k in ('name', 'error_pct', 'mean_ms')} for f, r in best.items()},
           'frontier': {f: [{k: r[k] for k in ('name', 'error_pct', 'mean_ms')} for r in frontier(rows)]
                        for f, rows in families.items()}}
    out['cutoffs_file'] = str(cutoffs_path)
    (HERE / 'results' / f'picks{suffix}.json').write_text(json.dumps(out, indent=2) + '\n')

    folder = HERE / 'submissions'
    folder.mkdir(exist_ok=True)
    for entry in picks:
        for family in families:
            if entry[family]:
                label = entry[family]['choice']
                (folder / f'{label}.py').write_text(mlp_family.source(*mlp_family.parse(label)))

    def cell(choice):
        if not choice:
            return 'none within 60 s | |'
        return f"`{choice['choice']}`, {choice['dev_error_pct']:.2f}% | {choice['dev_mean_ms']:,.0f} ms"

    lines = ['| Level | Labels N | Cutoff | Eager MLP (dev error) | Time | Graph-captured MLP (dev error) | Time |',
             '| ---: | ---: | ---: | --- | ---: | --- | ---: |']
    for e in picks:
        lines.append(f"| {e['level']} | {e['n']:,} | {e['cutoff_pct']:.3f}% | {cell(e['eager'])} | {cell(e['graphed'])} |")
    lines += ['', f"Host speed ratio, graphed container / eager container, on shared eager configurations: "
                  f"{host_ratio:.3f} ({', '.join(f'{r:.3f}' for r in ratios)})" if host_ratio else '']
    for family, row in best.items():
        lines.append(f"Best {family} MLP under 60 s: `{row['name']}`, {row['error_pct']:.2f}% at {row['mean_ms']:,.0f} ms.")
    for family, rows in families.items():
        lines += ['', f'{family.capitalize()} frontier:', '', '| Configuration | Dev error | Time per call |',
                  '| --- | ---: | ---: |']
        lines += [f"| `{r['name']}` | {r['error_pct']:.2f}% | {r['mean_ms']:,.1f} ms |" for r in frontier(rows)]
    (HERE / 'results' / f'picks{suffix}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoffs', default=str(ROOT / 'analysis' / 'cutoffs.json'))
    parser.add_argument('--suffix', default='', help="e.g. -24k writes results/picks-24k.json")
    args = parser.parse_args()
    main(args.cutoffs, args.suffix)
