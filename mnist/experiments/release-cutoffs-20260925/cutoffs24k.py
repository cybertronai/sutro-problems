"""Five cutoffs from 200 to 10,000 labels, from the 24,000-step recipe.

    /tmp/penv/bin/python cutoffs24k.py      # after cutoffs-24k/score.py --stage final

Levels N = round(200 * 50**(i/4)) = 200, 532, 1,414, 3,761, 10,000. The recipe is
ladder-is06-long with full minibatches trained 24,000 steps, learner seed 11
(candidate ladder-xlong-s11). Sources: cutoffs-24k/results/scores_final.json for the
first four levels and below-2pct/results/scores_final.json for 10,000 labels, which is
the same recipe on the same seeds and inputs. Same statistics and band rule as
cutoffs.py. Writes analysis/cutoffs-24k.json and .md. Reads no labels.
"""
import json
from pathlib import Path

import cutoffs as base

ROOT = Path(__file__).resolve().parent
LEVELS = [200, 532, 1414, 3761, 10000]
RECIPE = 'ladder-xlong-s11'
SOURCES = [ROOT / 'cutoffs-24k' / 'results' / 'scores_final.json',
           ROOT / 'below-2pct' / 'results' / 'scores_final.json']


def main():
    rows = {}
    for path in SOURCES:
        for n, jobs in base.jobs(path, path.parent.parent.name, RECIPE).items():
            if n in LEVELS:
                assert n not in rows, f'level {n} found twice'
                rows[n] = jobs
    table = []
    for index, n in enumerate(LEVELS):
        jobs = rows.get(n)
        if not jobs or len(jobs) != 11 or len({j['seed'] for j in jobs}) != 11:
            raise SystemExit(f'level {n}: need eleven distinct final seeds, have {len(jobs or [])}')
        table.append({'level': index + 1, 'n': n, 'source': jobs[0]['source'], 'recipe': RECIPE, **base.stats(jobs)})
    for previous, current in zip(table, table[1:]):
        current['ratio_vs_previous'] = previous['pooled_error_pct'] / current['pooled_error_pct']
    out = {'recipe': RECIPE, 'steps': 24000, 'levels': LEVELS, 'cutoffs': table,
           'sources': [str(p) for p in SOURCES]}
    (ROOT / 'analysis' / 'cutoffs-24k.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| Level | Labels N | Cutoff (mean error) | 95% CI | SD (pp) | Band, rounded up |',
             '| ---: | ---: | ---: | --- | ---: | ---: |']
    for r in table:
        lines.append(f"| {r['level']} | {r['n']:,} | {r['pooled_error_pct']:.4f}% | {r['ci95_pct'][0]:.3f}-"
                     f"{r['ci95_pct'][1]:.3f}% | {r['sd_pp']:.3f} | {r['band_pct']:.2f}% |")
    (ROOT / 'analysis' / 'cutoffs-24k.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
