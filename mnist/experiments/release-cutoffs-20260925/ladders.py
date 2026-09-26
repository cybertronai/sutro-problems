"""Five accuracy thresholds from the frozen final scores (results/scores_final.json plus
results/scores_final-ensemble.json for the declared ensemble finalist).

For every finalist measured on all eleven final seeds at a level, the pooled error is
100 * sum(wrong) / sum(queries) from integer counts; the SD and the Student-t 95% CI
(10 df) describe the spread over the eleven draws; "pass" counts the draws at or below
the pooled mean. The threshold at a level is the finalist with the lowest pooled error
there (the frozen rule's per-level best); the SOTA recipe is the finalist with the
lowest mean of its five per-level errors. Rounding is decimal half-up. Reads no labels.
"""
import json, math, sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEVELS = [500, 1057, 2236, 4729, 10000]
T10 = 2.2281388519649385


def rnd(value, step):
    q = (Decimal(repr(value)) / Decimal(str(step))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return float(q * Decimal(str(step)))


def stats(rows):
    errs = [100.0 * (r['total'] - r['correct']) / r['total'] for r in rows]
    wrong = sum(r['total'] - r['correct'] for r in rows); total = sum(r['total'] for r in rows)
    pooled = 100.0 * wrong / total
    n = len(errs); mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1)) if n > 1 else float('nan')
    half = T10 * sd / math.sqrt(n) if n == 11 else float('nan')
    walls = [r['metrics'].get('fit_wall_seconds') or 0.0 for r in rows]
    return {'pooled_error_pct': pooled, 'accuracy_pct': 100.0 - pooled, 'sd_pp': sd,
            'ci95_pct': [mean - half, mean + half], 'draws': n, 'wrong': wrong, 'total': total,
            'min_pct': min(errs), 'max_pct': max(errs), 'per_draw_pct': errs,
            'pass_draws': sum(1 for e in errs if e <= pooled), 'mean_fit_wall_s': sum(walls) / n,
            'max_fit_wall_s': max(walls)}


def main(paths=(ROOT / 'results/scores_final.json', ROOT / 'results/scores_final-ensemble.json')):
    by = defaultdict(list)
    used = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        used.append(str(path))
        for job in json.loads(path.read_text())['jobs']:
            by[(job['candidate_id'], job['n'])].append(job)
    for key, rows in by.items():
        seeds = [r['seed'] for r in rows]
        assert len(seeds) == len(set(seeds)), f'duplicate seeds for {key}'
    cands = sorted({c for c, _ in by})
    table = {c: {n: stats(by[(c, n)]) for n in LEVELS if len(by.get((c, n), [])) == 11} for c in cands}
    complete = [c for c in cands if len(table[c]) == len(LEVELS)]
    sota = min(complete, key=lambda c: sum(table[c][n]['pooled_error_pct'] for n in LEVELS)) if complete else None
    ladder = []
    for i, n in enumerate(LEVELS):
        options = [(table[c][n]['pooled_error_pct'], c) for c in cands if n in table[c]]
        if not options:
            continue
        err, best = min(options)
        s = table[best][n]
        ladder.append({'level': i + 1, 'n': n, 'recipe': best, **s,
                       'threshold_error_rounded_0p05': rnd(s['pooled_error_pct'], 0.05),
                       'threshold_error_rounded_0p1': rnd(s['pooled_error_pct'], 0.1),
                       'others': {c: table[c][n]['pooled_error_pct'] for _, c in options if c != best}})
    for k in range(1, len(ladder)):
        ladder[k]['error_ratio_vs_previous'] = ladder[k - 1]['pooled_error_pct'] / ladder[k]['pooled_error_pct']
    out = {'sources': used, 'levels': LEVELS, 'sota_recipe': sota, 'ladder': ladder, 'by_recipe': table}
    (ROOT / 'analysis').mkdir(exist_ok=True)
    (ROOT / 'analysis/ladder.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| Level | Examples | Recipe | Expected error | Accuracy | 95% CI | SD (pp) | Pass | Rounded 0.05 / 0.1 |',
             '| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |']
    for r in ladder:
        lines.append(f"| {r['level']} | {r['n']:,} | `{r['recipe']}` | {r['pooled_error_pct']:.4f}% | {r['accuracy_pct']:.4f}% | "
                     f"[{r['ci95_pct'][0]:.3f}, {r['ci95_pct'][1]:.3f}]% | {r['sd_pp']:.3f} | {r['pass_draws']}/11 | "
                     f"{r['threshold_error_rounded_0p05']:.2f}% / {r['threshold_error_rounded_0p1']:.1f}% |")
    lines += ['', f'SOTA recipe (lowest mean of per-level errors): `{sota}`', '',
              '| Recipe | ' + ' | '.join(f'{n:,}' for n in LEVELS) + ' |', '| --- |' + ' ---: |' * len(LEVELS)]
    for c in cands:
        lines.append(f'| `{c}` | ' + ' | '.join(f"{table[c][n]['pooled_error_pct']:.3f}%" if n in table[c] else '—' for n in LEVELS) + ' |')
    (ROOT / 'analysis/ladder.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:] or (ROOT / 'results/scores_final.json', ROOT / 'results/scores_final-ensemble.json'))
