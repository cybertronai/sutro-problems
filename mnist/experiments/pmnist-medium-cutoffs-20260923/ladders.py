"""Assemble the two five-level ladders from results/scores_final.json.

Ladder A (dense, permutation-invariant): per level, the candidate with the lowest
pooled error among the dense candidates measured at that level on all eleven
final seeds (frozen rule: kr-arccos1-d3 at every level, ladder-retuned-tq at
N=1000). Ladder B (unrestricted): topo-cnn09-x3 at every level. Pooled error is
100*(sum(total)-sum(correct))/sum(total) from integer counts; CI is Student-t
with 10 df on the per-draw errors; pass fraction counts draws at or below the
pooled mean. Rounding uses decimal half-up. Never reads labels.
"""
import json, math, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent
LEVELS = [1000, 1778, 3162, 5623, 10000]
T10 = 2.2281388519649385  # t_{0.975, 10}
def rnd(v, step):
    return float((Decimal(str(v)) / Decimal(str(step))).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(step)))
def stats(rows):
    errs = [100.0 * (r['total'] - r['correct']) / r['total'] for r in rows]
    pooled = 100.0 * (sum(r['total'] for r in rows) - sum(r['correct'] for r in rows)) / sum(r['total'] for r in rows)
    n = len(errs); mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1)) if n > 1 else float('nan')
    half = T10 * sd / math.sqrt(n) if n == 11 else float('nan')
    return {'pooled_error_pct': pooled, 'mean_of_draws_pct': mean, 'sd_pp': sd, 'ci95': [mean - half, mean + half],
            'min': min(errs), 'max': max(errs), 'draws': n, 'pass_fraction': sum(1 for e in errs if e <= pooled) / n,
            'correct': sum(r['correct'] for r in rows), 'total': sum(r['total'] for r in rows), 'per_draw': errs,
            'mean_fit_wall_s': sum(r['metrics'].get('fit_wall_seconds') or 0 for r in rows) / n}
def main(scores_path):
    scores = json.loads(Path(scores_path).read_text())
    by = defaultdict(list)
    for j in scores['jobs']:
        by[(j['candidate_id'], j['n'])].append(j)
    dense = ['kr-arccos1-d3', 'ladder-retuned-tq']
    out = {'source': str(scores_path), 'levels': LEVELS, 'A_dense': [], 'B_unrestricted': [], 'candidates': {}}
    for (cid, n), rows in sorted(by.items()):
        if len(rows) == 11:
            out['candidates'][f'{cid}@{n}'] = stats(rows)
    for i, n in enumerate(LEVELS):
        options = [(stats(by[(c, n)])['pooled_error_pct'], c) for c in dense if len(by.get((c, n), [])) == 11]
        err, cid = min(options)
        s = stats(by[(cid, n)])
        out['A_dense'].append({'level': i + 1, 'n': n, 'candidate': cid, **s,
                               'also_measured': {c: stats(by[(c, n)])['pooled_error_pct'] for _, c in options if c != cid},
                               'rounded_0p05': rnd(s['pooled_error_pct'], 0.05), 'rounded_0p1': rnd(s['pooled_error_pct'], 0.1)})
        t = stats(by[('topo-cnn09-x3', n)])
        out['B_unrestricted'].append({'level': i + 1, 'n': n, 'candidate': 'topo-cnn09-x3', **t,
                                      'rounded_0p05': rnd(t['pooled_error_pct'], 0.05), 'rounded_0p1': rnd(t['pooled_error_pct'], 0.1)})
    for key in ('A_dense', 'B_unrestricted'):
        lad = out[key]
        for k in range(1, len(lad)):
            lad[k]['step_ratio_vs_previous'] = lad[k - 1]['pooled_error_pct'] / lad[k]['pooled_error_pct']
        lad[0]['step_ratio_vs_previous'] = None
    Path(ROOT / 'analysis').mkdir(exist_ok=True)
    (ROOT / 'analysis/ladders.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = []
    for key, title in (('A_dense', 'Ladder A: dense permutation-invariant (no topology recovery)'),
                       ('B_unrestricted', 'Ladder B: unrestricted permutation-invariant (topology recovery + CNN)')):
        lines += [f'## {title}', '', '| Level | Examples | Recipe | Expected error | Accuracy | 95% CI | SD (pp) | Pass | Rounded 0.05 / 0.1 | Error ratio vs previous |',
                  '| ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | ---: |']
        for r in out[key]:
            ratio = '' if r['step_ratio_vs_previous'] is None else f"{r['step_ratio_vs_previous']:.3f}x"
            lines.append(f"| {r['level']} | {r['n']:,} | `{r['candidate']}` | {r['pooled_error_pct']:.4f}% | {100 - r['pooled_error_pct']:.4f}% | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]% | {r['sd_pp']:.3f} | {int(round(r['pass_fraction'] * 11))}/11 | {r['rounded_0p05']:.2f}% / {r['rounded_0p1']:.1f}% | {ratio} |")
        lines.append('')
    (ROOT / 'analysis/ladders.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'results/scores_final.json')
