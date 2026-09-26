"""The cutoff table: the fixed state-of-the-art recipe's error at nine label counts.

    /tmp/penv/bin/python cutoffs.py      # after score.py --stage final

Sources, both frozen and scored:
  * results/scores_final.json (this study): N = 100 ... 5,623, eleven final seeds each;
  * ../release-ladder-20260924/results/scores_final.json, candidate ladder-is06-long:
    N = 10,000 (the same eleven seeds with bit-identical inputs, reused by protocol), and
    N = 500, 1,057, 2,236, 4,729, shown as extra points between the grid levels;
  * full-batches/results/scores_final.json and full-batches-3162/results/scores_final.json,
    candidate ladder-is06-long-full: N = 316, 562, 1,778 and 3,162 rerun with every minibatch
    full (their protocol.draft.json files). Where it exists it sets the cutoff, and the frozen
    recipe's value is kept beside it.

Per level: pooled error = 100 * sum(wrong) / sum(queries) from integer counts; SD and
Student-t 95% CI (10 df) over the eleven per-draw errors; "at or below" counts draws
whose error is at most the pooled mean; the board band is the pooled error rounded UP
to the next 0.05 point (the popcorn3 convention, so the recipe that sets a band is not
asked to beat its own mean). Reads no labels.
"""
import json
import math
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EARLIER = ROOT.parent / 'release-ladder-20260924' / 'results' / 'scores_final.json'
FULL = [ROOT / 'full-batches' / 'results' / 'scores_final.json',
        ROOT / 'full-batches-3162' / 'results' / 'scores_final.json']
RECIPE = 'ladder-is06-long'
FULL_RECIPE = 'ladder-is06-long-full'
GRID = [100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000]     # round(100 * 10**(i/4))
BETWEEN = [500, 1057, 2236, 4729]                               # release-ladder-20260924's levels
T10 = 2.2281388519649385


def ceil_to(value, step):
    q = (Decimal(repr(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_CEILING)
    return float(q * Decimal(str(step)))


def stats(rows):
    errs = [100.0 * (r['total'] - r['correct']) / r['total'] for r in rows]
    wrong = sum(r['total'] - r['correct'] for r in rows)
    total = sum(r['total'] for r in rows)
    pooled = 100.0 * wrong / total
    n = len(errs)
    mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1))
    half = T10 * sd / math.sqrt(n)
    walls = [r['metrics'].get('fit_wall_seconds') or 0.0 for r in rows]
    return {'pooled_error_pct': pooled, 'accuracy_pct': 100.0 - pooled, 'sd_pp': sd,
            'ci95_pct': [mean - half, mean + half], 'draws': n, 'wrong': wrong, 'total': total,
            'per_draw_pct': errs, 'min_pct': min(errs), 'max_pct': max(errs),
            'at_or_below': sum(1 for e in errs if e <= pooled),
            'band_pct': ceil_to(pooled, 0.05), 'bp_ceiling': int(ceil_to(pooled, 0.01) * 100 + 0.5),
            'mean_fit_wall_s': sum(walls) / n, 'devices': sorted({str(r['metrics'].get('device_name')) for r in rows})}


def jobs(path, source, recipe=RECIPE):
    out = {}
    if not Path(path).exists():
        return out
    for job in json.loads(Path(path).read_text())['jobs']:
        if job['candidate_id'] == recipe:
            out.setdefault(job['n'], []).append({**job, 'source': source})
    return out


def main():
    here = jobs(ROOT / 'results' / 'scores_final.json', 'this study')
    earlier = jobs(EARLIER, 'release-ladder-20260924')
    full = {}
    for path in FULL:
        for n, rows in jobs(path, path.parent.parent.name, FULL_RECIPE).items():
            assert n not in full, f'level {n} rerun twice'
            full[n] = rows
    table = []
    for index, n in enumerate(GRID):
        frozen = here.get(n) or earlier.get(n)
        rows = full.get(n) or frozen
        for group in (rows, frozen):
            if not group or len(group) != 11 or len({r['seed'] for r in group}) != 11:
                raise SystemExit(f'level {n}: need eleven distinct final seeds, have {len(group or [])}')
        entry = {'level': index + 1, 'n': n, 'source': rows[0]['source'],
                 'recipe': FULL_RECIPE if n in full else RECIPE, **stats(rows)}
        if n in full:
            entry['frozen_recipe'] = stats(frozen)
        table.append(entry)
    for previous, current in zip(table, table[1:]):
        current['ratio_vs_previous'] = previous['pooled_error_pct'] / current['pooled_error_pct']
    between = [{'n': n, 'source': 'release-ladder-20260924', **stats(earlier[n])} for n in BETWEEN]
    out = {'recipe': RECIPE, 'grid': GRID, 'cutoffs': table, 'between': between,
           'sources': [str(ROOT / 'results' / 'scores_final.json'), str(EARLIER)]}
    (ROOT / 'analysis').mkdir(exist_ok=True)
    (ROOT / 'analysis' / 'cutoffs.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| Level | Labels N | Cutoff (mean error) | Accuracy | 95% CI | SD (pp) | Band, rounded up | Recipe | Frozen recipe |',
             '| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |']
    for r in table:
        frozen = f"{r['frozen_recipe']['pooled_error_pct']:.4f}%" if 'frozen_recipe' in r else 'same'
        lines.append(f"| {r['level']} | {r['n']:,} | {r['pooled_error_pct']:.4f}% | {r['accuracy_pct']:.4f}% | "
                     f"{r['ci95_pct'][0]:.3f}-{r['ci95_pct'][1]:.3f}% | {r['sd_pp']:.3f} | {r['band_pct']:.2f}% | "
                     f"{'full batches' if r['recipe'] == FULL_RECIPE else 'frozen'} | {frozen} |")
    lines += ['', 'Between the grid levels (release-ladder-20260924, same recipe and seeds):', '',
              '| Labels N | Mean error | 95% CI |', '| ---: | ---: | --- |']
    for r in between:
        lines.append(f"| {r['n']:,} | {r['pooled_error_pct']:.4f}% | {r['ci95_pct'][0]:.3f}-{r['ci95_pct'][1]:.3f}% |")
    (ROOT / 'analysis' / 'cutoffs.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
