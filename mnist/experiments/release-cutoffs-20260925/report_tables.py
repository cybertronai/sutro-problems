"""Fill README.md's generated tables from the result files; no number is typed by hand.

    /tmp/penv/bin/python report_tables.py

Reads analysis/cutoffs.json, mlp_timing/results/picks.json and the confirmation runs
mlp_timing/results/confirm-L*.json (12,000-step cutoffs); analysis/cutoffs-24k.json,
picks-24k.json and confirm-24k-L*.json (24,000-step cutoffs); the Ladder timing summaries;
and replaces the text between each <!-- NAME START --> and <!-- NAME END --> marker pair
in README.md.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMING = ROOT / 'mlp_timing' / 'results'


def confirm_files(tag=''):
    """Confirmation runs for one set of cutoffs (tag '' or '24k-'), runner-up runs last."""
    return sorted(TIMING.glob(f'confirm-{tag}L*.json')) + sorted(TIMING.glob(f'confirm-{tag}runner-up-L*.json'))


def harness_rows(tag=''):
    """{submission stem: row} from the confirmation runs, the latest file last."""
    rows = {}
    for path in confirm_files(tag):
        for row in json.loads(path.read_text())['rows']:
            rows[Path(row['submission']).stem] = row
    return rows


def level_rows(tag):
    """{(level, submission stem): row}: one set's confirmation runs, keyed by the level they ran at."""
    rows = {}
    for path in confirm_files(tag):
        for row in json.loads(path.read_text())['rows']:
            rows[int(path.stem.rsplit('L', 1)[1]), Path(row['submission']).stem] = row
    return rows


def main_table():
    cut = json.loads((ROOT / 'analysis' / 'cutoffs.json').read_text())['cutoffs']
    picks = {p['level']: p for p in json.loads((TIMING / 'picks.json').read_text())['picks']}
    harness = harness_rows()

    def time_cell(choice):
        if not choice:
            return 'not reached'
        row = harness.get(choice['choice'])
        if row and row['passed'] and row['ranked_ms'] is not None:
            return f"{row['ranked_ms']:,.1f} ms"
        if row and not row['passed']:
            return f"{choice['dev_mean_ms']:,.0f} ms (dev); failed in the harness"
        return f"{choice['dev_mean_ms']:,.0f} ms (dev, not yet confirmed)"

    lines = ['| Level | Labels N | Cutoff: mean error | 95% CI | Band, rounded up | Eager MLP | Graph-captured MLP |',
             '| ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in cut:
        p = picks[r['level']]
        mark = ' *' if r.get('recipe') == 'ladder-is06-long-full' else ''
        lines.append(f"| {r['level']} | {r['n']:,}{mark} | {r['pooled_error_pct']:.3f}% | "
                     f"{r['ci95_pct'][0]:.2f}-{r['ci95_pct'][1]:.2f}% | {r['band_pct']:.2f}% | "
                     f"{time_cell(p.get('eager'))} | {time_cell(p.get('graphed'))} |")
    return '\n'.join(lines)


def mlp_table(suffix='', tag=''):
    path = TIMING / f'picks{suffix}.json'
    if not path.exists():
        return '(not picked yet)'
    picks = json.loads(path.read_text())['picks']
    harness = level_rows(tag)
    lines = ['| Labels N | Band | Family | Configuration | Dev error | Dev time | Harness: verdict, MNIST accuracy, ranked time |',
             '| ---: | ---: | --- | --- | ---: | ---: | --- |']
    for p in picks:
        for family in ('eager', 'graphed'):
            choice = p.get(family)
            if not choice:
                continue
            row = harness.get((p['level'], choice['choice']))
            verdict = (f"{'pass' if row['passed'] else 'FAIL'}, {row['mnist']}, "
                       f"{row['ranked_ms']:,.1f} ms" if row and row['ranked_ms'] is not None else
                       (f"{'pass' if row['passed'] else 'FAIL'}, {row['mnist']}" if row else 'not run'))
            lines.append(f"| {p['n']:,} | {p['band_bp'] / 100:.2f}% | {family} | `{choice['choice']}` | "
                         f"{choice['dev_error_pct']:.2f}% | {choice['dev_mean_ms']:,.0f} ms | {verdict} |")
    return '\n'.join(lines)


def frozen_vs_full():
    cut = json.loads((ROOT / 'analysis' / 'cutoffs.json').read_text())['cutoffs']
    lines = ['| Labels N | Last minibatch, frozen recipe | Frozen recipe | Full batches | Change |',
             '| ---: | ---: | ---: | ---: | ---: |']
    for r in cut:
        if 'frozen_recipe' in r:
            last = r['n'] % 250
            old, new = r['frozen_recipe']['pooled_error_pct'], r['pooled_error_pct']
            lines.append(f"| {r['n']:,} | {last} rows | {old:.3f}% | {new:.3f}% | {new - old:+.3f} pp |")
    return '\n'.join(lines)


def five_table():
    """The five thresholds (every other grid level) with Ladder and MLP times to reach each."""
    cut = {r['n']: r for r in json.loads((ROOT / 'analysis' / 'cutoffs.json').read_text())['cutoffs']}
    picks = {p['n']: p for p in json.loads((TIMING / 'picks.json').read_text())['picks']}
    summary = json.loads((ROOT / 'ladder_timing' / 'results' / 'summary.json').read_text())
    top = max(summary['curve'], key=lambda p: p['steps'])
    harness = harness_rows()

    def secs(ms):
        return f'{ms:,.0f} ms' if ms < 1000 else f'{ms / 1000:,.1f} s'

    def mlp(n, family):
        choice = picks[n].get(family)
        row = harness.get(choice['choice']) if choice else None
        return secs(row['ranked_ms']) if row and row['passed'] else 'not reached'

    lines = ['| Level | Labels N | Cutoff | Band | Ladder, 10,000 labels | MLP, eager | MLP, graph-captured |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for level, n in enumerate([100, 316, 1000, 3162, 10000], 1):
        r = cut[n]
        reach = summary['reach'][str(n)]
        if n == 10000:
            ladder = f"{secs(top['ms'])} at {top['steps']:,} steps (sets it)"
        elif reach is None:
            ladder = 'not reached'
        elif reach['kind'] == 'at most':
            ladder = f"at most {secs(reach['ms'])} ({reach['steps']:,} steps)"
        else:
            ladder = f"about {secs(reach['ms'])} (about {round(reach['steps'], -2):,.0f} steps)"
        lines.append(f"| {level} | {n:,} | {r['pooled_error_pct']:.3f}% | {r['band_pct']:.2f}% | {ladder} | "
                     f"{mlp(n, 'eager')} | {mlp(n, 'graphed')} |")
    return '\n'.join(lines)


def below_table():
    """Every pre-registered below-2% candidate, from below-2pct/results/below2.json."""
    path = ROOT / 'below-2pct' / 'results' / 'below2.json'
    if not path.exists():
        return '(not scored yet)'
    rows = json.loads(path.read_text())
    lines = ['| Candidate, N = 10,000 labels | Pooled error | 95% CI | SD (pp) | At or below 2.00% |',
             '| --- | ---: | --- | ---: | --- |']
    for name, r in rows.items():
        lines.append(f"| {name}{' (primary)' if r['primary'] else ''} | {r['pooled_error_pct']:.3f}% | "
                     f"{r['ci95_pct'][0]:.3f}-{r['ci95_pct'][1]:.3f}% | {r['sd_pp']:.3f} | "
                     f"{'yes' if r['at_or_below_2pct'] else 'no'} |")
    return '\n'.join(lines)


def long_table():
    """The five cutoffs from the 24,000-step recipe (cutoffs24k.py), with Ladder and MLP times.

    Ladder: the 12,000-step recipe's timing sweep on all 10,000 labels (summary-24k.json).
    Level 5 is the 24,000-step run itself; its time is an estimate, twice the measured
    12,000-step time on the median host (per-step cost is flat from 4,000 steps on).
    """
    path = ROOT / 'analysis' / 'cutoffs-24k.json'
    if not path.exists():
        return '(not scored yet)'
    cut = json.loads(path.read_text())['cutoffs']
    picks = {p['n']: p for p in json.loads((TIMING / 'picks-24k.json').read_text())['picks']}
    summary = json.loads((ROOT / 'ladder_timing' / 'results' / 'summary-24k.json').read_text())
    top = max(summary['curve'], key=lambda p: p['steps'])
    harness = level_rows('24k-')

    def secs(ms):
        return f'{ms:,.0f} ms' if ms < 1000 else f'{ms / 1000:,.1f} s'

    def mlp(level, n, family):
        choice = picks[n].get(family)
        row = harness.get((level, choice['choice'])) if choice else None
        if choice and not (row and row['passed']):
            return f"{choice['dev_mean_ms']:,.0f} ms (dev); " + ('failed in the harness' if row else 'not confirmed')
        return secs(row['ranked_ms']) if row else 'not reached'

    lines = ['| Level | Labels N | Cutoff | 95% CI | Band | Ladder, 10,000 labels | MLP, eager | MLP, graph-captured |',
             '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in cut:
        level, n = r['level'], r['n']
        reach = summary['reach'][str(n)]
        if n == 10000:
            ladder = f"about {secs(2 * top['ms'])} at 24,000 steps (sets it; estimate)"
        elif reach is None:
            ladder = f"not within {top['steps']:,} steps"
        elif reach['kind'] == 'at most':
            ladder = f"at most {secs(reach['ms'])} ({reach['steps']:,} steps)"
        else:
            ladder = f"about {secs(reach['ms'])} (about {round(reach['steps'], -2):,.0f} steps)"
        lines.append(f"| {level} | {n:,} | {r['pooled_error_pct']:.3f}% | {r['ci95_pct'][0]:.2f}-{r['ci95_pct'][1]:.2f}% | "
                     f"{r['band_pct']:.2f}% | {ladder} | {mlp(level, n, 'eager')} | {mlp(level, n, 'graphed')} |")
    return '\n'.join(lines)


def main():
    readme = ROOT / 'README.md'
    text = readme.read_text()
    for name, build in (('MAIN_TABLE', main_table), ('MLP_TABLE', mlp_table), ('FIX_TABLE', frozen_vs_full),
                        ('FIVE_TABLE', five_table), ('BELOW_TABLE', below_table), ('LONG_TABLE', long_table),
                        ('LONG_MLP_TABLE', lambda: mlp_table('-24k', '24k-'))):
        pattern = re.compile(rf'(<!-- {name} START -->\n)(?:.*?\n)?(<!-- {name} END -->)', re.S)
        if not pattern.search(text):
            raise SystemExit(f'README.md has no {name} markers')
        text = pattern.sub(lambda m: m.group(1) + build() + '\n' + m.group(2), text)
    readme.write_text(text)
    print(main_table())


if __name__ == '__main__':
    main()
