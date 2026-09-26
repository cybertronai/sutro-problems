"""Confirm each chosen MLP in the popcorn3 harness, with the band set to its cutoff.

    /tmp/penv/bin/python confirm.py                    # every pick in results/picks.json
    /tmp/penv/bin/python confirm.py --runner-up        # the next-fastest choice where a pick failed

For each level with a pick, one tools/score_submissions.py run on Modal A100-80GB
containers (leaderboard mode: compile step, test, benchmark, leaderboard; 11 MNIST +
4 hold-out calls) with --case error_bp=<the cutoff rounded up to 1 basis point>, so
the harness itself decides whether the MLP reaches the cutoff. At most three runs at
a time (two files each). Summary in results/confirm.md.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
POPCORN = HERE.parents[3] / 'popcorn3'
PYTHON = '/tmp/penv/bin/python'


def run(level, files, band_bp, tag):
    out = HERE / 'results' / f'confirm-{tag}L{level}.json'
    log = out.with_suffix('.log')
    command = [PYTHON, 'tools/score_submissions.py', *map(str, files), '--board', 'mnist-medium-3p40pct',
               '--case', f'error_bp={band_bp}', '--where', 'modal', '--out', str(out)]
    with log.open('w') as stream:
        subprocess.run(command, cwd=POPCORN, stdout=stream, stderr=subprocess.STDOUT)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner-up', action='store_true')
    parser.add_argument('--levels', help='comma-separated level numbers (default: every level with a pick)')
    parser.add_argument('--picks', default=str(HERE / 'results' / 'picks.json'))
    parser.add_argument('--tag', default='', help="prefix for the output files, e.g. 24k-")
    args = parser.parse_args()
    picks = json.loads(Path(args.picks).read_text())['picks']
    if args.levels:
        wanted = {int(v) for v in args.levels.split(',')}
        picks = [p for p in picks if p['level'] in wanted]
    tasks = []
    for entry in picks:
        names = []
        for family in ('eager', 'graphed'):
            choice = entry.get(family)
            if not choice:
                continue
            if args.runner_up:
                previous = HERE / 'results' / f"confirm-{args.tag}L{entry['level']}.json"
                rows = json.loads(previous.read_text())['rows'] if previous.exists() else []
                failed = any(Path(r['submission']).stem == choice['choice'] and not r['passed'] for r in rows)
                if failed and choice.get('runner_up'):
                    names.append(choice['runner_up'])
            else:
                names.append(choice['choice'])
        if names:
            files = []
            for name in names:
                path = HERE / 'submissions' / f'{name}.py'
                if not path.exists():
                    sys.path.insert(0, str(HERE))
                    import mlp_family
                    path.write_text(mlp_family.source(*mlp_family.parse(name)))
                files.append(path)
            tasks.append((entry['level'], files, entry['band_bp']))
    tag = args.tag + ('runner-up-' if args.runner_up else '')
    print(f'{len(tasks)} harness run(s): ' + '; '.join(f"L{l}: {', '.join(f.stem for f in fs)} at {b} bp"
                                                       for l, fs, b in tasks), flush=True)
    with ThreadPoolExecutor(3) as pool:
        outs = list(pool.map(lambda t: run(*t, tag), tasks))
    lines = ['| Level | Band | MLP | Verdict | Harness ranked time | MNIST | Host GPU |', '| ---: | ---: | --- | --- | ---: | ---: | --- |']
    for (level, files, band_bp), out in zip(tasks, outs):
        if not out.exists():
            lines.append(f'| {level} | {band_bp / 100:.2f}% | (run failed, see {out.with_suffix(".log").name}) | | | | |')
            continue
        for row in json.loads(out.read_text())['rows']:
            ms = f"{row['ranked_ms']:,.1f} ms" if row['ranked_ms'] is not None else '—'
            lines.append(f"| {level} | {band_bp / 100:.2f}% | `{Path(row['submission']).stem}` | "
                         f"{'pass' if row['passed'] else 'FAIL'} | {ms} | {row['mnist'] or '—'} | {row['device'] or '—'} |")
    suffix = f"-L{args.levels.replace(',', '-')}" if args.levels else ''
    (HERE / 'results' / f'confirm-{tag}summary{suffix}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
