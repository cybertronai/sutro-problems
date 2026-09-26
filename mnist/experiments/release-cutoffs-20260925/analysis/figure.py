"""analysis/figures/cutoffs.png: the cutoffs (top) and the MLP time to reach each (bottom).

    /tmp/penv/bin/python analysis/figure.py     # after cutoffs.py and mlp_timing/analyze.py

Two panels on one shared x axis (labels N, log scale) instead of two y scales on one
panel. Colours are the reference palette's first three categorical slots, which pass
the all-pairs colour-vision checks as a set; every series is also labelled directly.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SOTA, EAGER, GRAPHED = '#2a78d6', '#eb6834', '#1baf7a'


def main():
    cut = json.loads((ROOT / 'analysis' / 'cutoffs.json').read_text())
    picks = json.loads((ROOT / 'mlp_timing' / 'results' / 'picks.json').read_text())['picks']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={'height_ratios': [1.15, 1], 'hspace': 0.28})
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.set_xscale('log')
        ax.grid(True, which='major', color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    levels = cut['cutoffs']
    xs = [r['n'] for r in levels]
    ys = [r['pooled_error_pct'] for r in levels]
    lo = [r['pooled_error_pct'] - r['ci95_pct'][0] for r in levels]
    hi = [r['ci95_pct'][1] - r['pooled_error_pct'] for r in levels]
    top.errorbar(xs, ys, yerr=[lo, hi], fmt='o-', color=SOTA, ecolor=SOTA, elinewidth=1.2, capsize=3,
                 linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for r in levels:
        top.annotate(f"{r['pooled_error_pct']:.2f}%", (r['n'], r['pooled_error_pct']), textcoords='offset points',
                     xytext=(7, 6), ha='left', fontsize=8.5, color=INK)
    top.set_yscale('log')
    top.set_yticks([2, 3, 5, 10, 20, 30])
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}%'))
    top.yaxis.set_minor_formatter(NullFormatter())
    top.set_ylim(1.7, 40)
    top.set_ylabel('Mean query error (the cutoff)')
    top.set_title('The state-of-the-art recipe\'s error at each label count is that level\'s cutoff',
                  loc='left', fontsize=11.5, color=INK, pad=10)
    top.text(0.99, 0.95, 'bars: 95% CI over eleven draws\n316, 562 and 1,778 labels: full minibatches',
             transform=top.transAxes, ha='right', va='top', fontsize=8.5, color=MUTED)

    for family, colour, label, offset in (('eager', EAGER, 'eager MLP', (-8, 12)),
                                          ('graphed', GRAPHED, 'graph-captured MLP', (10, -14))):
        points = [(p['n'], p[family]['dev_mean_ms'] / 1000.0) for p in picks if p.get(family)]
        if points:
            bottom.plot(*zip(*points), 'o-', color=colour, linewidth=2, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            x_at, y_at = points[2] if family == 'eager' else points[3]
            bottom.annotate(label, (x_at, y_at), textcoords='offset points', xytext=offset,
                            ha='right' if family == 'eager' else 'left', va='center', fontsize=9, color=INK)
    unreachable = [p['n'] for p in picks if not p.get('eager') and not p.get('graphed')]
    bottom.set_yscale('log')
    bottom.set_ylim(0.004, 150)
    bottom.axhline(60, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    bottom.text(101, 75, '60 s per-call limit', fontsize=8.5, color=MUTED, va='bottom')
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g} s'))
    bottom.set_ylabel('Time per call on an A100\n(10,000 labels, dev draws)')
    bottom.set_title('Time for an MLP trained on all 10,000 labels to reach each cutoff', loc='left',
                     fontsize=11.5, color=INK, pad=10)
    if unreachable:
        for n in unreachable:
            bottom.plot([n], [60], marker='x', color=INK, markersize=8, markeredgewidth=1.8, zorder=4)
        bottom.annotate('no MLP reaches these\nwithin 60 s (best: 2.75%)', (unreachable[0], 60),
                        textcoords='offset points', xytext=(-6, -26), ha='left', va='top', fontsize=8.5,
                        color=MUTED)
    bottom.set_xlabel('Labelled training examples N (the state-of-the-art recipe\'s budget)')
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
    bottom.set_xticks(xs)
    bottom.set_xlim(75, 13500)
    bottom.xaxis.set_minor_formatter(NullFormatter())
    plt.setp(bottom.get_xticklabels(), rotation=0, fontsize=8.5)
    out = ROOT / 'analysis' / 'figures' / 'cutoffs.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
