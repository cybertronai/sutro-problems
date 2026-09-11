"""Plot retained validation evidence without retraining or reading test data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(path.read_text()) for path in sorted((HERE / 'results').glob('search-*.json'))]
    if not rows:
        raise ValueError('No completed initial search runs')
    errors = [1200-row['best']['validation']['correct'] for row in rows]
    colors = ['#bd7d16' if row['config']['activation'] == 'relu' else
              '#138d85' if row['config']['augmentation'] != 'none' else '#41688b' for row in rows]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(11, 5.2), layout='constrained')
        bars = ax.bar([row['config']['id'] for row in rows], errors, color=colors, width=.7)
        ax.bar_label(bars, labels=[str(x) for x in errors], padding=4)
        ax.axhline(24, color='#b64432', linestyle='--', linewidth=1.4, label='98%: at most 24 validation errors')
        ax.set_ylim(0, max(max(errors)+6, 32))
        ax.set_ylabel('Errors out of 1,200 validation examples')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title('ConvNet validation errors at the selected checkpoint', loc='left', pad=16, weight='bold')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e5e9eb')
        ax.legend(loc='upper right', frameon=False)
        fig.supxlabel('Initial seed 11 · Each checkpoint selected on this validation set', fontsize=10)
        fig.savefig(HERE / 'validation-search.svg')
        svg = HERE / 'validation-search.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        fig.savefig(HERE / 'validation-search.png', dpi=150)
    print('Wrote validation-search.svg and validation-search.png')


if __name__ == '__main__':
    main()
