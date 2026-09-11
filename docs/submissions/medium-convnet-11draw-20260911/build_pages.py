"""Publish an explicit allowlist of submission evidence as standalone HTML."""
import argparse
import importlib.util
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    'report_style', REPO / 'mnist/submissions/1nn-v4-20260911/build_pages.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=REPO / 'docs/submissions/medium-convnet-11draw-20260911')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {'.py', '.md', '.txt', '.json', '.svg'}:
            shutil.copy2(source, args.output / source.name)
    # Raw inputs, execution logs and local credentials are never published.
    for folder in ('draws', 'results', 'predictions', 'logits', 'checkpoints', 'benchmark'):
        directory = HERE / folder
        if not directory.exists():
            continue
        for source in sorted(directory.rglob('*')):
            if source.is_file() and source.suffix in {'.json', '.npy', '.npz', '.pt'}:
                target = args.output / source.relative_to(HERE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    for source, target in (('report.md', 'index.html'), ('ambiguities.md', 'ambiguities.html'),
                           ('session.md', 'session.html')):
        page = style.render((HERE / source).read_text(), target)
        page = page.replace('MNIST-small / 1-nearest-neighbor', 'MNIST-medium / 11-dataset ConvNet')
        page = page.replace('Reproducible MNIST-small 1-nearest-neighbor submission attempt, model scores and measured A100 energy.',
                            'A frozen three-ConvNet learner evaluated on eleven independent MNIST-medium dataset draws, with complete-task A100 measurements.')
        page = page.replace('Standalone record. Model scores use declared conventions; acceptance remains subject to review.',
                            'Submission attempt. Accuracy and measured performance are separate from model-translation completeness; acceptance remains subject to review.')
        page = page.replace('</style>', 'main img{display:block;max-width:100%;height:auto}</style>')
        (args.output / target).write_text(page)
    print(args.output)


if __name__ == '__main__':
    main()
