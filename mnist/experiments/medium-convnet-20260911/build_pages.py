"""Render the accuracy study, separate issues and visible session for GitHub Pages."""
from pathlib import Path
import argparse
import importlib.util
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('report_style', REPO / 'mnist/submissions/1nn-v4-20260911/build_pages.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'docs/submissions/medium-convnet-20260911')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Publish only explicit scientific artifacts. Raw execution logs and inputs stay local.
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {'.py', '.md', '.txt', '.json', '.svg', '.png'}:
            shutil.copy2(source, args.output / source.name)
    for folder in ('results', 'final', 'validation_logits'):
        source_dir = HERE / folder
        if source_dir.exists():
            for source in sorted(source_dir.rglob('*')):
                if source.is_file() and source.suffix in {'.json', '.npy', '.npz', '.pt', '.svg', '.png'}:
                    target = args.output / source.relative_to(HERE)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    for name, target in [('report.md', 'index.html'), ('ambiguities.md', 'ambiguities.html'), ('session.md', 'session.html')]:
        page = style.render((HERE / name).read_text(), target)
        page = page.replace('MNIST-small / 1-nearest-neighbor', 'MNIST-medium / ConvNet accuracy study')
        page = page.replace('Submission attempt · September 2026', 'Accuracy study · September 2026')
        page = page.replace('Reproducible MNIST-small 1-nearest-neighbor submission attempt, model scores and measured A100 energy.',
                            'ConvNet architecture search on the canonical MNIST-medium dataset, with training-only selection and a separate test evaluation.')
        page = page.replace('Standalone record. Model scores use declared conventions; acceptance remains subject to review.',
                            'Accuracy feasibility study. Cost translation and complete-task performance measurements are separate stages.')
        page = page.replace('</style>', 'main img{display:block;max-width:100%;height:auto}</style>')
        (args.output / target).write_text(page)
    print(args.output)


if __name__ == '__main__':
    main()
