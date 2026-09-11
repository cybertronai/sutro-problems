"""Render the medium report, separate issues, and visible session snapshot."""
import argparse
import importlib.util
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('baseline_pages',HERE.parent/'1nn-v4-20260911/build_pages.py')
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/submissions/medium-affine-20260911'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {'.py','.md','.txt','.json','.npy','.npz','.ptx'}:
            shutil.copy2(source,args.output/source.name)
    for name,target in [('report.md','index.html'),('ambiguities.md','ambiguities.html'),('session.md','session.html')]:
        page=baseline.render((HERE/name).read_text(),target)
        page=page.replace('MNIST-small / 1-nearest-neighbor','MNIST-medium / 98% target attempt')
        page=page.replace('Reproducible MNIST-small 1-nearest-neighbor submission attempt, model scores and measured A100 energy.',
                          'MNIST-medium: train-only validation, a frozen MLP, compact exact model scores, and complete-task A100 measurements.')
        (args.output/target).write_text(page)
    print(args.output)


if __name__=='__main__':
    main()
