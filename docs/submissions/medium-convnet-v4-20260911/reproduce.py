"""Prepare an isolated reproduction tree; optionally execute the complete run."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='New empty directory')
    p.add_argument('--raw', type=Path, required=True, help='Canonical gzip IDX directory')
    p.add_argument('--run', action='store_true', help='Execute accuracy, constants, model scoring and A100 measurement; requires uv and configured Modal')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        p.error('Output directory must be empty; retained evidence is never overwritten')
    target = output/'mnist/submissions'/HERE.name
    target.mkdir(parents=True)
    for source in HERE.glob('*.py'):
        shutil.copy2(source, target/source.name)
    for source in (HERE/'ordered_backend').glob('*.py'):
        dest = target/'ordered_backend'/source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
    for name in ('config.json', 'selection.json'):
        shutil.copy2(HERE/name, target/name)
    code = output/'mnist/code'
    code.mkdir(parents=True)
    shutil.copy2(ROOT/'mnist/code/data.py', code/'data.py')
    sources = sorted(json.loads((HERE/'protocol.json').read_text())['source_sha256'])
    commands = [
        [sys.executable, str(target/'evaluation.py'), 'prepare', '--raw', str(args.raw.resolve()),
            '--config', str(target/'config.json'), '--sources', *sources],
        ['uvx', '--with', 'numpy==2.2.6', 'modal==1.5.5', 'run', str(target/'modal_run.py'), '--mode', 'accuracy'],
        [sys.executable, str(target/'evaluation.py'), 'freeze'],
        [sys.executable, str(target/'evaluation.py'), 'evaluate', '--raw', str(args.raw.resolve())],
        ['uvx', 'modal==1.5.5', 'run', str(target/'export_constants_modal.py')],
        [sys.executable, str(target/'ir_score.py'), '--config', str(target/'config.json'),
            '--constants', str(target/'constants/constants.json'), '--output', str(target), '--repeats', '3'],
        ['uvx', '--with', 'numpy==2.2.6', 'modal==1.5.5', 'run', str(target/'modal_run.py'), '--mode', 'benchmark'],
    ]
    for index, command in enumerate(commands):
        print(shlex.join(command), flush=True)
        if index == 0 or args.run:
            subprocess.run(command, cwd=output, check=True)
    print('Reproduction tree:', output)

if __name__ == '__main__':
    main()
