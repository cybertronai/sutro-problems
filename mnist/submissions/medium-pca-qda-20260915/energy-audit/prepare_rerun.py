"""Prepare a fresh, self-contained rerun directory without overwriting evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
INPUT_ARTIFACT = 'yaroslavvb/sutro-mnist-tiers/pca-qda-frozen-inputs-rha1uwss:v0'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='A new directory outside the evidence directory')
    parser.add_argument('--payloads', type=Path, help='Existing exact payloads; otherwise download the pinned W&B input artifact')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh directory to preserve earlier evidence.')
    plan = json.loads((HERE / 'plan-wandb.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if sha256(HERE.parent / name) != expected:
            raise ValueError(f'Submitted learner source changed: {name}')
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in plan['source_sha256']:
        shutil.copy2(HERE.parent / name, output / 'source' / name)
    for name in ('run_wandb.py', 'verify_wandb.py', 'modal_same_hardware.py',
                 'independent_measure.py', 'analyze.py', 'requirements-controller.txt',
                 'published-gpu-results.json'):
        shutil.copy2(HERE / name, output / name)
    payloads = output / 'payloads'
    if args.payloads:
        shutil.copytree(args.payloads.expanduser().resolve(), payloads)
    else:
        import wandb
        wandb.Api().artifact(INPUT_ARTIFACT).download(root=str(payloads))
    expected_manifest = json.loads((HERE / 'payload-manifest.json').read_text())
    if sha256(payloads / 'manifest.json') != sha256(HERE / 'payload-manifest.json'):
        raise ValueError('Payload manifest differs from the measured input artifact')
    for row in expected_manifest['draws']:
        if sha256(payloads / row['path']) != row['sha256']:
            raise ValueError(f"Payload checksum mismatch: {row['path']}")
    print(f'Prepared {output}; verified learner source and all 11 payload checksums.')
    print('Run: uv run --python 3.11 --with-requirements requirements-controller.txt python run_wandb.py')


if __name__ == '__main__':
    main()
