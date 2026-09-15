"""Run frozen evidence against its exact dataset helper, not a moving upstream file.

The measured learner, protocol and predictions remain byte-for-byte unchanged.
Only dependency resolution is pinned here because upstream added a medium profile
after this run. Existing protocol checks still enforce the original helper hash.
"""
import argparse
import hashlib
from pathlib import Path
import runpy

import frozen_data
import run

HERE = Path(__file__).resolve().parent
GENERATOR_SHA256 = '72e203322357befa1aeeae89b7d20f760ae7bf841008625fdb8ce3f2231563b3'
assert hashlib.sha256(Path(frozen_data.__file__).read_bytes()).hexdigest() == GENERATOR_SHA256
run.generator = frozen_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','fit','evaluate','verify','audit','report','gpu-reference'))
    parser.add_argument('--raw', type=Path, default=HERE.parents[1] / 'data/raw')
    args = parser.parse_args()
    if args.phase == 'prepare':
        run.prepare(args.raw)
    elif args.phase == 'fit':
        run.fit()
    elif args.phase in ('evaluate','verify'):
        run.evaluate(args.raw, verify=args.phase == 'verify')
    else:
        script = {'audit':'audit.py','report':'build_report.py','gpu-reference':'prepare_gpu.py'}[args.phase]
        runpy.run_path(str(HERE / script), run_name='__main__')


if __name__ == '__main__':
    main()
