"""Reproduce the fixed learner on the five recorded pilot seeds.

This rerun uses the submitted configuration; it does not establish historical
selection order. Dataset preparation is shared with run.py.
"""
import argparse
import json
from pathlib import Path
import time

from reference import train_predict
from run import _load, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write results to a new JSON file')
    args = parser.parse_args()
    records = []
    for seed in range(20261121, 20261126):
        _, test, x, q, labels_train, _, labels = _load(seed)
        start = time.perf_counter()
        _, _, predictions = train_predict(x, labels_train, q)
        record = {'dataset_seed': seed, 'correct': int((predictions == labels[test]).sum()),
                  'total': len(test), 'elapsed_seconds': time.perf_counter() - start}
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {'draws': records,
              'mean_accuracy': sum(r['correct'] for r in records) / sum(r['total'] for r in records),
              'minimum_accuracy': min(r['correct'] / r['total'] for r in records)}
    if args.output is not None:
        write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
