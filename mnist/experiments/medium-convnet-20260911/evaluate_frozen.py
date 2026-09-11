"""Freeze all ConvNet predictions, then evaluate them in a separate invocation.

Only the 'evaluate' phase opens test labels. The learner never imports this file.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(path):
    x = np.load(path, allow_pickle=False)
    if x.shape != (6000,) or x.dtype.kind not in 'iu' or np.any((x < 0) | (x > 9)):
        raise ValueError(f'Invalid predictions: {path}')
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--predictions', action='append', default=[], help='NAME=PATH; freeze phase only')
    parser.add_argument('--selected-name', help='Name fixed by validation selection')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=REPO / 'mnist/data/medium.npz')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source_sha = sha(Path(__file__))
    selection = json.loads(args.selection.read_text())
    if args.phase == 'freeze':
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        if not args.predictions or args.selected_name is None:
            parser.error('freeze requires --predictions and --selected-name')
        if args.selected_name != selection['selected_name']:
            raise ValueError('Selected predictor must match the frozen validation choice')
        entries = []
        for item in args.predictions:
            name, filename = item.split('=', 1)
            path = Path(filename).resolve()
            load_predictions(path)
            import os
            entries.append({'name': name, 'path': os.path.relpath(path, args.manifest.parent.resolve()),
                            'sha256': sha(path)})
        names = [x['name'] for x in entries]
        if len(set(names)) != len(names) or args.selected_name not in names:
            raise ValueError('Duplicate names or missing selected prediction')
        record = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                  'selection_sha256': sha(args.selection), 'evaluator_sha256': source_sha,
                  'selected_name': args.selected_name, 'predictions': entries,
                  'test_labels_opened': False}
        args.manifest.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
        return
    if args.output is None:
        parser.error('evaluate requires --output')
    if args.output.exists():
        raise FileExistsError(args.output)
    record = json.loads(args.manifest.read_text())
    if record['selected_name'] != selection['selected_name']:
        raise ValueError('Selected predictor no longer matches the frozen validation choice')
    if record['selection_sha256'] != sha(args.selection) or record['evaluator_sha256'] != source_sha:
        raise ValueError('Frozen selection or evaluator changed')
    preds = {}
    for entry in record['predictions']:
        path = args.manifest.parent / entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError(f'Frozen prediction changed: {entry["name"]}')
        preds[entry['name']] = load_predictions(path)
    # All predictions have been verified before accessing the label member.
    with np.load(args.data, allow_pickle=False) as archive:
        labels = archive['test_labels']
    canonical = json.loads((REPO / 'mnist/doc/dataset_manifest.json').read_text())['tiers']['medium']['arrays']['test_labels']
    label_sha = hashlib.sha256(np.ascontiguousarray(labels.astype('<i8')).tobytes()).hexdigest()
    if list(labels.shape) != canonical['shape'] or str(labels.dtype) != canonical['dtype'] or label_sha != canonical['sha256_c_order_little_endian']:
        raise ValueError('Noncanonical test labels')
    repo_target = json.loads((REPO / 'mnist/doc/accuracy_targets.json').read_text())['medium']
    results = []
    for name, pred in preds.items():
        correct = int(np.count_nonzero(pred == labels))
        targets = {}
        for label, threshold in [('requested', '98'), ('repository', repo_target)]:
            required = int((Decimal(str(threshold)) * len(labels) / 100).to_integral_value(rounding=ROUND_CEILING))
            targets[label] = {'percent': threshold, 'required_correct': required,
                              'meets_target': correct >= required, 'margin_correct': correct-required}
        confusion = np.bincount(labels*10+pred, minlength=100).reshape(10, 10)
        results.append({'name': name, 'correct': correct, 'total': len(labels),
                        'accuracy': correct / len(labels), 'targets': targets,
                        'confusion_matrix': confusion.tolist()})
    output = {'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
              'prediction_manifest_sha256': sha(args.manifest), 'selection_sha256': sha(args.selection),
              'test_labels_sha256': label_sha, 'selected_name': record['selected_name'], 'results': results}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'results'}))
    print(json.dumps([{k:v for k,v in x.items() if k != 'confusion_matrix'} for x in results], indent=2))


if __name__ == '__main__':
    main()
