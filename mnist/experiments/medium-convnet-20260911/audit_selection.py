"""Independently recompute validation selection using only Python's standard library.

Reads a physically training-only NPZ and retained validation logits. It imports
no training code, opens no test arrays, and never trains or evaluates a model.
Default output is stdout; pass --output to retain a JSON audit.
"""
import argparse
import array
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npy(raw):
    require(raw[:6] == b'\x93NUMPY', 'Not an NPY payload')
    version = tuple(raw[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported NPY version')
    start = 10 if version == (1, 0) else 12
    length = struct.unpack('<H' if start == 10 else '<I', raw[8:start])[0]
    metadata = ast.literal_eval(raw[start:start+length].decode('utf-8'))
    require(not metadata['fortran_order'], 'Expected C-order arrays')
    return metadata, raw[start+length:]


def numbers(raw, dtype, shape):
    metadata, payload = read_npy(raw)
    require(metadata['shape'] == shape and metadata['descr'] == dtype,
            'Unexpected array shape or dtype')
    values = array.array({'<f4': 'f', '<i8': 'q'}[dtype])
    require(values.itemsize == (4 if dtype == '<f4' else 8), 'Unsupported native scalar size')
    values.frombytes(payload)
    if sys.byteorder != 'little':
        values.byteswap()
    expected_size = 1
    for dimension in shape:
        expected_size *= dimension
    require(len(values) == expected_size, 'Unexpected NPY payload length')
    return values, hashlib.sha256(payload).hexdigest()


def initial_rank(row):
    return (-row['best']['validation']['correct'], row['best']['validation']['loss'],
            row['parameter_count'], row['config']['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=HERE)
    parser.add_argument('--data', type=Path, default=REPO/'mnist/data/medium-train-only.npz')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    study = args.study
    protocol = json.loads((study/'protocol.json').read_text())
    split = json.loads((study/'validation_split.json').read_text())
    selected = json.loads((study/'validation_selection.json').read_text())
    frozen = json.loads((study/'selection.json').read_text())
    canonical = json.loads((REPO/'mnist/doc/dataset_manifest.json').read_text())['tiers']['medium']['arrays']
    require(sha(study/'protocol.json') == frozen['protocol_sha256'], 'Changed frozen protocol')
    require(sha(study/'validation_selection.json') == frozen['validation_selection_sha256'],
            'Changed validation selection')
    require(sha(study/'validation_audit.json') == frozen['validation_audit_sha256'], 'Changed run audit')
    require(sha(study/'validation_split.json') == protocol['split_file_sha256'], 'Changed split')
    for name, digest in frozen['source_sha256'].items():
        require(sha(HERE/name) == digest, 'Changed source: '+name)
    with zipfile.ZipFile(args.data) as archive:
        require(set(archive.namelist()) <= {'train_images.npy', 'train_labels.npy', 'train_indices.npy'},
                'Expected physically training-only input')
        for name in ('train_images', 'train_labels'):
            metadata, payload = read_npy(archive.read(name+'.npy'))
            spec = canonical[name]
            expected_dtype = '<f4' if name == 'train_images' else '<i8'
            digest = hashlib.sha256(payload).hexdigest()
            require(metadata['shape'] == tuple(spec['shape']) and metadata['descr'] == expected_dtype,
                    'Noncanonical input shape or dtype: '+name)
            require(digest == spec['sha256_c_order_little_endian'] == protocol['input_sha256'][name],
                    'Noncanonical training input: '+name)
        labels, _ = numbers(archive.read('train_labels.npy'), '<i8', (6000,))
    fit, validation = split['fit_positions'], split['validation_positions']
    require(len(fit) == 4800 and len(validation) == 1200 and sorted(fit+validation) == list(range(6000)),
            'Invalid training/validation partition')
    require(split['seed'] == protocol['split_seed'] == 20260914, 'Changed split seed')
    truth = [labels[i] for i in validation]
    configs = {c['id']: c for c in protocol['candidates']}
    runs, retained_logits, evidence = [], {}, []
    expected_provenance = {'protocol_sha256': sha(study/'protocol.json'),
        'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
        'split_file_sha256': protocol['split_file_sha256'],
        'allowed_input_arrays': ['train_images', 'train_labels']}
    actual = set()
    for path in sorted((study/'results').glob('*.json')):
        row = json.loads(path.read_text())
        key = (row['config']['id'], row['phase'], row['seed'])
        require(key not in actual, 'Duplicate run'); actual.add(key)
        require(row['config'] == configs[key[0]] and row['provenance'] == expected_provenance,
                'Changed run configuration or provenance')
        require(row['id'] == '%s-%s-s%s' % (key[1], key[0], key[2]) and path.stem == row['id'],
                'Incorrect run identity')
        logits_path = study/row['validation_logits_file']
        logits, digest = numbers(logits_path.read_bytes(), '<f4', (1200, 10))
        require(digest == row['best_validation_logits_sha256'], 'Changed validation logits')
        require(all(math.isfinite(v) for v in logits), 'Nonfinite validation logits')
        predictions, losses = [], []
        for i, label in enumerate(truth):
            values = logits[i*10:i*10+10]
            predictions.append(max(range(10), key=lambda j: values[j]))
            maximum = max(values)
            losses.append(math.log(sum(math.exp(v-maximum) for v in values))-(values[label]-maximum))
        correct = sum(a == b for a, b in zip(predictions, truth))
        require(correct == row['best']['validation']['correct'] and
                predictions == row['best_validation_predictions'], 'Incorrect saved predictions or count')
        recomputed_loss = sum(losses)/1200
        require(abs(recomputed_loss-row['best']['validation']['loss']) < 2e-6, 'Incorrect saved loss')
        require([h['epoch'] for h in row['history']] == list(range(1, row['config']['epochs']+1)),
                'Incomplete epoch history')
        best = min(row['history'], key=lambda h: (-h['validation']['correct'], h['validation']['loss'], h['epoch']))
        require(best == row['best'] and row['last'] == row['history'][-1], 'Incorrect best checkpoint')
        require(sha(study/row['checkpoint_file']) == row['best_checkpoint_sha256'], 'Changed checkpoint')
        runs.append(row); retained_logits[row['id']] = logits
        evidence.append({'id': row['id'], 'result_sha256': sha(path),
                         'logits_file_sha256': sha(logits_path), 'correct': correct,
                         'recomputed_cross_entropy': recomputed_loss})
    search = sorted([r for r in runs if r['phase'] == 'search'], key=initial_rank)
    top3 = [r['config']['id'] for r in search[:3]]
    expected = {(c, 'search', 11) for c in configs} | {(c, 'replicate', seed) for c in top3 for seed in (22, 33)}
    require(actual == expected and len(runs) == 16, 'Incomplete or unexpected run set')
    ranking = []
    for config_id in top3:
        reps = sorted([r for r in runs if r['config']['id'] == config_id], key=lambda r: r['seed'])
        logits = [retained_logits[r['id']] for r in reps]
        ensemble = [max(range(10), key=lambda j: sum(a[i*10+j] for a in logits)/3) for i in range(1200)]
        ensemble_correct = sum(a == b for a, b in zip(ensemble, truth))
        ranking.append({'config_id': config_id,
            'mean_best_validation_correct': statistics.mean(r['best']['validation']['correct'] for r in reps),
            'mean_best_validation_loss': statistics.mean(r['best']['validation']['loss'] for r in reps),
            'parameter_count': reps[0]['parameter_count'],
            'refit_epochs': int(statistics.median(r['best']['epoch'] for r in reps)),
            'best_epochs_by_seed': {str(r['seed']): r['best']['epoch'] for r in reps},
            'individual_correct_by_seed': {str(r['seed']): r['best']['validation']['correct'] for r in reps},
            'validation_logits_average_ensemble_correct': ensemble_correct,
            'selected_inference': 'ensemble' if ensemble_correct > reps[0]['best']['validation']['correct'] else 'single'})
    ranking.sort(key=lambda r: (-r['mean_best_validation_correct'], r['mean_best_validation_loss'],
                               r['parameter_count'], r['config_id']))
    require([r['config_id'] for r in ranking] == [r['config']['id'] for r in selected['ranked_candidates']],
            'Incorrect aggregate ranking')
    for recomputed, saved in zip(ranking, selected['ranked_candidates']):
        for key, value in recomputed.items():
            if key not in ('config_id', 'selected_inference'):
                require(value == saved[key], 'Incorrect aggregate field: '+key)
    winner = ranking[0]
    require(selected['preferred_candidate'] == selected['ranked_candidates'][0], 'Incorrect preferred candidate')
    require(frozen['config'] == configs[winner['config_id']] and frozen['epochs'] == winner['refit_epochs'],
            'Frozen architecture or epoch disagrees with validation')
    require(frozen['selected_inference'] == selected['preferred_candidate']['selected_inference'] == winner['selected_inference'],
            'Incorrect single/ensemble decision')
    require(frozen['seeds'] == [101, 102, 103] and frozen['primary_single_seed'] == 101 and
            frozen['schedule_epochs'] == 100, 'Changed refit policy')
    require(frozen['selected_name'] == ('ensemble' if winner['selected_inference'] == 'ensemble' else 'seed101'),
            'Incorrect selected prediction name')
    record = {'passed': True, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'test_arrays_opened': False, 'training_code_imported': False,
        'implementation': 'Standard-library NPY decoding and Python float64 arithmetic',
        'scope': 'Saved logits, histories, aggregate selection and frozen configuration; no retraining or checkpoint tensor replay',
        'audit_source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(study/'protocol.json'),
        'selection_sha256': sha(study/'selection.json'),
        'validation_selection_sha256': sha(study/'validation_selection.json'),
        'initial_top3': top3, 'replicated_ranking': ranking,
        'selected_name': frozen['selected_name'], 'runs': evidence}
    serialized = json.dumps(record, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
