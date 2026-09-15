"""Prepare, freeze, and score the QDA submission's draws.

Recorded evidence is read-only. Use --evidence-dir generated/reproduction for
new prepare/freeze/score runs; verify.py checks the imported evidence by default.
Predictions are written and hashed before any evaluation-label slice is taken.
The seeds and config come from protocol.json, or from the protocol named by
SUTRO_PROTOCOL (protocol_fresh.json for the independently frozen evaluation in
evidence/fresh/accuracy, the one writable directory under evidence/).
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence/accuracy'
sys.path.insert(0, str(HERE))
from reference import train_predict, f32

PROTOCOL = Path(os.environ.get('SUTRO_PROTOCOL', HERE / 'protocol.json'))
_protocol = json.loads(PROTOCOL.read_text())
SEEDS = list(_protocol['dataset_seeds'])
CONFIG = _protocol['config']
FRESH = HERE / 'evidence/fresh'       # the independently frozen 2026-09-15 evaluation (protocol_fresh.json)


def _repo():
    if os.environ.get('SUTRO_REPO'):
        return Path(os.environ['SUTRO_REPO'])
    for base in (HERE, *HERE.parents):
        if (base / 'mnist/code/data.py').exists():
            return base
    raise SystemExit('sutro-problems checkout not found; set SUTRO_REPO')


sys.path.insert(0, str(_repo()))
from mnist.code import data as ds


def _raw():
    if os.environ.get('SUTRO_RAW'):
        return Path(os.environ['SUTRO_RAW'])
    for base in (HERE, *HERE.parents):
        for candidate in (base / 'mnist/data/raw', base / 'data/raw'):
            if (candidate / 'train-images-idx3-ubyte.gz').exists():
                return candidate
    raise SystemExit('raw MNIST dir not found; set SUTRO_RAW (run python -m mnist.code.data first)')


RAW = _raw()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def writable_evidence(evidence):
    resolved = evidence.resolve()
    require(not resolved.is_relative_to((HERE / 'evidence').resolve()) or resolved.is_relative_to(FRESH.resolve()),
            'Imported evidence is read-only; use --evidence-dir generated/reproduction')


def verify_raw():
    """Verify the two canonical MNIST gzip sources used by these draws."""
    records = {}
    for key in ('train_images', 'train_labels'):
        filename, expected = ds.SOURCES[key]
        path = RAW / filename
        actual = ds.file_hash(path, 'md5')
        require(actual == expected, f'Invalid canonical MNIST MD5: {filename}')
        records[filename] = {'md5': actual, 'sha256': ds.file_hash(path), 'bytes': path.stat().st_size}
    return records


@lru_cache(maxsize=1)
def _arrays():
    verify_raw()
    return (ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True),
            ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False))


def resize_recorded(images, size=3):
    """Reproduce the source BLAS's ordered FP32 multiply-adds for the 3x3 area resize.

    Different BLAS builds round these small matrix products differently, so the
    repository's `area_resize` (a float32 matmul) does not reproduce the archived
    input hashes on every machine. This keeps the repository's box-area weights
    and increasing reduction order, taking a float64 product and sum before each
    FP32 accumulation; it reproduces all 22 archived input hashes of the original
    draws and the fresh draws. Same construction as the merged medium PCA-QDA
    entry's `resize_recorded`, at size 3.
    """
    weights = ds.area_weights(28, size).astype(np.float64)
    images = np.asarray(images, dtype=f32)
    horizontal = np.zeros((len(images), size, 28), dtype=f32)
    for k in range(28):
        horizontal = (horizontal.astype(np.float64)
                      + weights[None, :, k, None] * images[:, None, k, :].astype(np.float64)).astype(f32)
    result = np.zeros((len(images), size, size), dtype=f32)
    for k in range(28):
        result = (result.astype(np.float64)
                  + horizontal[:, :, k, None].astype(np.float64) * weights[None, None, :, k]).astype(f32)
    return result


def _load(seed):
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, test = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
    pixels, labels = _arrays()
    x = resize_recorded(pixels[train].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
    q = resize_recorded(pixels[test].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
    return train, test, x, q, labels[train], pixels, labels


def validate_records(document, name):
    require(document['seeds'] == SEEDS, f'{name}: incorrect seeds')
    require(len(document['draws']) == len(SEEDS), f'{name}: expected 11 draws')
    for index, (seed, record) in enumerate(zip(SEEDS, document['draws'], strict=True)):
        require(record['draw'] == index and record['dataset_seed'] == seed,
                f'{name}: draw/seed/order mismatch at draw {index}')


def check_inputs(record, train, test, x, q):
    for key, array in (('train_indices', train), ('test_indices', test),
                       ('train_input', x), ('test_input', q)):
        require(ds.array_hash(array) == record[key + '_sha256'],
                f'Draw {record["draw"]}: {key} hash mismatch; use requirements.txt and Python 3.11')
    require(record['train_indices'] == train.tolist() and record['test_indices'] == test.tolist(),
            f'Draw {record["draw"]}: recorded indices differ')
    require(len(np.unique(np.concatenate((train, test)))) == 2000,
            f'Draw {record["draw"]}: duplicate or overlapping indices')


def frozen_predictions(evidence):
    """Verify every prediction file before the caller scores any evaluation labels."""
    manifest = read_json(evidence / 'prediction_manifest.json')
    validate_records(manifest, 'prediction manifest')
    require(manifest['config'] == CONFIG and manifest['test_labels_opened'] is False,
            'Prediction manifest config or freeze flag differs')
    predictions = []
    for record in manifest['draws']:
        require(record['path'] == f'predictions/draw-{record["draw"]:02d}.npy',
                'Unexpected prediction file path')
        path = evidence / record['path']
        require(ds.file_hash(path) == record['prediction_sha256'], f'Prediction file hash differs: {path}')
        pred = np.load(path, allow_pickle=False)
        require(pred.shape == (1000,) and pred.dtype == np.dtype('int64'), 'Invalid prediction array layout')
        require(bool(np.all((0 <= pred) & (pred < 10))), 'Prediction outside classes 0 through 9')
        require(ds.array_hash(pred) == record['array_sha256'], f'Prediction array hash differs: {path}')
        predictions.append(pred)
    return manifest, predictions


def prepare(evidence=EVIDENCE):
    writable_evidence(evidence)
    require(not (evidence / 'draw_manifest.json').exists(), 'Draw manifest exists; choose a fresh evidence directory')
    draws = []
    for index, seed in enumerate(SEEDS):
        train, test, x, q, *_ = _load(seed)
        draws.append({'draw': index, 'dataset_seed': seed,
                      'train_indices_sha256': ds.array_hash(train), 'test_indices_sha256': ds.array_hash(test),
                      'train_input_sha256': ds.array_hash(x), 'test_input_sha256': ds.array_hash(q),
                      'train_indices': train.tolist(), 'test_indices': test.tolist()})
        print('prepared draw', index, flush=True)
    write_new_json(evidence / 'draw_manifest.json', {'seeds': SEEDS, 'protocol': PROTOCOL.name, 'protocol_sha256': ds.file_hash(PROTOCOL),
                                                      'prepared_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'draws': draws})


def freeze(evidence=EVIDENCE):
    writable_evidence(evidence)
    draws = read_json(evidence / 'draw_manifest.json')
    validate_records(draws, 'draw manifest')
    require(not (evidence / 'prediction_manifest.json').exists(), 'Do not overwrite frozen predictions')
    predictions = evidence / 'predictions'
    predictions.mkdir(parents=True, exist_ok=False)
    records = []
    for index, seed in enumerate(SEEDS):
        train, test, x, q, ytrain, *_ = _load(seed)
        check_inputs(draws['draws'][index], train, test, x, q)
        params, scores, pred = train_predict(x, ytrain, q)
        path = predictions / f'draw-{index:02d}.npy'
        np.save(path, pred)
        records.append({'draw': index, 'dataset_seed': seed, 'path': str(path.relative_to(evidence)),
                        'prediction_sha256': ds.file_hash(path), 'array_sha256': ds.array_hash(pred),
                        'mu_sha256': ds.array_hash(params['mu']), 'packed_sha256': ds.array_hash(params['packed']),
                        'kappa_sha256': ds.array_hash(params['kappa']), 'scores_sha256': ds.array_hash(scores)})
        print('frozen draw', index, flush=True)
    write_new_json(evidence / 'prediction_manifest.json',
                   {'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'config': CONFIG, 'protocol': PROTOCOL.name,
                    'protocol_sha256': ds.file_hash(PROTOCOL), 'learner_sha256': ds.file_hash(HERE / 'reference.py'),
                    'seeds': SEEDS, 'draws': records, 'test_labels_opened': False})


def accuracy_result(counts):
    acc = np.array([c['correct'] / c['total'] for c in counts])
    total = sum(c['correct'] for c in counts)
    return {'draws': counts, 'correct': total, 'total': 11000, 'mean_accuracy': float(acc.mean()),
            'sample_sd_pp': float(acc.std(ddof=1) * 100), 'target_correct': 7370, 'target_met': total >= 7370}


def score(evidence=EVIDENCE):
    writable_evidence(evidence)
    for name in ('accuracy.json', 'evaluation_freeze.json'):
        require(not (evidence / name).exists(), f'{name} exists; choose a fresh evidence directory')
    manifest, predictions = frozen_predictions(evidence)
    draws = read_json(evidence / 'draw_manifest.json')
    validate_records(draws, 'draw manifest')
    # Validate all draw indices and inputs before taking evaluation-label slices.
    test_indices = []
    for record in draws['draws']:
        train, test, x, q, *_ = _load(record['dataset_seed'])
        check_inputs(record, train, test, x, q)
        test_indices.append(test)
    labels = _arrays()[1]
    counts = []
    for record, test, pred in zip(manifest['draws'], test_indices, predictions, strict=True):
        counts.append({'draw': record['draw'], 'dataset_seed': record['dataset_seed'],
                       'correct': int((pred == labels[test]).sum()), 'total': 1000})
    result = accuracy_result(counts)
    write_new_json(evidence / 'accuracy.json', result)
    write_new_json(evidence / 'evaluation_freeze.json',
                   {'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'prediction_manifest_sha256': ds.file_hash(evidence / 'prediction_manifest.json')})
    print(json.dumps(result, indent=2))


def gpu_payloads(evidence=EVIDENCE, output_dir=None, single=False):
    """Export verified normalized inputs and frozen predictions for GPU reproduction."""
    manifest, predictions = frozen_predictions(evidence)
    draws = read_json(evidence / 'draw_manifest.json')
    validate_records(draws, 'draw manifest')
    out = output_dir or (HERE / 'generated' if single else HERE / 'generated/payloads')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, seed in enumerate(SEEDS[:1] if single else SEEDS):
        train, test, x, q, ytrain, _, labels = _load(seed)
        check_inputs(draws['draws'][index], train, test, x, q)
        arrays = dict(x=x, labels=ytrain.astype(np.int64), q=q,
                      frozen_predictions=predictions[index], test_labels=labels[test].astype(np.int64),
                      dataset_seed=np.asarray(seed, dtype=np.int64))
        path = out / ('payload.npz' if single else f'draw-{index:02d}.npz')
        np.savez(path, **arrays)
        records.append({'draw': index, 'dataset_seed': seed, 'path': path.name, 'sha256': ds.file_hash(path),
                        'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': ds.array_hash(v)}
                                   for k, v in arrays.items()},
                        'frozen_prediction_path': manifest['draws'][index]['path'],
                        'frozen_prediction_sha256': manifest['draws'][index]['prediction_sha256']})
        print('wrote', path, flush=True)
    if not single:
        sources = {name: ds.file_hash(HERE / name) for name in ('run.py', 'reference.py', 'protocol.json')}
        sources['mnist/code/data.py'] = ds.file_hash(Path(ds.__file__))
        result = {'seeds': SEEDS, 'draws': records, 'source_sha256': sources,
                  'prediction_manifest_sha256': ds.file_hash(evidence / 'prediction_manifest.json'),
                  'draw_manifest_sha256': ds.file_hash(evidence / 'draw_manifest.json')}
        (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'score', 'gpu-payload', 'gpu-payloads'))
    parser.add_argument('--evidence-dir', type=Path, default=EVIDENCE)
    parser.add_argument('--output-dir', type=Path, help='GPU payload output directory')
    args = parser.parse_args()
    if args.command.startswith('gpu-payload'):
        gpu_payloads(args.evidence_dir, args.output_dir, single=args.command == 'gpu-payload')
    else:
        require(args.output_dir is None, '--output-dir applies only to GPU payload commands')
        {'prepare': prepare, 'freeze': freeze, 'score': score}[args.command](args.evidence_dir)


if __name__ == '__main__':
    main()
