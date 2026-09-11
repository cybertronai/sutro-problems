"""Train-only-selected ReLU/SGD feasibility study with ordered FP32 arithmetic.

Three separate phases enforce a frozen shortlist and frozen predictions before
the local evaluator is allowed to open test_labels. The learner itself accepts
only train_images, train_labels, and test_images. No BLAS is used: unoptimized
einsum is checked against scalar-reduction FP32 operations on this installation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np

WIDTHS = (16, 32, 64)
EPOCHS = (100, 300, 1000)
RATES = (0.01, 0.05, 0.2)
TARGETS = (55, 60, 65, 70, 75)
SEARCH_SEED = 11
FINAL_SEEDS = (101, 102, 103)
SPLIT_SEED = 20260912
BATCH = 30


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ordered_mm(a, b):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        out = out + a[:, k, None] * b[None, k, :]
    return out


def fast_mm(a, b):
    # A strided/transposed RHS can select a different einsum reduction kernel.
    # Materializing its row-major layout keeps the ascending-K operation order.
    return np.einsum('ik,kj->ij', a, np.ascontiguousarray(b),
                     optimize=False, dtype=np.float32)


def ordered_rows(a):
    out = np.zeros(a.shape[1], dtype=np.float32)
    for row in a:
        out = out + row
    return out


def fast_rows(a):
    return np.einsum('ij->j', a, optimize=False, dtype=np.float32)


def parameters(width, seed):
    # Random constants depend only on predeclared architecture and seed, never
    # on training or test examples. Their raw FP32 bits can be emitted as set.
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1 / 3, 1 / 3, (9, width)).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                        (width, 10)).astype(np.float32),
            np.zeros(10, dtype=np.float32)]


def transform(images):
    return images.reshape(len(images), 9) * np.float32(4) - np.float32(0.5)


def forward(x, params, mm=fast_mm):
    w1, b1, w2, b2 = params
    z = mm(x, w1) + b1
    h = np.where(z > np.float32(0), z, np.float32(0))
    return z, h, mm(h, w2) + b2


def predict(x, params, mm=fast_mm):
    # First-column tie breaking is the same as an ascending strict-> scan.
    return np.argmax(forward(x, params, mm)[2], axis=1).astype(np.int64)


def update(x, target, params, rate, mm=fast_mm, rows=fast_rows):
    w1, b1, w2, b2 = params
    z, h, scores = forward(x, params, mm)
    d2 = scores - target
    back = mm(d2, w2.T)
    d1 = np.where(z > np.float32(0), back, np.float32(0))
    g1, gb1 = mm(x.T, d1), rows(d1)
    g2, gb2 = mm(h.T, d2), rows(d2)
    # All four gradients use the same pre-update weights. Squared-error loss
    # is 0.5*sum_class(error**2), averaged across this complete minibatch.
    step = np.float32(rate / len(x))
    return [w1 - step * g1, b1 - step * gb1,
            w2 - step * g2, b2 - step * gb2]


def epoch(x, target, params, rate, mm=fast_mm, rows=fast_rows):
    for start in range(0, len(x), BATCH):
        params = update(x[start:start+BATCH], target[start:start+BATCH],
                        params, rate, mm, rows)
    return params


def inputs(path, manifest_path):
    # This is the learner's only NPZ access; test_labels is deliberately absent.
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in ('train_images', 'train_labels', 'test_images')}
    manifest = json.loads(manifest_path.read_text())
    canonical = manifest['tiers']['small']['arrays']
    for key, value in data.items():
        assert list(value.shape) == canonical[key]['shape'], key
        assert str(value.dtype) == canonical[key]['dtype'], key
        assert digest(value) == canonical[key]['sha256_c_order_little_endian'], key
    return data


def checks(data):
    rng = np.random.Generator(np.random.PCG64(941))
    evidence = []
    for n, k, m in ((30, 9, 64), (30, 64, 10), (30, 10, 64),
                    (9, 30, 64), (64, 30, 10)):
        a = rng.normal(size=(n, k)).astype(np.float32)
        b = rng.normal(size=(k, m)).astype(np.float32)
        assert np.array_equal(ordered_mm(a, b), fast_mm(a, b))
        evidence.append({'matrix_shape': [n, k, m], 'bitwise_equal': True})
    x = transform(data['train_images'])
    target = (data['train_labels'][:, None] == np.arange(10)).astype(np.float32)
    p = parameters(64, 101)
    fast = epoch(x, target, p, 0.05)
    slow = epoch(x, target, p, 0.05, ordered_mm, ordered_rows)
    assert all(np.array_equal(a, b) for a, b in zip(fast, slow))
    evidence.append({'full_training_epoch_width_64': 'all 4 parameter arrays bitwise equal',
                     'minibatches': 20})
    return evidence


def plan():
    return {'widths': WIDTHS, 'checkpoint_epochs': EPOCHS, 'learning_rates': RATES,
            'validation_search_seed': SEARCH_SEED, 'final_seeds': FINAL_SEEDS,
            'targets_percent': TARGETS, 'batch_size': BATCH,
            'split_seed': SPLIT_SEED,
            'split': 'one fixed PCG64 permutation; first 480 training, last 120 validation',
            'selection': 'best validation count at each epoch budget, plus minimum epochs*width '
                         'candidate meeting each validation target; ties lower epochs*width, '
                         'lower width, lower learning rate; deduplicate; freeze before any test evaluation',
            'test_use': 'generate every final prediction with frozen configs and seeds before opening labels',
            'learner_npz_allowlist': ['train_images', 'train_labels', 'test_images'],
            'algorithm': '9-H-10 ReLU, fixed x*4-0.5, squared-error SGD, cyclic ordered minibatches; '
                         'separate FP32 multiply/add and ascending reduction order; first-index argmax'}


def search(args):
    if (args.output / 'shortlist.json').exists():
        raise RuntimeError('Shortlist already exists; use a fresh directory to preserve evidence')
    write_json(args.output / 'predeclared_plan.json', plan())
    data = inputs(args.data, args.manifest)
    arithmetic_checks = checks(data)
    order = np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(600)
    train, val = order[:480], order[480:]
    x = transform(data['train_images'])
    y = data['train_labels']
    target = (y[train, None] == np.arange(10)).astype(np.float32)
    split = {'training_rows': train.tolist(), 'validation_rows': val.tolist(),
             'allowed_input_hashes': {k: digest(v) for k, v in data.items()},
             'arithmetic_checks': arithmetic_checks}
    write_json(args.output / 'validation_split.json', split)
    results = []
    for width in WIDTHS:
        for rate in RATES:
            params = parameters(width, SEARCH_SEED)
            begin = time.perf_counter()
            for ep in range(1, max(EPOCHS) + 1):
                with np.errstate(over='ignore', invalid='ignore'):
                    params = epoch(x[train], target, params, rate)
                if not all(np.isfinite(p).all() for p in params):
                    results.append({'width': width, 'learning_rate': rate, 'epochs': ep,
                                    'status': 'nonfinite_training', 'validation_correct': None})
                    print(f'width={width} lr={rate}: nonfinite at epoch {ep}', flush=True)
                    break
                if ep in EPOCHS:
                    correct = int((predict(x[val], params) == y[val]).sum())
                    row = {'width': width, 'learning_rate': rate, 'epochs': ep,
                           'validation_correct': correct, 'validation_total': len(val),
                           'validation_accuracy': correct / len(val), 'status': 'finite',
                           'training_correct': int((predict(x[train], params) == y[train]).sum()),
                           'train_reference_seconds_to_checkpoint': time.perf_counter() - begin}
                    results.append(row)
                    write_json(args.output / 'validation_results.json', results)
                    print(f'width={width} lr={rate} epoch={ep} validation={correct}/120', flush=True)
    finite = [r for r in results if r['status'] == 'finite']
    def cost_key(r):
        return (r['epochs'] * r['width'], r['width'], r['learning_rate'])
    selected = []
    for ep in EPOCHS:
        eligible = [r for r in finite if r['epochs'] == ep]
        if eligible:
            selected.append(min(eligible, key=lambda r: (-r['validation_correct'], *cost_key(r))))
    for target_percent in TARGETS:
        eligible = [r for r in finite if 100*r['validation_correct'] >= target_percent*r['validation_total']]
        if eligible:
            selected.append(min(eligible, key=cost_key))
    unique = {f"h{r['width']}-e{r['epochs']}-lr{r['learning_rate']:g}": r for r in selected}
    shortlist = [{'id': name, **r} for name, r in sorted(unique.items())]
    write_json(args.output / 'validation_results.json', results)
    write_json(args.output / 'shortlist.json', {'selection_plan': plan()['selection'], 'candidates': shortlist,
               'final_seeds': FINAL_SEEDS, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(shortlist, indent=2), flush=True)


def extend(args):
    """One declared train-only extension; stage-one evidence remains immutable."""
    if (args.output / 'frozen_predictions.json').exists() or (args.output / 'extension_plan.json').exists():
        raise RuntimeError('Extension must precede final predictions and run only once')
    stage1 = json.loads((args.output / 'validation_results.json').read_text())
    write_json(args.output / 'stage1_validation_results.json', stage1)
    (args.output/'stage1_shortlist.json').write_text((args.output/'shortlist.json').read_text())
    extension = {'decision_evidence': 'Stage-one validation best was 71/120; no test labels opened.',
                 'widths': [32, 64, 128], 'learning_rates': [0.2, 0.5],
                 'checkpoint_epochs': [3000, 10000],
                 'selection': plan()['selection'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output / 'extension_plan.json', extension)
    data = inputs(args.data, args.manifest)
    split = json.loads((args.output/'validation_split.json').read_text())
    train, val = np.array(split['training_rows']), np.array(split['validation_rows'])
    x, y = transform(data['train_images']), data['train_labels']
    target = (y[train, None] == np.arange(10)).astype(np.float32)
    results = list(stage1)
    for width in extension['widths']:
        for rate in extension['learning_rates']:
            params, begin = parameters(width, SEARCH_SEED), time.perf_counter()
            for ep in range(1, max(extension['checkpoint_epochs'])+1):
                with np.errstate(over='ignore', invalid='ignore'):
                    params = epoch(x[train], target, params, rate)
                if not all(np.isfinite(p).all() for p in params):
                    results.append({'width': width, 'learning_rate': rate, 'epochs': ep,
                                    'status': 'nonfinite_training', 'validation_correct': None})
                    print(f'extension width={width} lr={rate}: nonfinite at {ep}', flush=True)
                    break
                if ep in extension['checkpoint_epochs']:
                    correct = int((predict(x[val], params) == y[val]).sum())
                    results.append({'width': width, 'learning_rate': rate, 'epochs': ep,
                                    'validation_correct': correct, 'validation_total': len(val),
                                    'validation_accuracy': correct/len(val), 'status': 'finite',
                                    'training_correct': int((predict(x[train], params)==y[train]).sum()),
                                    'train_reference_seconds_to_checkpoint': time.perf_counter()-begin})
                    write_json(args.output/'validation_results.json', results)
                    print(f'extension width={width} lr={rate} epoch={ep} validation={correct}/120', flush=True)
    finite = [r for r in results if r['status']=='finite']
    def cost_key(r):
        return (r['epochs']*r['width'], r['width'], r['learning_rate'])
    selected = []
    for ep in (*EPOCHS, *extension['checkpoint_epochs']):
        candidates = [r for r in finite if r['epochs']==ep]
        if candidates:
            selected.append(min(candidates, key=lambda r: (-r['validation_correct'], *cost_key(r))))
    for t in TARGETS:
        candidates = [r for r in finite if 100*r['validation_correct']>=t*r['validation_total']]
        if candidates:
            selected.append(min(candidates, key=cost_key))
    unique = {f"h{r['width']}-e{r['epochs']}-lr{r['learning_rate']:g}": r for r in selected}
    shortlist = [{'id': name, **r} for name,r in sorted(unique.items())]
    write_json(args.output/'validation_results.json', results)
    write_json(args.output/'shortlist.json', {'selection_plan': plan()['selection'], 'candidates': shortlist,
               'final_seeds': FINAL_SEEDS, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'search_extension': extension})
    print(json.dumps(shortlist,indent=2),flush=True)


def final(args):
    if (args.output / 'frozen_predictions.json').exists():
        raise RuntimeError('Predictions already frozen; do not overwrite evidence')
    data = inputs(args.data, args.manifest)
    selected = json.loads((args.output / 'shortlist.json').read_text())
    x, q = transform(data['train_images']), transform(data['test_images'])
    target = (data['train_labels'][:, None] == np.arange(10)).astype(np.float32)
    runs = []
    for config in selected['candidates']:
        for seed in selected['final_seeds']:
            name = f"{config['id']}-s{seed}"
            params = parameters(config['width'], seed)
            initial_hashes = [digest(p) for p in params]
            begin = time.perf_counter()
            for ep in range(config['epochs']):
                params = epoch(x, target, params, config['learning_rate'])
            predictions = predict(q, params)
            wall = time.perf_counter() - begin
            # Validate full final inference reductions against explicit primitives.
            ordered_prediction = predict(q, params, ordered_mm)
            assert np.array_equal(predictions, ordered_prediction)
            assert np.array_equal(forward(q, params)[2], forward(q, params, ordered_mm)[2])
            np.save(args.output / f'{name}-predictions.npy', predictions)
            # Plain JSON is sufficient to preserve predictions; binary checkpoints
            # are optional local evidence, not part of scored input.
            run = {'id': name, 'config_id': config['id'], 'width': config['width'],
                   'epochs': config['epochs'], 'learning_rate': config['learning_rate'], 'seed': seed,
                   'initial_parameter_sha256': initial_hashes,
                   'final_parameter_sha256': [digest(p) for p in params],
                   'predictions': predictions.tolist(), 'predictions_sha256_int64_le': digest(predictions),
                   'prediction_file': f'{name}-predictions.npy',
                   'training_correct': int((predict(x, params) == data['train_labels']).sum()),
                   'cpu_reference_train_and_infer_seconds': wall,
                   'final_inference_explicit_primitives_bitwise_equal': True}
            runs.append(run)
            print(f'{name}: frozen predictions, {wall:.2f}s train+infer', flush=True)
    write_json(args.output / 'frozen_predictions.json', {'runs': runs,
               'shortlist_sha256': hashlib.sha256((args.output/'shortlist.json').read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}})


def evaluate(args):
    # Evaluation is deliberately separate and only runs after all predictions
    # and the entire shortlist have been frozen on disk.
    frozen_file = args.output / 'frozen_predictions.json'
    frozen = json.loads(frozen_file.read_text())
    if (args.output / 'accuracy_results.json').exists():
        raise RuntimeError('Test evaluation already recorded; do not overwrite it')
    with np.load(args.data, allow_pickle=False) as z:
        test_labels = z['test_labels']
    manifest = json.loads(args.manifest.read_text())
    assert digest(test_labels) == manifest['tiers']['small']['arrays']['test_labels']['sha256_c_order_little_endian']
    rows = []
    for run in frozen['runs']:
        pred = np.asarray(run['predictions'], dtype=np.int64)
        assert digest(pred) == run['predictions_sha256_int64_le']
        row = {k: v for k, v in run.items() if k != 'predictions'}
        row.update(correct=int((pred == test_labels).sum()), total=len(test_labels))
        row['accuracy_percent'] = 100 * row['correct'] / row['total']
        rows.append(row)
    summaries = []
    for config_id in sorted({r['config_id'] for r in rows}):
        config_runs = [r for r in rows if r['config_id'] == config_id]
        values = [r['accuracy_percent'] for r in config_runs]
        summaries.append({'config_id': config_id, 'width': config_runs[0]['width'],
                          'epochs': config_runs[0]['epochs'], 'learning_rate': config_runs[0]['learning_rate'],
                          'correct_by_seed': [r['correct'] for r in config_runs],
                          'mean_accuracy_percent': float(np.mean(values)),
                          'sample_sd_percentage_points': float(np.std(values, ddof=1)),
                          'targets': {str(t): {'seeds_meeting': sum(v >= t for v in values),
                                             'seeds_evaluated': len(values)} for t in TARGETS}})
    out = {'kind': 'exploratory fixed-dataset higher-accuracy feasibility; not a new A100 submission',
           'dataset_profile': manifest['profile'], 'dataset_seed': manifest['seed'],
           'test_labels_sha256': digest(test_labels),
           'frozen_predictions_sha256': hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'a100_time_ps': None, 'a100_energy_fJ': None,
           'target_percentages': TARGETS, 'runs': rows, 'configurations': summaries,
           'limits': ['All final configurations and seeds were frozen using training validation only.',
                      'Results share one fixed 600-example test set; seed SD is not population uncertainty.',
                      'This finite search does not prove an unmet target impossible.',
                      'No A100 measurement has been made for these MLP configurations.']}
    write_json(args.output / 'accuracy_results.json', out)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('search', 'extend', 'final', 'evaluate'), required=True)
    p.add_argument('--data', type=Path, default=Path('mnist/data/small.npz'))
    p.add_argument('--manifest', type=Path, default=Path('mnist/doc/dataset_manifest.json'))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
