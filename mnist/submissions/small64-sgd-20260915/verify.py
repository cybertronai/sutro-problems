"""Re-check frozen manifests, inputs, predictions, accuracy and A100 results."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE / 'evidence/accuracy'


def repo():
    env = os.environ.get('SUTRO_REPO')
    if env:
        return Path(env)
    for base in (HERE, *HERE.parents):
        if (base / 'mnist/code/data.py').exists():
            return base
        if (base / 'vendor/sutro-problems/mnist/code/data.py').exists():
            return base / 'vendor/sutro-problems'
    raise SystemExit('sutro-problems checkout not found; set SUTRO_REPO')


sys.path.insert(0, str(repo()))
sys.path.insert(0, str(HERE))
from mnist.code import data as ds  # noqa: E402
import reference  # noqa: E402
import run  # noqa: E402


def raw():
    env = os.environ.get('SUTRO_RAW')
    if env:
        return Path(env)
    candidates = [HERE / 'data/certification/draw-00/raw', repo() / 'matmul/mnist_cache']
    for base in (HERE, *HERE.parents):
        candidates.append(base / 'data/certification/draw-00/raw')
    for candidate in candidates:
        if (candidate / 'train-images-idx3-ubyte.gz').exists():
            return candidate
    raise SystemExit('raw MNIST dir not found; set SUTRO_RAW')


draws = json.loads((E / 'draw_manifest.json').read_text())
preds = json.loads((E / 'prediction_manifest.json').read_text())
acc = json.loads((E / 'accuracy.json').read_text())
assert [d['dataset_seed'] for d in draws['draws']] == [d['dataset_seed'] for d in preds['draws']]
labels = ds.read_idx(raw() / 'train-labels-idx1-ubyte.gz', 60000, False)
pixels = ds.read_idx(raw() / 'train-images-idx3-ubyte.gz', 60000, True)
for name, expected in draws['raw_gz_sha256'].items():
    assert ds.file_hash(raw() / name) == expected, f'raw MNIST {name} SHA-256 differs from the draw manifest'
f32 = np.float32
total = 0
for d, record in zip(draws['draws'], preds['draws'], strict=True):
    order = np.random.Generator(np.random.PCG64(d['dataset_seed'])).permutation(60000)
    train, test = order[:1000], order[1000:2000]
    assert ds.array_hash(train) == d['train_indices_sha256']
    assert ds.array_hash(test) == d['test_indices_sha256']
    inputs = run._arrays(pixels, labels, train, test)
    for name, value in inputs.items():
        assert ds.array_hash(value) == d['input_sha256'][name], \
            f'draw {d["draw"]}: {name} preprocessed-input hash differs from the draw manifest'
    path = E / record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['prediction_sha256']
    pred = np.load(path, allow_pickle=False)
    assert pred.shape == (1000,) and ((pred >= 0) & (pred < 10)).all()
    assert ds.array_hash(pred) == record['array_sha256']
    total += int((pred == labels[test]).sum())
assert total == acc['correct'] == sum(r['correct'] for r in acc['draws'])
assert acc['target_met'] is True
print('PASS: draws, indices, prediction hashes and label-derived accuracy consistent')
print('PASS: raw-source SHA-256 and all 11 draws\' train-image, train-label and test-image '
      'input hashes match the draw manifest')

# Re-derive draw 0 end-to-end from the ordered FP32 reference and compare bits.
# Downsampling uses run.resize_recorded: ordered accumulation with float64
# product/sum intermediates and a float32 cast after each step, so the inputs
# are bit-identical on every platform instead of BLAS-dispatch-dependent.
d0 = draws['draws'][0]
order = np.random.Generator(np.random.PCG64(d0['dataset_seed'])).permutation(60000)
train0, test0 = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
x0 = run.resize_recorded(pixels[train0].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
q0 = run.resize_recorded(pixels[test0].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
target0 = (labels[train0][:, None] == np.arange(10)).astype(f32)
params0, scores0, pred0 = reference.train_predict(x0, q0, target0, dict(reference.CONFIG))
assert ds.array_hash(pred0) == preds['draws'][0]['array_sha256'], 'draw 0 prediction bits changed'
assert ds.array_hash(params0) == preds['draws'][0]['parameter_sha256'], 'draw 0 parameter bits changed'
assert ds.array_hash(scores0) == preds['draws'][0]['scores_sha256'], 'draw 0 score bits changed'
print('PASS: draw 0 reproduced bit-exactly from the ordered FP32 reference and deterministic area resize')

# The pilot sweep used ascending-K einsum matmuls; confirm they agree bit-for-bit
# with the ordered loops on the frozen configuration and draw 0.
def fast_mm(a, b):
    return np.einsum('ik,kj->ij', a, np.ascontiguousarray(b), optimize=False, dtype=np.float32)


def fast_rows(a):
    return np.einsum('ij->j', a, optimize=False, dtype=np.float32)


_, _, pred_fast = reference.train_predict(x0, q0, target0, dict(reference.CONFIG), fast_mm, fast_rows)
assert np.array_equal(pred_fast, pred0), 'einsum and ordered predictions diverge'
print('PASS: pilot einsum matmuls are bit-equivalent to ordered loops (draw 0)')

gpu = HERE / 'results/gpu_results.json'
if gpu.exists():
    gpu_results = json.loads(gpu.read_text())
    assert gpu_results['model_config'] == {'width': 64, 'epochs': 500, 'learning_rate': 0.1,
                                           'seed': 101, 'batch_size': 25}
    canonical = gpu_results['validation']['canonical']
    assert canonical['prediction_matches'] == 1000 and canonical['parameter_bitwise_matches'] == 1290
    frozen0 = np.load(E / preds['draws'][0]['path'], allow_pickle=False)
    assert np.array_equal(np.array(gpu_results['predictions'], dtype=np.int64), frozen0), \
        'GPU canonical predictions differ from frozen CPU predictions'
    manifest_hash = preds['draws'][0]['parameter_sha256']
    gpu_param_bits = np.array(gpu_results['final_parameter_bits_u32'], dtype=np.uint32)
    assert ds.array_hash(gpu_param_bits.astype('<u4')) == manifest_hash, 'GPU parameter bits differ'
    # Recompute idle-adjusted energy from raw counters.
    for trial in gpu_results['trials']:
        active, idle = trial['active'], trial['paired_idle_power_w']
        recomputed = (trial['active']['energy_j'] - idle * trial['active']['duration_s']) / trial['invocations']
        assert abs(recomputed - trial['idle_adjusted_j_per_invocation']) < 1e-12
    print('PASS: A100 results consistent with frozen evidence; idle-adjusted energy recomputed')
else:
    print('NOTE: results/gpu_results.json absent; A100 checks skipped')
