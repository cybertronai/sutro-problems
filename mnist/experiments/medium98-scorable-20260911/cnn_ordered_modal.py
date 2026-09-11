"""Training-only validation of the exact ordered CNN backend.

This stage accepts only the canonical training arrays. Validation rows and
labels are isolated from Trainer; they select stopping epochs locally in the
validation harness. There is deliberately no test or benchmark phase.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import modal

HERE = Path(__file__).resolve().parent
BACKEND = HERE / 'ordered_backend'
sys.path.insert(0, str(HERE))
if not (HERE/'cnn_softmax_runner.py').exists():
    sys.path.insert(0,'/root/ordered')
import cnn_softmax_runner as native

SOURCE_NAMES = ['cnn_ordered_modal.py', 'cnn_softmax_runner.py',
                'ordered_backend/ops.py', 'ordered_backend/network.py',
                'ordered_backend/schedule.py']
image = modal.Image.from_registry(native.IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6')
for name in SOURCE_NAMES:
    image = image.add_local_file(str(HERE / name), remote_path='/root/ordered/' + name)
app = modal.App('sutro-medium98-ordered-cnn-validation')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def archive_bytes(arrays):
    import numpy as np
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def validation_metrics(scores, labels):
    """Host-only checkpoint diagnostic; never part of the update direction."""
    import numpy as np
    p = np.clip(scores - scores.max(axis=1, keepdims=True), -16., 0.)
    p = p * np.float32(1 / 1024) + np.float32(1)
    for _ in range(10):
        p = p * p
    denominator = np.zeros((len(p), 1), np.float32)
    for digit in range(10):
        denominator = denominator + p[:, digit:digit + 1]
    p = p / denominator
    error = p.astype(np.float64) - np.eye(10, dtype=np.float64)[labels]
    return {'correct': int((scores.argmax(1) == labels).sum()),
            'total': int(len(labels)),
            'half_brier': float(np.square(error).sum() * .5 / len(labels))}


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=16384, timeout=14400,
              min_containers=0, max_containers=3, buffer_containers=0,
              scaledown_window=2, retries=0)
def run(payload, split, config, seed, provenance):
    import os
    import platform
    import time
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    sys.path.insert(0, '/root/ordered/ordered_backend')
    from network import Trainer
    from schedule import array_hash
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, digest in provenance['source_sha256'].items():
        assert sha(Path('/root/ordered') / name) == digest
    assert set(payload) == {'train_images', 'train_labels'}
    arrays = {name: np.frombuffer(value['bytes'], dtype=value['dtype'])
              .reshape(value['shape']).copy() for name, value in payload.items()}
    for name, value in arrays.items():
        assert array_hash(value) == provenance['input_sha256'][name]
    fit = np.asarray(split['fit_positions'])
    val = np.asarray(split['validation_positions'])
    trainer = Trainer(np.ascontiguousarray(arrays['train_images'][fit]),
                      np.ascontiguousarray(arrays['train_labels'][fit]), config, seed)
    initial = archive_bytes(trainer.initial_arrays)
    started = time.perf_counter()
    trainer.prepare()
    prepare_seconds = time.perf_counter() - started
    print(f"{config['id']} seed{seed}: prepared in {prepare_seconds:.1f}s", flush=True)
    queries = torch.from_numpy(np.ascontiguousarray(arrays['train_images'][val])).cuda()
    labels = arrays['train_labels'][val]
    trainer.initialize()
    history, all_logits, best, best_logits, best_parameters = [], [], None, None, None
    status = 'complete'
    for epoch in range(1, config['epochs'] + 1):
        torch.cuda.synchronize()
        before = time.perf_counter()
        trainer.train_epoch(epoch)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - before
        before = time.perf_counter()
        scores = trainer.network.infer(queries)
        validation_seconds = time.perf_counter() - before
        if not np.isfinite(scores).all():
            status = 'nonfinite'
            print(f"{config['id']} seed{seed} nonfinite epoch{epoch}", flush=True)
            break
        metrics = validation_metrics(scores, labels)
        row = {'epoch': epoch, 'learning_rate': trainer.rates[epoch],
               'validation': metrics, 'training_seconds': training_seconds,
               'validation_seconds': validation_seconds,
               'scores_sha256': array_hash(scores)}
        history.append(row)
        all_logits.append(scores.copy())
        if best is None or (-metrics['correct'], metrics['half_brier'], epoch) < (
                -best['validation']['correct'], best['validation']['half_brier'], best['epoch']):
            best = row.copy()
            best_logits = scores.copy()
            state = trainer.network.state()
            best_parameters = archive_bytes(state)
            best['parameter_sha256'] = {k: array_hash(v) for k, v in state.items()}
        if epoch <= 3 or epoch % 10 == 0:
            print(f"{config['id']} seed{seed} epoch{epoch}: {metrics['correct']}/1200; "
                  f"best {best['validation']['correct']}/1200; epoch {training_seconds:.2f}s", flush=True)
    result = {'id': config['id'] + f'-s{seed}', 'config': config, 'seed': seed,
              'status': status, 'provenance': provenance, 'history': history,
              'best': best, 'schedule_manifests': trainer.schedule_manifests,
              'initial_parameter_sha256': {k: array_hash(v) for k, v in trainer.initial_arrays.items()},
              'initial_archive_sha256': hashlib.sha256(initial).hexdigest(),
              'parameter_count': sum(v.size for v in trainer.initial_arrays.values()),
              'prepare_seconds': prepare_seconds, 'test_arrays_accessed': False,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(0)},
              'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    if best_logits is not None:
        result['validation_logits_sha256'] = array_hash(best_logits)
        result['best_archive_sha256'] = hashlib.sha256(best_parameters).hexdigest()
    epoch_scores = np.stack(all_logits) if all_logits else np.empty((0,1200,10),np.float32)
    epoch_archive = archive_bytes({'scores': epoch_scores})
    result['epoch_scores_sha256'] = array_hash(epoch_scores)
    result['epoch_archive_sha256'] = hashlib.sha256(epoch_archive).hexdigest()
    return result, initial, best_parameters, None if best_logits is None else best_logits.tobytes(), epoch_archive


@app.local_entrypoint()
def main(phase: str='prepare', data: str='', output: str='', family: str='', screen: str='dropout'):
    import numpy as np
    root = Path(output) if output else HERE
    root.mkdir(parents=True, exist_ok=True)
    canonical = json.loads((HERE.parents[1] / 'doc/dataset_manifest.json').read_text())
    images, labels = native.load_training(data, canonical)
    sources = {name: sha(HERE / name) for name in SOURCE_NAMES}
    if phase == 'prepare':
        assert not (root / 'cnn_ordered_protocol.json').exists(), 'Preserve existing frozen protocol'
        assert screen in ('softmax', 'bn', 'dropout', 'adam')
        native_summary_path = root / f'cnn_{screen}_summary.json'
        native_summary = json.loads(native_summary_path.read_text())
        assert native_summary['phase'] == 'replicate'
        candidates = [row['config'] for row in native_summary['ensemble_validation']
                      if row['config']['id'] == family]
        assert len(candidates) == 1
        split = json.loads((root / f'cnn_{screen}_split.json').read_text())
        write(root / 'cnn_ordered_split.json', split)
        protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                    'attempt_target_percent':97,'published_requirement_percent':98,
                    'purpose': 'Training-only validation of exact ordered-FP32 CNN, including new fixed seed-only schedules',
                    'source_sha256': sources, 'config': candidates[0], 'seeds': [11, 22, 33],
                    'input_allowlist': ['train_images', 'train_labels'],
                    'input_sha256': {'train_images': native.array_hash(images), 'train_labels': native.array_hash(labels)},
                    'split_sha256': sha(root / 'cnn_ordered_split.json'),
                    'native_screen_file': native_summary_path.name,
                    'native_screen_sha256': sha(native_summary_path),
                    'checkpoint_rule': 'Maximum validation correct, minimum FP64-reduced half-Brier diagnostic from ordered-FP32 probabilities, earliest epoch',
                    'ensemble_rule': 'Ordered FP32 raw-logit sum starting positive zero for seeds11,22,33 at each common epoch; argmax, no division. Select a common epoch by highest ensemble validation correct, lowest half-Brier diagnostic, earliest epoch. Individual best checkpoints are diagnostics only.',
                    'schedule': 'PCG64 seed-only schedules, explicit four-neighbor bilinear arithmetic and scalar-ordered separate FP32 operations; differs from preliminary native CUDA schedule/reductions',
                    'test_gate': 'No test arrays accepted; freeze formal learner after this stage before preparing eleven new draws',
                    'container_image': native.IMAGE_REF, 'test_arrays_accessed': False}
        write(root / 'cnn_ordered_protocol.json', protocol)
        print('Frozen one-family, three-seed ordered training-only protocol.', flush=True)
        return
    assert phase == 'run'
    protocol = json.loads((root / 'cnn_ordered_protocol.json').read_text())
    split = json.loads((root / 'cnn_ordered_split.json').read_text())
    assert sources == protocol['source_sha256']
    assert sha(root / 'cnn_ordered_split.json') == protocol['split_sha256']
    for name, value in [('train_images', images), ('train_labels', labels)]:
        assert native.array_hash(value) == protocol['input_sha256'][name]
    provenance = {k: protocol[k] for k in ('source_sha256', 'input_sha256', 'split_sha256')}
    provenance['protocol_sha256'] = sha(root / 'cnn_ordered_protocol.json')
    payload = {name: {'shape': value.shape, 'dtype': str(value.dtype), 'bytes': value.tobytes()}
               for name, value in [('train_images', images), ('train_labels', labels)]}
    for name in ('cnn_ordered_results', 'cnn_ordered_logits', 'cnn_ordered_parameters'):
        (root / name).mkdir(exist_ok=True)
    assert not list((root / 'cnn_ordered_results').glob('*.json')), 'Run once into an empty result directory'
    jobs = [(payload, split, protocol['config'], seed, provenance) for seed in protocol['seeds']]
    rows = []
    for row, initial, parameters, raw, epoch_archive in run.starmap(jobs, order_outputs=False):
        assert row['provenance'] == provenance
        prefix = row['id']
        assert hashlib.sha256(initial).hexdigest() == row['initial_archive_sha256']
        row['initial_file'] = f'cnn_ordered_parameters/{prefix}-initial.npz'
        (root / row['initial_file']).write_bytes(initial)
        assert hashlib.sha256(epoch_archive).hexdigest() == row['epoch_archive_sha256']
        with np.load(io.BytesIO(epoch_archive),allow_pickle=False) as archive:
            epoch_scores = archive['scores']
        assert native.array_hash(epoch_scores) == row['epoch_scores_sha256']
        assert len(epoch_scores) == len(row['history'])
        for values, epoch_row in zip(epoch_scores,row['history']):
            assert native.array_hash(values) == epoch_row['scores_sha256']
        row['epoch_scores_file'] = f'cnn_ordered_logits/{prefix}-all-epochs.npz'
        (root / row['epoch_scores_file']).write_bytes(epoch_archive)
        if raw is not None:
            logits = np.frombuffer(raw, dtype=np.float32).reshape(1200, 10)
            assert native.array_hash(logits) == row['validation_logits_sha256']
            assert hashlib.sha256(parameters).hexdigest() == row['best_archive_sha256']
            row['validation_logits_file'] = f'cnn_ordered_logits/{prefix}.npy'
            row['best_parameters_file'] = f'cnn_ordered_parameters/{prefix}-best.npz'
            np.save(root / row['validation_logits_file'], logits)
            (root / row['best_parameters_file']).write_bytes(parameters)
        write(root / 'cnn_ordered_results' / (prefix + '.json'), row)
        rows.append(row)
    rows.sort(key=lambda row: row['seed'])
    summary = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'provenance': provenance, 'config': protocol['config'],
               'members': [{'seed': row['seed'], 'best': row['best'], 'status': row['status']} for row in rows],
               'test_arrays_accessed': False}
    if all(row['status'] == 'complete' for row in rows):
        logits = []
        for row in rows:
            with np.load(root / row['epoch_scores_file'],allow_pickle=False) as archive:
                logits.append(archive['scores'])
        ensemble_all = np.zeros_like(logits[0])
        for values in logits:
            ensemble_all = ensemble_all + values
        truth = labels[np.asarray(split['validation_positions'])]
        summary['ensemble_history'] = [{'epoch':epoch+1,**validation_metrics(values,truth)}
                                       for epoch,values in enumerate(ensemble_all)]
        selected = min(summary['ensemble_history'],key=lambda row:(-row['correct'],row['half_brier'],row['epoch']))
        summary['selected_common_epoch'] = selected['epoch']
        ensemble = ensemble_all[selected['epoch']-1]
        path = root / 'cnn_ordered_logits/ensemble.npy'
        np.save(path, ensemble)
        summary['ensemble'] = selected
        summary['ensemble_scores_sha256'] = native.array_hash(ensemble)
    write(root / 'cnn_ordered_summary.json', summary)
    print(json.dumps(summary), flush=True)
