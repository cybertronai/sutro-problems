#!/usr/bin/env python3
"""Verify and benchmark the frozen PCA-QDA MNIST-medium GPU implementation.

Each CUDA graph replay receives 10,000 normalized training images and 10,000
queries already on the device, then computes Newton-Raphson square-root
features, the 40-dimensional basis, projections, class covariances, inverses,
and all predictions. No learned state is carried between replays. Transfers,
area resizing, allocation, JIT and graph capture are outside the measured scope.

The GPU uses float32 matrix products, raw-moment covariance accumulation,
classical Gram-Schmidt, and torch.log. The ordered CPU reference uses raw
moments, sequential Gram-Schmidt updates and a lowered polynomial logarithm.
All frozen prediction disagreements are reported; bitwise intermediate equality
is not claimed. Verification reuses one graph across all eleven datasets and a
training-label perturbation, poisoning retained state before every replay.

    python gpu_benchmark.py generated/payloads results/gpu_verification_benchmark.json 3000 5

The benchmark takes five blocks of 3,000 complete graph replays with 5-second
idle windows before and after each block. NVML total-energy subtraction is
signed; all rounds, including any negative estimates, are retained. A single
historical .npz payload is also supported; --verify-only skips the benchmark.
"""
import hashlib, json, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import initial_basis        # the fixed literal start W0 (numpy PCG64 seed 0)


def gauss_jordan(S):
    """Batched inverse of SPD (C,K,K) matrices without pivoting, as in reference.py; returns (P, log det)."""
    n = S.shape[1]; A = S.clone(); P = torch.eye(n, device=S.device, dtype=S.dtype).expand(S.shape[0], n, n).clone()
    logdet = torch.zeros(S.shape[0], device=S.device, dtype=S.dtype)
    for p in range(n):
        piv = A[:, p, p].clone(); logdet = logdet + torch.log(piv)
        A[:, p, :] = A[:, p, :] / piv[:, None]; P[:, p, :] = P[:, p, :] / piv[:, None]
        fct = A[:, :, p].clone(); fct[:, p] = 0
        A = A - fct[:, :, None] * A[:, p, :][:, None, :]; P = P - fct[:, :, None] * P[:, p, :][:, None, :]
    return P, logdet


def nr_sqrt(x):
    y = x + TINY
    for _ in range(NR_ITERATIONS): y = 0.5 * (y + x / y)
    return y


def basis_gpu(u, W0):
    """m and W by ROUNDS rounds of subspace iteration with column-wise Gram-Schmidt (classical form, vectorized
    over the previous columns) and max-|entry| scaling; same subspace construction as reference.py, not bitwise."""
    up = u[:NP]; m = up.mean(0); d = up - m; S = d.T @ d / NP; W = W0
    for _ in range(ROUNDS):
        V = S @ W
        den = torch.zeros(K, device=u.device, dtype=u.dtype)
        for j in range(K):
            if j:
                f = (V[:, :j].T @ V[:, j]) / den[:j]
                V[:, j] = V[:, j] - V[:, :j] @ f
            den[j] = V[:, j] @ V[:, j]
        W = V / V.abs().max(0).values
    return m, W


def pca_qda_gpu(x, y, q, W0, scratch=None):
    """x (N,81) float32 cuda (pixels/255), y (N,) int64 cuda, q (Q,81).  Returns labels (Q,) int64."""
    N = x.shape[0]
    u = nr_sqrt(x); m, W = basis_gpu(u, W0); z = (u - m) @ W; zq = (nr_sqrt(q) - m) @ W
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)          # (N, C)
    count = onehot.sum(0)                                                   # (C,)
    mu = (onehot.T @ z) / count[:, None]                                    # (C, K)
    zz = z[:, :, None] * z[:, None, :]                                      # (N, K, K)
    mom = torch.einsum('nc,nij->cij', onehot, zz) / count[:, None, None]    # (C, K, K)
    S = mom - mu[:, :, None] * mu[:, None, :]
    tr = torch.diagonal(S, dim1=1, dim2=2).sum(1)
    S = (1 - SHRINK) * S + (SHRINK * tr / K)[:, None, None] * torch.eye(K, device=S.device)
    P, logdet = gauss_jordan(S)                                             # capturable, no cuSOLVER
    kappa = torch.log(count / N) - 0.5 * logdet                             # (C,)
    d = zq[:, None, :] - mu[None, :, :]                                     # (Q, C, K)
    s = torch.einsum('qci,cij,qcj->qc', d, P, d)
    scores = kappa[None, :] - 0.5 * s
    if scratch is not None:
        # Retain already-live tensors for poisoning outside graph capture.
        # This adds no GPU operations to the captured learning task.
        scratch.update(u=u, m=m, W=W, z=z, zq=zq, onehot=onehot, count=count,
                       mu=mu, zz=zz, mom=mom, S=S, P=P, logdet=logdet,
                       kappa=kappa, d=d, s=s, scores=scores)
    return scores.argmax(1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payloads(path):
    """Check the exported manifest before loading any GPU inputs."""
    if path.is_file():
        records = [{'path': path.name, 'sha256': sha256(path)}]
        base, provenance = path.parent, {}
    else:
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        records, base = manifest['draws'], path
        if len(records) != 11 or [r['draw'] for r in records] != list(range(11)):
            raise ValueError('the frozen submission requires exactly draws 0 through 10')
        if manifest['seeds'] != list(range(20261501, 20261512)):
            raise ValueError('unexpected frozen dataset seeds')
        provenance = {'payload_manifest_sha256': sha256(manifest_path),
                      'payload_manifest': manifest}
    payloads = []
    for record in records:
        payload_path = base / record['path']
        if not payload_path.resolve().is_relative_to(base.resolve()):
            raise ValueError('payload path leaves manifest directory')
        if sha256(payload_path) != record['sha256']:
            raise ValueError(f'payload hash mismatch: {payload_path.name}')
        with np.load(payload_path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        for name, shape in [('x', (10000, 81)), ('q', (10000, 81)),
                            ('labels', (10000,)), ('test_labels', (10000,)),
                            ('frozen_predictions', (10000,)), ('dataset_seed', ())]:
            value = payload[name]
            if value.shape != shape:
                raise ValueError(f'{payload_path.name}: wrong {name} shape')
            if name in ('x', 'q'):
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'{payload_path.name}: {name} must be finite float32')
            elif value.dtype.kind not in 'iu':
                raise ValueError(f'{payload_path.name}: {name} must be integral')
            elif name != 'dataset_seed' and not ((value >= 0) & (value < C)).all():
                raise ValueError(f'{payload_path.name}: invalid {name}')
        if 'dataset_seed' in record and int(payload['dataset_seed']) != record['dataset_seed']:
            raise ValueError('payload seed differs from manifest')
        if 'draw' in record and int(payload['dataset_seed']) != 20261501 + record['draw']:
            raise ValueError('payload seed is not the frozen seed for its draw')
        for name, expected in record.get('arrays', {}).items():
            value = payload[name]
            canonical = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<'), copy=False))
            actual = hashlib.sha256(canonical.tobytes(order='C')).hexdigest()
            if (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']
                    or actual != expected['sha256']):
                raise ValueError(f'{payload_path.name}: {name} differs from its array manifest')
        payloads.append((record, payload))
    return payloads, provenance


def main():
    import argparse
    import importlib.metadata
    import os
    import threading
    import pynvml as nv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('payload', type=Path, help='payload .npz or all-draw payload directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('replays', type=int, nargs='?', default=3000)
    parser.add_argument('rounds', type=int, nargs='?', default=5)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.replays <= 0 or args.rounds <= 0:
        parser.error('replays and rounds must be positive')
    payloads, provenance = load_payloads(args.payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nv.nvmlInit()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_uuid = str(properties.uuid)
    if not device_uuid.startswith(('GPU-', 'MIG-')):
        device_uuid = 'GPU-' + device_uuid
    handle = nv.nvmlDeviceGetHandleByUUID(device_uuid)

    def other_compute_processes():
        return [{'pid': item.pid, 'used_gpu_memory_bytes': item.usedGpuMemory}
                for item in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                if item.pid != os.getpid()]

    initial_processes = other_compute_processes()
    if initial_processes:
        nv.nvmlShutdown()
        raise SystemExit('GPU is in use by other compute processes; retry when idle: '
                         + json.dumps(initial_processes))
    # Match the original torch 2.5.1 float32 matrix-product default explicitly.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.empty((10000, D), device='cuda', dtype=torch.float32)
    y = torch.empty(10000, device='cuda', dtype=torch.int64)
    q = torch.empty((10000, D), device='cuda', dtype=torch.float32)
    initial = initial_basis()
    W0 = torch.tensor(initial, device='cuda')
    result = torch.empty(10000, device='cuda', dtype=torch.int64)

    def load_device(payload):
        x.copy_(torch.from_numpy(payload['x']))
        y.copy_(torch.from_numpy(payload['labels'].astype(np.int64)))
        q.copy_(torch.from_numpy(payload['q']))
        W0.copy_(torch.from_numpy(initial))

    load_device(payloads[0][1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pca_qda_gpu(x, y, q, W0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    scratch = {}
    with torch.cuda.graph(graph, stream=stream):
        result.copy_(pca_qda_gpu(x, y, q, W0, scratch=scratch))
    torch.cuda.synchronize()

    def poison_scratch():
        result.fill_(-1)
        for value in scratch.values():
            value.fill_(float('nan'))

    def replay_and_check(eager):
        poison_scratch()
        graph.replay()
        replay = result.cpu().numpy().copy()
        finite = all(bool(torch.isfinite(scratch[name]).all())
                     for name in ('W', 'mu', 'S', 'P', 'logdet', 'kappa', 'scores'))
        initial_unchanged = np.array_equal(W0.cpu().numpy(), initial)
        return bool(np.array_equal(replay, eager)), finite, initial_unchanged

    draws = []
    first_predictions = None
    for index, (record, payload) in enumerate(payloads):
        load_device(payload)
        poison_scratch()
        eager = pca_qda_gpu(x, y, q, W0).cpu().numpy().copy()
        if first_predictions is None:
            first_predictions = eager.copy()
        checks = [replay_and_check(eager) for _ in range(3)]
        frozen = payload['frozen_predictions']
        mismatch = np.flatnonzero(eager != frozen)
        draw = {'draw': record.get('draw', index), 'dataset_seed': int(payload['dataset_seed']),
                'payload_sha256': record['sha256'],
                'gpu_agrees_with_frozen_cpu_predictions': int((eager == frozen).sum()),
                'total': 10000, 'gpu_correct': int((eager == payload['test_labels']).sum()),
                'cpu_correct': int((frozen == payload['test_labels']).sum()),
                'gpu_predictions_int64_sha256': hashlib.sha256(eager.astype('<i8').tobytes()).hexdigest(),
                'mismatches': [{'index': int(i), 'gpu': int(eager[i]), 'cpu': int(frozen[i]),
                                'true_label': int(payload['test_labels'][i])} for i in mismatch],
                'graph_replays_equal_eager': [c[0] for c in checks],
                'graph_model_and_scores_finite': [c[1] for c in checks],
                'initial_basis_unchanged': [c[2] for c in checks],
                'scratch_poisoned_before_each_replay': sorted(scratch)}
        draws.append(draw)
        print(json.dumps({'verification': draw}), flush=True)

    # A training-label perturbation with unchanged images must be reflected by
    # this same graph. This catches accidentally cached learned parameters.
    load_device(payloads[0][1])
    y.copy_(torch.from_numpy((payloads[0][1]['labels'].astype(np.int64) + 1) % C))
    rotated_eager = pca_qda_gpu(x, y, q, W0).cpu().numpy().copy()
    graph_equal, finite, basis_unchanged = replay_and_check(rotated_eager)
    perturbation = {'training_labels_rotated_modulo': C,
                    'graph_matches_eager': graph_equal, 'model_and_scores_finite': finite,
                    'initial_basis_unchanged': basis_unchanged,
                    'predictions_changed': int((rotated_eager != first_predictions).sum()),
                    'predictions_equal_rotated_original_labels': int((rotated_eager == (first_predictions + 1) % C).sum()),
                    'total': 10000}
    validation = {'draws': draws, 'total': sum(r['total'] for r in draws),
                  'gpu_agrees_with_frozen_cpu_predictions': sum(r['gpu_agrees_with_frozen_cpu_predictions'] for r in draws),
                  'gpu_correct': sum(r['gpu_correct'] for r in draws),
                  'cpu_correct': sum(r['cpu_correct'] for r in draws),
                  'all_graph_replays_equal_eager': all(all(r['graph_replays_equal_eager']) for r in draws),
                  'all_graph_model_and_scores_finite': all(all(r['graph_model_and_scores_finite']) for r in draws),
                  'all_initial_basis_checks_passed': all(all(r['initial_basis_unchanged']) for r in draws),
                  'intermediate_float_values_bitwise_equal_to_reference': False,
                  'all_eleven_frozen_draws_checked': len(draws) == 11,
                  'one_captured_graph_reused_across_changed_inputs': True,
                  'training_label_perturbation': perturbation,
                  'numerical_scope': 'GPU matrix products change reduction order for raw-moment covariance, Gram-Schmidt projects onto prior columns simultaneously, and logarithms use torch.log. CPU uses ordered raw moments, sequential Gram-Schmidt updates, and a lowered polynomial logarithm. Frozen-label disagreements are counted explicitly.'}
    validation['frozen_label_agreement_exact'] = validation['gpu_agrees_with_frozen_cpu_predictions'] == validation['total']
    validation['graph_verification_passed'] = (validation['all_graph_replays_equal_eager']
        and validation['all_graph_model_and_scores_finite'] and validation['all_initial_basis_checks_passed']
        and graph_equal and finite and basis_unchanged and perturbation['predictions_changed'] > 0)
    def nvtext(value):
        return value.decode() if isinstance(value, bytes) else value

    provenance.update({'runner_sha256': sha256(__file__),
                       'reference_sha256': sha256(Path(__file__).with_name('reference.py')),
                       'initial_basis_float32_sha256': hashlib.sha256(initial.astype('<f4').tobytes()).hexdigest(),
                       'benchmark_dataset_seed': int(payloads[0][1]['dataset_seed']),
                       'benchmark_payload_sha256': payloads[0][0]['sha256']})
    doc = {'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'hardware': {'name': nvtext(nv.nvmlDeviceGetName(handle)),
                        'uuid': nvtext(nv.nvmlDeviceGetUUID(handle)),
                        'memory_total_bytes': int(properties.total_memory),
                        'multiprocessors': properties.multi_processor_count,
                        'compute_capability': [properties.major, properties.minor]},
           'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                        'nvidia-ml-py': importlib.metadata.version('nvidia-ml-py'),
                        'cuda': torch.version.cuda,
                        'driver': nvtext(nv.nvmlSystemGetDriverVersion()),
                        'python': platform.python_version(), 'platform': platform.platform()},
           'provenance': provenance, 'validation': validation,
           'numeric_configuration': {'float32_matmul_precision': torch.get_float32_matmul_precision(),
                                     'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                     'classes': C, 'input_features': D, 'basis_dimensions': K,
                                     'basis_samples': NP, 'subspace_rounds': ROUNDS,
                                     'sqrt_iterations': NR_ITERATIONS, 'tiny': TINY, 'shrinkage': SHRINK},
           'scope': 'CUDA-graph replays of complete feature transform, basis fitting, projection, QDA training and 10000 predictions on device-resident normalized float32 inputs; includes host-dispatch gaps between graph replays; excludes transfers, area-resize preprocessing, allocation, JIT and capture. Signed NVML total-energy deltas minus mean paired idle power times measured wall duration; negative estimates are retained.'}
    verification_path = args.output if args.verify_only else args.output.parent / 'gpu_verification.json'
    verification_path.write_text(json.dumps(doc, indent=2) + '\n')
    if not validation['graph_verification_passed']:
        nv.nvmlShutdown()
        raise SystemExit(f'GPU graph verification failed; see {verification_path}')
    if args.verify_only:
        nv.nvmlShutdown()
        return

    load_device(payloads[0][1])
    poison_scratch()
    graph.replay()
    torch.cuda.synchronize()

    def stamp():
        before = time.perf_counter()
        energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
        after = time.perf_counter()
        return {'energy_mj': energy, 'time_s': (before + after) / 2,
                'read_duration_s': after - before}

    def interval(start, end):
        seconds = end['time_s'] - start['time_s']
        energy_mj = end['energy_mj'] - start['energy_mj']
        if seconds <= 0 or energy_mj < 0:
            raise RuntimeError('non-monotonic time or NVML energy counter')
        return seconds, energy_mj

    def idle():
        torch.cuda.synchronize()
        start = stamp()
        time.sleep(5.0)
        end = stamp()
        seconds, energy_mj = interval(start, end)
        return {'start': start, 'end': end, 'power_w': energy_mj / 1000 / seconds}

    # A workload can start after the initial idle check. Monitor throughout
    # timing, retain contaminated measurements, and require a complete clean run.
    monitor_stop = threading.Event()
    process_samples, monitor_errors = [], []

    def sample_processes():
        try:
            process_samples.append({'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                    'other_compute_processes': other_compute_processes()})
        except Exception as error:
            monitor_errors.append(str(error))

    def monitor_processes():
        while not monitor_stop.wait(1.0):
            sample_processes()

    def interference_detected():
        return bool(monitor_errors or any(row['other_compute_processes'] for row in process_samples))

    sample_processes()
    monitor = threading.Thread(target=monitor_processes, daemon=True)
    monitor.start()
    rounds = []
    try:
        for index in range(args.rounds):
            if interference_detected():
                break
            before = idle()
            sample_processes()
            if interference_detected():
                break
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start = stamp()
            start_event.record()
            for _ in range(args.replays):
                graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            end = stamp()
            after = idle()
            seconds, gross_mj = interval(start, end)
            idle_w = (before['power_w'] + after['power_w']) / 2
            row = {'round': index, 'repeats': args.replays,
                   'cuda_ms': start_event.elapsed_time(end_event) / args.replays,
                   'wall_ms': seconds * 1000 / args.replays,
                   'gross_mj': gross_mj / args.replays,
                   'adjusted_mj': (gross_mj - idle_w * seconds * 1000) / args.replays,
                   'idle_before_w': before['power_w'], 'idle_after_w': after['power_w'],
                   'average_power_w': gross_mj / 1000 / seconds,
                   'measurement_start': start, 'measurement_end': end,
                   'idle_before': before, 'idle_after': after}
            rounds.append(row)
            print(json.dumps(row), flush=True)
            sample_processes()
            if interference_detected():
                break
    finally:
        monitor_stop.set()
        monitor.join()
        nv.nvmlShutdown()
    doc['concurrency_monitor'] = {'interval_s': 1.0, 'sample_count': len(process_samples),
                                  'interference_observations': [row for row in process_samples if row['other_compute_processes']],
                                  'errors': monitor_errors}
    doc['measurement_valid'] = len(rounds) == args.rounds and not interference_detected()
    doc['requested_rounds'] = args.rounds
    if not rounds:
        doc.update({'rounds': [], 'summary': None,
                    'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        args.output.write_text(json.dumps(doc, indent=2) + '\n')
        raise SystemExit('No uncontended benchmark rounds completed; see concurrency_monitor')
    adjusted = [r['adjusted_mj'] for r in rounds]
    summary = {'cuda_ms_median': statistics.median(r['cuda_ms'] for r in rounds),
               'wall_ms_median': statistics.median(r['wall_ms'] for r in rounds),
               'gross_mj_median': statistics.median(r['gross_mj'] for r in rounds),
               'adjusted_mj_median': statistics.median(adjusted),
               'adjusted_mj_min': min(adjusted), 'adjusted_mj_max': max(adjusted),
               'adjusted_mj_sample_sd': statistics.stdev(adjusted) if len(adjusted) > 1 else None,
               'negative_adjusted_rounds': sum(value < 0 for value in adjusted),
               'energy_interpretation': 'Idle subtraction measures a small difference between much larger totals. Round spread and idle drift limit precision; this is not a statistical confidence interval.'}
    doc.update({'rounds': rounds, 'summary': summary,
                'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    args.output.write_text(json.dumps(doc, indent=2) + '\n')
    if not doc['measurement_valid']:
        raise SystemExit('Benchmark interrupted by other GPU work; partial measurements retained, repeat the complete run when idle')
    print(json.dumps({'summary': summary,
                      'validation': {k: v for k, v in validation.items() if k != 'draws'},
                      'hardware': doc['hardware']}, indent=2))


if __name__ == '__main__':
    main()
