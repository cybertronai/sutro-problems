"""KernelBot evaluator for MNIST energy leaderboards (prototype).

Modes (run by KernelBot as separate invocations):
  test         one draw; output format and a loose accuracy sanity check
  benchmark    accuracy over `draws` secret-seeded draws against the band's
               exact threshold; reports per-call time
  leaderboard  the same accuracy gate on fresh draws, then the energy protocol
               (mnist/code/energy.py); the ranked score is net nanojoules per
               complete training-and-prediction call

Trust boundary: this process never imports the submission or touches CUDA. It
keeps test labels, draw seeds and each draw's secret label permutation, reads
NVML, and freezes the submission's worker process (SIGSTOP) whenever idle,
reference or control energy is being measured.

Case keys (integers, from task.yml tests/benchmarks): size, train, test,
error_bp (maximum mean error in basis points), draws, seed; energy overrides
rounds, active_s, idle_s, settle_s, reference_s, staged.
"""
import base64
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import sys
import time
import zlib

import numpy as np

import energy
import mnist_data
import nvml
import worker

EXIT_VALIDATE_FAIL = 112
EXIT_INFRASTRUCTURE = 120  # telemetry/host problem: resubmit, not the submitter's fault
SANITY_SLACK = 0.03
# KernelBot reads the result pipe only after eval.py exits, so everything logged must
# fit in the pipe buffer (64 KiB on Linux) or the write blocks forever.
MAX_LOGGED_RECORD_BYTES = 32_000
CACHE = Path('.mnist-cache')


class PopcornOutput:
    def __init__(self, fd):
        self.file = os.fdopen(fd, 'w')
        os.set_inheritable(fd, False)

    def log(self, key, value):
        print(f'{key}: {value}', file=self.file, flush=True)

    def close(self):
        self.file.close()


def debug(*parts):
    print(*parts, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- cases and seeds

def combine(a, b):
    """KernelBot's secret-seed combination (Cantor pairing), as in its example eval.py."""
    return int(a + (a + b) * (a + b + 1) // 2)


def read_cases(path, secret):
    cases = []
    for line in Path(path).read_text().splitlines():
        case = {}
        for part in line.split(';'):
            match = re.fullmatch(r'\s*([a-zA-Z_]+):\s*([+-]?[0-9]+)\s*', part)
            if not match:
                raise ValueError(f'invalid case {line!r}')
            case[match[1]] = int(match[2])
        if secret is not None:
            case['seed'] = combine(case['seed'], secret)
        case['spec'] = line
        cases.append(case)
    return cases


# ---------------------------------------------------------------- data

def load_pool(size):
    """Official MNIST training split, box-area resized to size x size, pixels in [0, 1]."""
    path = CACHE / f'pool-{size}.npz'
    if path.exists():
        with np.load(path) as archive:
            return archive['images'], archive['labels']
    raw = CACHE / 'raw'
    images_file, images_md5 = mnist_data.SOURCES['train_images']
    labels_file, labels_md5 = mnist_data.SOURCES['train_labels']
    images = mnist_data.read_idx(mnist_data.download_source(raw, images_file, images_md5), 60000, images=True)
    labels = mnist_data.read_idx(mnist_data.download_source(raw, labels_file, labels_md5), 60000, images=False)
    pixels = (mnist_data.area_resize(images, size) / np.float32(255))[:, None, :, :].astype(np.float32)
    np.savez(path, images=pixels, labels=labels)
    return pixels, labels


def make_draw(pool, seed, n_train, n_test):
    """Same split as the repository's medium-error-targets-v1 profile, plus a secret label permutation.

    Returns the arrays the learner may see and the permuted test labels the parent keeps.
    """
    images, labels = pool
    order = mnist_data.source_permutations(seed=seed)[0]
    train, test = order[:n_train], order[n_train:n_train + n_test]
    relabel = np.random.default_rng([seed, 0x5eed]).permutation(10)
    visible = (images[train], relabel[labels[train]].astype(np.int64), images[test])
    return visible, relabel[labels[test]].astype(np.int64)


def required_correct(total, error_bp):
    return math.ceil(total * (10000 - error_bp) / 10000)


# ---------------------------------------------------------------- workers

class WorkerHandle:
    def __init__(self, role):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker.main, args=(child, role, os.getcwd()), daemon=True)
        self.process.start()
        child.close()  # otherwise a dead worker never produces EOF and recv() hangs
        status, value = self.connection.recv()
        if status != 'ready':
            raise RuntimeError(f'{role} worker failed to start: {value}')
        self.role, self.frozen = role, False

    def call(self, command, *args):
        if self.frozen:
            raise RuntimeError(f'{self.role} worker is frozen')
        self.connection.send((command, args))
        status, value = self.connection.recv()
        if status != 'ok':
            raise RuntimeError(f'{self.role} worker {command} failed: {value}')
        return value

    def freeze(self):
        if not self.frozen:
            os.kill(self.process.pid, signal.SIGSTOP)
            self.frozen = True

    def thaw(self):
        if self.frozen:
            os.kill(self.process.pid, signal.SIGCONT)
            self.frozen = False

    def stop(self):
        self.thaw()
        if self.process.is_alive():
            try:
                self.connection.send(('stop', ()))
                self.connection.recv()
            except (EOFError, OSError):
                pass
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()


def score(handle, draws):
    """Warm up on the first draw, then predict every draw; returns per-draw counts and call times."""
    handle.call('predict', draws[0][0])
    rows, times = [], []
    for visible, truth in draws:
        predictions, seconds = handle.call('predict', visible)
        if predictions.shape != truth.shape:
            raise ValueError(f'predictions shape {predictions.shape}, expected {truth.shape}')
        rows.append(int((predictions == truth).sum()))
        times.append(seconds)
    return rows, times


# ---------------------------------------------------------------- energy adapters

class Telemetry:
    extra_columns = ['temperature_c', 'pstate', 'gpu_util_percent']

    def __init__(self, device, max_processes):
        self.nvml = device
        self.max_processes = max_processes

    def energy_mj(self):
        return self.nvml.energy_mj()

    def foreign_processes(self):
        return nvml.excess_processes(self.nvml, self.max_processes)

    def sensor(self):
        return nvml.Sensor(self.max_processes)

    def describe(self):
        return {**self.nvml.describe(), 'max_compute_processes': self.max_processes}


class Counted:
    """Adapts a worker command to the protocol's run()/synchronize() workload interface."""

    def __init__(self, handle, command):
        self.handle, self.command, self.pending = handle, command, 0

    def run(self):
        self.pending += 1

    def synchronize(self):
        if self.pending:
            count, self.pending = self.pending, 0
            self.handle.call(self.command, count)


class SubmissionTask(Counted):
    name = 'submission'
    scope = ('One complete training-and-prediction call on device-resident inputs, cycling staged draws. '
             'Staging copies are measured by the control window and subtracted.')

    def __init__(self, handle, pool, case, staged_truth, error_bp):
        super().__init__(handle, 'run')
        self.pool, self.case, self.staged_truth, self.error_bp = pool, case, staged_truth, error_bp
        self.fresh_seed = case['seed'] + 3000
        self.ran = 0

    def synchronize(self):
        self.ran += self.pending
        super().synchronize()

    def calibrate(self, seconds):
        count, elapsed = self.handle.call('calibrate', seconds)
        self.ran += count
        return count, elapsed

    def _floor(self, total):
        return total * (1 - self.error_bp / 10000 - SANITY_SLACK)

    def verify(self):
        fresh = [make_draw(self.pool, self.fresh_seed + i, self.case['train'], self.case['test']) for i in range(2)]
        self.fresh_seed += 2
        checks = []
        for visible, truth in fresh:
            predictions, _ = self.handle.call('predict', visible)
            checks.append({'kind': 'fresh', 'correct': int((predictions == truth).sum()), 'total': len(truth)})
        if self.ran:
            outputs = self.handle.call('staged_outputs')
            for truth, predictions in zip(self.staged_truth, outputs):
                checks.append({'kind': 'staged', 'correct': int((predictions == truth).sum()), 'total': len(truth)})
        return {'passed': all(c['correct'] >= self._floor(c['total']) for c in checks), 'checks': checks}


class Reference(Counted):
    def __init__(self, handle, dim):
        super().__init__(handle, 'reference_run')
        self.dim = dim

    def prepare(self):
        self.handle.call('reference_prepare', self.dim)

    def finish(self):
        if not self.handle.call('reference_finish'):
            raise energy.Aborted('reference matmul produced wrong output')


# ---------------------------------------------------------------- modes

def run_test(out, case):
    pool = load_pool(case['size'])
    handle = WorkerHandle('submission')
    try:
        visible, truth = make_draw(pool, case['seed'], case['train'], case['test'])
        predictions, _ = handle.call('predict', visible)
    finally:
        handle.stop()
    correct = int((predictions == truth).sum())
    floor = len(truth) * (1 - case['error_bp'] / 10000 - SANITY_SLACK)
    out.log('test-count', 1)
    out.log('test.0.spec', case['spec'])
    out.log('test.0.status', 'pass' if correct >= floor else 'fail')
    out.log('test.0.message', f'{correct}/{len(truth)} correct on one draw')
    out.log('check', 'pass' if correct >= floor else 'fail')
    return 0 if correct >= floor else EXIT_VALIDATE_FAIL


def accuracy_gate(out, handle, pool, case, seed_offset):
    draws = [make_draw(pool, case['seed'] + seed_offset + i, case['train'], case['test']) for i in range(case['draws'])]
    rows, times = score(handle, draws)
    total = case['test'] * case['draws']
    need = required_correct(total, case['error_bp'])
    out.log('accuracy.correct', sum(rows))
    out.log('accuracy.total', total)
    out.log('accuracy.required', need)
    out.log('accuracy.per_draw', json.dumps(rows))
    return sum(rows) >= need, times


def run_benchmark(out, case):
    pool = load_pool(case['size'])
    handle = WorkerHandle('submission')
    try:
        passed, times = accuracy_gate(out, handle, pool, case, seed_offset=0)
    finally:
        handle.stop()
    ns = [t * 1e9 for t in times]
    mean = sum(ns) / len(ns)
    sd = math.sqrt(sum((x - mean) ** 2 for x in ns) / (len(ns) - 1)) if len(ns) > 1 else 0.0
    out.log('benchmark-count', 1)
    out.log('benchmark.0.spec', case['spec'])
    for key, value in [('runs', len(ns)), ('mean', mean), ('std', sd), ('err', sd / math.sqrt(len(ns))),
                       ('best', min(ns)), ('worst', max(ns))]:
        out.log(f'benchmark.0.{key}', value)
    out.log('check', 'pass' if passed else 'fail')
    return 0 if passed else EXIT_VALIDATE_FAIL


def run_leaderboard(out, case):
    pool = load_pool(case['size'])
    out.log('benchmark-count', 1)
    out.log('benchmark.0.spec', case['spec'])
    device = nvml.open_nvml()
    if device.device_count != 1:
        raise energy.Aborted(f'expected exactly one visible GPU, found {device.device_count}')
    # KernelBot's runner may hold a context of its own; allow what exists now plus our two workers.
    preexisting = device.compute_processes()
    out.log('energy.preexisting_compute_processes', len(preexisting))
    submission = WorkerHandle('submission')
    harness = None
    try:
        passed, _ = accuracy_gate(out, submission, pool, case, seed_offset=1000)
        if not passed:
            out.log('benchmark.0.status', 'fail')
            out.log('benchmark.0.error', 'accuracy below the error band on fresh draws')
            out.log('check', 'fail')
            return EXIT_VALIDATE_FAIL
        staged = [make_draw(pool, case['seed'] + 2000 + i, case['train'], case['test'])
                  for i in range(case.get('staged', 8))]
        harness = WorkerHandle('harness')
        for handle in (submission, harness):
            handle.call('stage', [visible for visible, _ in staged])
        plan = energy.Plan(control=True, task_rounds=case.get('rounds', 3),
                           active_s=case.get('active_s', 20), idle_s=case.get('idle_s', 10),
                           settle_s=case.get('settle_s', 3), reference_s=case.get('reference_s', 10))
        telemetry = Telemetry(device, max_processes=len(preexisting) + 2)
        task = SubmissionTask(submission, pool, case, [truth for _, truth in staged], case['error_bp'])
        try:
            record = energy.run_protocol(telemetry, task, Reference(harness, plan.reference_dim), plan,
                                         record_path=Path('energy-record.json'), log=debug,
                                         control=Counted(harness, 'control'), quiesce=submission,
                                         metadata={'case': case['spec']})
        except energy.Aborted as error:
            record = json.loads(Path('energy-record.json').read_text()) if Path('energy-record.json').exists() else {}
            log_record(out, record)
            out.log('benchmark.0.status', 'fail')
            out.log('benchmark.0.error', f'energy measurement aborted: {error}')
            out.log('check', 'fail')
            infrastructure = 'reference' in str(error) or 'another process' in str(error)
            return EXIT_INFRASTRUCTURE if infrastructure else EXIT_VALIDATE_FAIL
    finally:
        submission.stop()
        if harness is not None:
            harness.stop()
    if 'summary' not in record:
        log_record(out, record)
        raise RuntimeError(f"energy analysis failed: {record.get('summary_error')}")
    summary = record['summary']
    rounds = [w['power']['net_mj'] for w in summary['windows'] if w['kind'] == 'task']
    harness_mj = summary['task']['harness_mj'] or 0.0
    net_nj = [(mj - harness_mj) * 1e6 for mj in rounds]
    mean = summary['task']['net_mj'] * 1e6  # median across rounds, harness-subtracted
    sd = float(np.std(net_nj, ddof=1)) if len(net_nj) > 1 else 0.0
    for key, value in [('runs', len(net_nj)), ('mean', mean), ('std', sd), ('err', sd / math.sqrt(len(net_nj))),
                       ('best', min(net_nj)), ('worst', max(net_nj))]:
        out.log(f'benchmark.0.{key}', value)
    out.log('energy.net_mj', summary['task']['net_mj'])
    out.log('energy.harness_mj', harness_mj)
    out.log('energy.wall_ms', summary['task']['wall_ms'])
    out.log('energy.gates', json.dumps({k: v['status'] for k, v in summary['gates'].items()}))
    out.log('energy.passed', summary['passed'])
    out.log('energy.windows', json.dumps([
        {'kind': w['kind'], 'units': w['units'], 's': round(w['seconds'], 2), 'idle_w': round(w['power']['idle_w'], 2),
         'active_w': round(w['power']['active_w'], 2), 'net_mj': round(w['power']['net_mj'], 4),
         'net_mj_counter': round(w['counter']['net_mj'], 4)} for w in summary['windows']], separators=(',', ':')))
    log_record(out, record)
    out.log('check', 'pass' if summary['passed'] else 'fail')
    return 0 if summary['passed'] else EXIT_VALIDATE_FAIL


def encode(record):
    return base64.b64encode(zlib.compress(json.dumps(record, separators=(',', ':')).encode(), 9)).decode()


def log_record(out, record):
    """Log the raw record when it fits the result pipe; it is always left in energy-record.json."""
    encoded = encode(record)
    if len(encoded) <= MAX_LOGGED_RECORD_BYTES:
        out.log('energy.record', encoded)
    else:
        out.log('energy.record_omitted_bytes', len(encoded))


def main():
    fd = os.getenv('POPCORN_FD')
    if not fd or len(sys.argv) < 3:
        return 111
    mode = sys.argv[1]
    secret = os.environ.pop('POPCORN_SEED', None)  # never visible to workers
    cases = read_cases(sys.argv[2], int(secret) if secret else None)
    CACHE.mkdir(exist_ok=True)
    out = PopcornOutput(int(fd))
    started = time.perf_counter()
    try:
        runner = {'test': run_test, 'benchmark': run_benchmark, 'leaderboard': run_leaderboard}.get(mode)
        if runner is None:
            return 2
        return runner(out, cases[0])
    except Exception as error:
        out.log('check', 'fail')
        out.log('error', repr(error)[:2000])
        raise
    finally:
        debug(f'{mode} finished in {time.perf_counter() - started:.1f} s')
        out.close()


if __name__ == '__main__':
    sys.exit(main())
