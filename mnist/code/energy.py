"""Shared GPU energy protocol for MNIST submissions.

One complete training-and-prediction run is far shorter than the GPU power
sensor's resolution, so a task is replayed many times inside each active
window and energy is expressed per replay:

    net mJ/task = (active J - mean(idle-before W, idle-after W) * active s) * 1000 / replays

Every window is bracketed by settled idle measurements. Idle-only shams bound
baseline-subtraction error, and a fixed FP32 matrix-multiply reference checks
that the power telemetry is plausible before task energy is trusted. Power is
polled from a separate process so the measured workload cannot bias sampling. Board
energy is read two ways, a trapezoidal integral of sampled power and the
cumulative energy counter; both come from the same GPU sensors and are not an
external wattmeter.

This module needs only NumPy. `run_protocol` drives any telemetry and workload
objects; `analyze` recomputes every reported number and gate from the raw
record. Hardware access lives in energy_nvml.py.

    python -m mnist.code.energy analyze RESULT.json[.gz]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import queue as queue_module
import statistics
import time

import numpy as np

HARNESS_VERSION = 1

# Net energy of the 4096 x 4096 FP32 matrix-multiply reference (TF32 off).
# Two healthy A100-SXM4-40GB hosts in small-qda-20260915/results measured
# 8.49 and 8.36 J per 10^12 FLOPs at 18.9-19.0 TFLOP/s, despite 320 W vs 400 W
# power limits and 38 W vs 58 W idle. The faulty host behind the original
# PCA-QDA claim read about 1.4 W above idle under dense FP32 matmul, which is
# roughly 0.07 J/TFLOP. The bands are deliberately wide until more hosts are
# measured; they are meant to reject broken telemetry, not rank boards.
REFERENCE_EXPECTATIONS = {
    'NVIDIA A100': {'net_j_per_tflop': [6.0, 11.0], 'tflops_per_s': [15.0, 23.0]},
}

DEFAULT_GATES = {
    'max_sample_gap_s': 0.5,
    'meter_agreement_max_rel': 0.03,
    'sham_residual_max_rel': 0.05,
    'round_spread_max_rel': 0.10,
}


@dataclass
class Plan:
    task_rounds: int = 3
    shams: int = 1
    settle_s: float = 3.0
    idle_s: float = 10.0
    active_s: float = 20.0
    reference_s: float = 10.0
    reference_dim: int = 4096
    sample_interval_s: float = 0.05
    process_check_interval_s: float = 1.0
    calibration_s: float = 1.0
    duty_cycles: list = field(default_factory=list)
    burst_period_s: float = 0.2
    control: bool = False
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))

    def sequence(self):
        """Reference first (abort early on bad telemetry); shams and the harness control after the first task window."""
        kinds = ['reference']
        for i in range(self.task_rounds):
            kinds.append('task')
            if i == 0:
                kinds.extend(['sham'] * self.shams)
                if self.control:
                    kinds.append('control')
        kinds.extend(f'duty:{d}' for d in self.duty_cycles)
        kinds.append('reference')
        return kinds


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- analysis

def _trace_arrays(record):
    columns = record['trace_columns']
    trace = np.array(sorted(record['trace'], key=lambda row: row[0]), dtype=np.float64)
    if trace.ndim != 2 or len(trace) < 2:
        raise ValueError('trace needs at least two samples')
    return trace[:, columns.index('time_s')], trace[:, columns.index('power_w')]


def integrate_power(times, power, start, end):
    """Trapezoidal integral of sampled power over [start, end], interpolating both endpoints."""
    if not (times[0] <= start < end <= times[-1]):
        raise ValueError(f'interval [{start}, {end}] is outside the power trace')
    inner = times[(times > start) & (times < end)]
    grid = np.r_[start, inner, end]
    return float(np.trapezoid(np.interp(grid, times, power), grid))


def interval_energy(times, power, interval):
    a, b = interval['start']['time_s'], interval['end']['time_s']
    seconds = b - a
    counter_j = (interval['end']['energy_mj'] - interval['start']['energy_mj']) / 1000
    if seconds <= 0 or counter_j < 0:
        raise ValueError(f"{interval['name']}: clock or energy counter moved backwards")
    inside = times[(times >= a) & (times <= b)]
    gaps = np.diff(np.r_[a, inside, b])
    return {'seconds': seconds, 'counter_j': counter_j,
            'power_j': integrate_power(times, power, a, b),
            'samples': int(len(inside)), 'max_sample_gap_s': float(gaps.max())}


def _window_rows(record, times, power):
    intervals = {row['name']: row for row in record['intervals']}
    energies = {name: interval_energy(times, power, row) for name, row in intervals.items()}
    rows = []
    for window in record['windows']:
        before, active, after = (energies[window[k]] for k in ('before', 'active', 'after'))
        units = window['nominal_units']
        row = {'kind': window['kind'], 'active': window['active'], 'units': intervals[window['active']]['units'],
               'nominal_units': units, 'seconds': active['seconds'],
               'wall_ms_per_unit': active['seconds'] * 1000 / units,
               'max_sample_gap_s': max(e['max_sample_gap_s'] for e in (before, active, after))}
        if 'duty_cycle' in window:
            row['duty_cycle'] = window['duty_cycle']
        for meter in ('power', 'counter'):
            idle_w = statistics.mean(e[meter + '_j'] / e['seconds'] for e in (before, after))
            active_w = active[meter + '_j'] / active['seconds']
            row[meter] = {'active_w': active_w, 'idle_w': idle_w, 'above_idle_w': active_w - idle_w,
                          'gross_mj': active[meter + '_j'] * 1000 / units,
                          'net_mj': (active[meter + '_j'] - idle_w * active['seconds']) * 1000 / units}
        if window['kind'] == 'reference':
            flops = 2 * window['reference_dim'] ** 3 * row['units']
            row['tflops_per_s'] = flops / 1e12 / active['seconds']
            row['net_j_per_tflop'] = {m: row[m]['net_mj'] * row['units'] / 1000 / (flops / 1e12)
                                      for m in ('power', 'counter')}
        rows.append(row)
    return rows


def _expectation(device_name):
    for prefix, expected in REFERENCE_EXPECTATIONS.items():
        if str(device_name).startswith(prefix):
            return prefix, expected
    return None, None


def _gate(passed, **detail):
    return {'status': 'pass' if passed else 'fail', **detail}


def reference_gate(rows, device_name):
    refs = [r for r in rows if r['kind'] == 'reference']
    prefix, expected = _expectation(device_name)
    if not refs:
        return {'status': 'fail', 'reason': 'no reference window completed'}
    if expected is None:
        return {'status': 'unchecked', 'reason': f'no reference expectation for {device_name!r}'}
    lo, hi = expected['net_j_per_tflop']
    tlo, thi = expected['tflops_per_s']
    checks = [{'window': r['active'], 'tflops_per_s': r['tflops_per_s'], 'net_j_per_tflop': r['net_j_per_tflop'],
               'ok': all(lo <= v <= hi for v in r['net_j_per_tflop'].values()) and tlo <= r['tflops_per_s'] <= thi}
              for r in refs]
    return _gate(all(c['ok'] for c in checks), expectation=prefix, net_j_per_tflop_band=[lo, hi],
                 tflops_per_s_band=[tlo, thi], windows=checks)


def _rel(a, b):
    return abs(a - b) / abs(b) if b else math.inf


def analyze(record):
    """Recompute all per-window values, headline estimates and gates from raw samples."""
    plan = record['plan']
    if sha256_json(plan) != record['plan_sha256']:
        raise ValueError('plan does not match its recorded hash')
    gates_cfg = plan['gates']
    times, power = _trace_arrays(record)
    if not (np.isfinite(power).all() and (power >= 0).all()):
        raise ValueError('power samples must be finite and non-negative')
    rows = _window_rows(record, times, power)
    tasks = [r for r in rows if r['kind'] == 'task']
    shams = [r for r in rows if r['kind'] == 'sham']
    controls = [r for r in rows if r['kind'] == 'control']
    device = record.get('device', {}).get('name', '')

    gates = {'reference': reference_gate(rows, device)}
    gates['telemetry'] = _gate(not record['monitor_errors'] and all(r['max_sample_gap_s'] <= gates_cfg['max_sample_gap_s'] for r in rows),
                               monitor_errors=len(record['monitor_errors']),
                               max_sample_gap_s=max((r['max_sample_gap_s'] for r in rows), default=None))
    gates['exclusive_gpu'] = _gate(not record.get('foreign_processes'), foreign_processes=record.get('foreign_processes', []))
    verification = record.get('verification', {})
    gates['task_verified'] = _gate(bool(verification) and all(v.get('passed') for v in verification.values()),
                                   checks=sorted(verification))
    summary = {'harness_version': record['harness_version'], 'device': device, 'windows': rows, 'gates': gates}
    if tasks:
        agreement = max(_rel(r['counter']['active_w'], r['power']['active_w']) for r in tasks)
        gates['meters_agree'] = _gate(agreement <= gates_cfg['meter_agreement_max_rel'], max_rel_difference=agreement)
        net = {m: [r[m]['net_mj'] for r in tasks] for m in ('power', 'counter')}
        raw_net = statistics.median(net['power'])
        spread = (max(net['power']) - min(net['power'])) / abs(raw_net) if raw_net else math.inf
        gates['round_spread'] = _gate(spread <= gates_cfg['round_spread_max_rel'], rel_range=spread)
        # The harness control replays the per-call input staging without the learner;
        # its net energy per unit is overhead the submission did not cause.
        harness = {m: statistics.median(r[m]['net_mj'] for r in controls) if controls else 0.0
                   for m in ('power', 'counter')}
        median_net = raw_net - harness['power']
        residual = max((abs(r[m]['net_mj']) for r in shams for m in ('power', 'counter')), default=math.inf)
        gates['sham_residual'] = _gate(residual <= gates_cfg['sham_residual_max_rel'] * abs(median_net),
                                       max_abs_mj_per_nominal_unit=residual,
                                       rel_to_task=residual / abs(median_net) if median_net else math.inf)
        summary['task'] = {
            'rounds': len(tasks),
            'net_mj': median_net,
            'net_mj_counter': statistics.median(net['counter']) - harness['counter'],
            'net_mj_before_harness_subtraction': raw_net,
            'harness_mj': harness['power'] if controls else None,
            'net_mj_range': [min(net['power']), max(net['power'])],
            'gross_mj': statistics.median(r['power']['gross_mj'] for r in tasks),
            'idle_w': statistics.median(r['power']['idle_w'] for r in tasks),
            'above_idle_w': statistics.median(r['power']['above_idle_w'] for r in tasks),
            'wall_ms': statistics.median(r['wall_ms_per_unit'] for r in tasks),
        }
        refs = [r['net_j_per_tflop']['power'] for r in rows if r['kind'] == 'reference']
        if refs and statistics.median(refs) > 0:  # a dead sensor reads zero; its gate fails instead
            summary['task']['reference_equivalent_tflops'] = median_net / 1000 / statistics.median(refs)
    else:
        gates['task_windows'] = {'status': 'fail', 'reason': 'no task window completed'}
    duty = [r for r in rows if r['kind'] == 'duty']
    if duty:
        summary['duty_cycle_diagnostic'] = [{'duty_cycle': r['duty_cycle'], 'net_mj': r['power']['net_mj'],
                                             'net_mj_counter': r['counter']['net_mj'],
                                             'wall_ms_per_unit': r['wall_ms_per_unit']} for r in duty]
    summary['passed'] = bool(record.get('complete')) and all(g['status'] == 'pass' for g in gates.values())
    return summary


# ---------------------------------------------------------------- protocol

class Aborted(RuntimeError):
    pass


def _sampler(sensor_spec, epoch, interval_s, check_s, queue, stop):
    """Child process: poll board power so the task's Python loop cannot starve sampling.

    A sampler thread in the measured process only runs when the GIL is free, so
    a host loop that holds it (Python dispatch, busy waits) biases samples
    toward idle gaps. perf_counter is system-wide on Linux and macOS, so child
    timestamps share the parent's clock.
    """
    sensor = sensor_spec()
    last_check = -math.inf
    try:
        while not stop.is_set():
            try:
                a = time.perf_counter() - epoch
                watts = float(sensor.power_w())
                b = time.perf_counter() - epoch
                queue.put(('row', [(a + b) / 2, watts, *sensor.extras()]))
                if b - last_check >= check_s:
                    last_check = b
                    foreign = sensor.foreign_processes()
                    if foreign:
                        queue.put(('foreign', {'time_s': b, 'processes': foreign}))
            except Exception as error:
                queue.put(('error', repr(error)))
            stop.wait(interval_s)
    finally:
        sensor.close()
        queue.put(('done', None))


class _Session:
    """Owns the sampler process and raw record for one protocol run."""

    def __init__(self, telemetry, plan, record, log):
        self.telemetry, self.plan, self.record, self.log = telemetry, plan, record, log
        context = multiprocessing.get_context('spawn')  # never fork a CUDA context
        self.queue, self.stop = context.Queue(), context.Event()
        self.epoch = time.perf_counter()
        self.process = context.Process(
            target=_sampler, daemon=True,
            args=(telemetry.sensor(), self.epoch, plan.sample_interval_s,
                  plan.process_check_interval_s, self.queue, self.stop))
        self.finished = False

    def now(self):
        return time.perf_counter() - self.epoch

    def _handle(self, message):
        kind, value = message
        if kind == 'row':
            self.record['trace'].append(value)
        elif kind == 'error':
            self.record['monitor_errors'].append(value)
        elif kind == 'foreign':
            self.record.setdefault('foreign_processes', []).append(value)
        elif kind == 'done':
            self.finished = True

    def drain(self):
        while True:
            try:
                self._handle(self.queue.get_nowait())
            except queue_module.Empty:
                return

    def wait_for_sample_after(self, t, timeout_s=30.0):
        deadline = time.perf_counter() + timeout_s
        while not (self.record['trace'] and self.record['trace'][-1][0] > t):
            if time.perf_counter() > deadline or not self.process.is_alive():
                raise Aborted('power sampler stopped producing samples')
            try:
                self._handle(self.queue.get(timeout=0.1))
            except queue_module.Empty:
                pass
        self.drain()

    def start(self):
        self.process.start()
        self.wait_for_sample_after(self.now(), timeout_s=120.0)

    def close(self):
        self.stop.set()
        deadline = time.perf_counter() + 30
        while not self.finished and time.perf_counter() < deadline and self.process.is_alive():
            try:
                self._handle(self.queue.get(timeout=0.1))
            except queue_module.Empty:
                pass
        self.drain()
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()

    def stamp(self):
        a = self.now()
        energy = int(self.telemetry.energy_mj())
        b = self.now()
        return {'time_s': (a + b) / 2, 'energy_mj': energy, 'read_s': b - a}

    def check(self):
        self.drain()
        foreign = self.telemetry.foreign_processes()
        if foreign:
            self.record.setdefault('foreign_processes', []).append({'time_s': self.now(), 'processes': foreign})
        if self.record.get('foreign_processes'):
            raise Aborted('another process is using the GPU: ' + json.dumps(self.record['foreign_processes'][-1]))
        if self.record['monitor_errors']:
            raise Aborted('telemetry sampler failed: ' + self.record['monitor_errors'][0])

    def interval(self, name, seconds=0.0, workload=None, units=0, burst=None, batch=10):
        self.check()
        if workload is not None:
            workload.synchronize()
        start = self.stamp()
        done = 0
        if workload is not None and units:
            for _ in range(units):
                workload.run()
            workload.synchronize()
            done = units
        elif workload is not None and burst:
            # Duty-cycle diagnostic: bursts of work separated by host sleeps.
            size, period = burst
            deadline = self.now() + seconds
            while self.now() < deadline:
                t0 = self.now()
                for _ in range(size):
                    workload.run()
                workload.synchronize()
                done += size
                time.sleep(max(0.0, period - (self.now() - t0)))
        elif workload is not None:
            deadline = self.now() + seconds
            while self.now() < deadline:
                for _ in range(batch):
                    workload.run()
                workload.synchronize()
                done += batch
        else:
            time.sleep(seconds)
        end = self.stamp()
        self.wait_for_sample_after(end['time_s'])
        self.check()
        row = {'name': name, 'start': start, 'end': end, 'units': done}
        self.record['intervals'].append(row)
        self.log({'interval': name, 'seconds': end['time_s'] - start['time_s'], 'units': done})
        return name

    def settled_idle(self, name):
        self.interval(name + '-settle', seconds=self.plan.settle_s)
        return self.interval(name, seconds=self.plan.idle_s)


def calibrate_units(task, seconds, target_s):
    """Run the (already warm) task for about `seconds` and size active windows to `target_s`.

    A task running in another process may provide calibrate(seconds) -> (count,
    elapsed) so calibration is not dominated by per-unit messaging.
    """
    if hasattr(task, 'calibrate'):
        count, elapsed = task.calibrate(seconds)
    else:
        task.synchronize()
        count, start = 0, time.perf_counter()
        while True:
            task.run()
            task.synchronize()
            count += 1
            elapsed = time.perf_counter() - start
            if elapsed >= seconds:
                break
    unit_s = elapsed / count
    return max(1, math.ceil(target_s / unit_s)), unit_s


class _NoQuiesce:
    def freeze(self):
        pass

    def thaw(self):
        pass


def run_protocol(telemetry, task, reference, plan, record_path=None, log=print, metadata=None,
                 control=None, quiesce=None):
    """Measure `task` and return the raw record with its summary.

    telemetry (parent process): energy_mj() cumulative board energy,
        foreign_processes() -> list, describe() -> dict, extra_columns, and
        sensor() -> a picklable zero-argument callable that builds, inside the
        sampler process, an object with power_w(), extras(), foreign_processes()
        and close().
    task / reference: run() launches one unit (may be asynchronous) and
        synchronize() waits for completion. task.verify() -> dict with 'passed';
        reference.prepare() / finish() bracket reference windows.
    control: optional workload replaying the harness's per-unit overhead without
        the learner (plan.control must be set); its net energy is subtracted.
    quiesce: optional freeze() / thaw(). The protocol keeps the task's process
        frozen except during calibration, task windows and verification, so
        untrusted code cannot run while idle, reference or control energy is measured.
    """
    quiesce = quiesce or _NoQuiesce()
    if plan.control and control is None:
        raise ValueError('plan.control requires a control workload')
    plan_doc = asdict(plan)
    record = {
        'harness_version': HARNESS_VERSION,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'plan': plan_doc, 'plan_sha256': sha256_json(plan_doc),
        'device': telemetry.describe(),
        'task': {'name': getattr(task, 'name', type(task).__name__), 'scope': getattr(task, 'scope', None),
                 **(metadata or {})},
        'trace_columns': ['time_s', 'power_w', *telemetry.extra_columns],
        'trace': [], 'intervals': [], 'windows': [], 'monitor_errors': [],
        'verification': {}, 'complete': False,
    }

    def save():
        if record_path is not None:
            Path(record_path).write_text(json.dumps(record, separators=(',', ':')) + '\n')

    session = _Session(telemetry, plan, record, log)
    try:
        record['verification']['before'] = task.verify()
        units, unit_s = calibrate_units(task, plan.calibration_s, plan.active_s)
        record['task'].update({'units_per_window': units, 'calibration_unit_ms': unit_s * 1000})
        quiesce.freeze()
        session.start()
        for index, kind in enumerate(plan.sequence()):
            name = f'{index:02d}-{kind}'
            window = {'kind': kind.split(':')[0], 'before': session.settled_idle(name + '-idle-before')}
            if kind == 'reference':
                reference.prepare()
                window['active'] = session.interval(name, seconds=plan.reference_s, workload=reference)
                reference.finish()
                window['reference_dim'] = plan.reference_dim
            elif kind == 'control':
                # Timed like a task window: staging alone is fast, and the energy
                # counter updates too coarsely for a window lasting milliseconds.
                window['active'] = session.interval(name, seconds=plan.active_s, workload=control, batch=units)
            elif kind == 'task':
                quiesce.thaw()
                window['active'] = session.interval(name, workload=task, units=units)
                quiesce.freeze()
            elif kind == 'sham':
                window['active'] = session.interval(name, seconds=plan.active_s)
            else:
                duty = float(kind.split(':')[1])
                size = max(1, round(duty * plan.burst_period_s / unit_s))
                window['duty_cycle'] = duty
                quiesce.thaw()
                window['active'] = session.interval(name, seconds=plan.active_s, workload=task,
                                                    burst=(size, plan.burst_period_s))
                quiesce.freeze()
            window['after'] = session.settled_idle(name + '-idle-after')
            completed = next(r for r in record['intervals'] if r['name'] == window['active'])
            window['nominal_units'] = completed['units'] or units
            record['windows'].append(window)
            save()
            if kind == 'reference' and index == 0:
                gate = reference_gate(_window_rows(record, *_trace_arrays(record)), record['device'].get('name', ''))
                log({'reference_gate': gate})
                if gate['status'] == 'fail':
                    raise Aborted('reference telemetry check failed; task energy would not be trustworthy')
        quiesce.thaw()
        record['verification']['after'] = task.verify()
        record['complete'] = True
    except BaseException as error:
        record['failure'] = repr(error)
        raise
    finally:
        quiesce.thaw()
        if session.process.pid is not None:
            session.close()
        record['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            record['summary'] = analyze(record)
        except Exception as error:
            record['summary_error'] = repr(error)
        save()
    return record


def main():
    parser = argparse.ArgumentParser(description='Recompute a saved energy record without a GPU.')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('analyze')
    check.add_argument('record', type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.record.suffix == '.gz' else open
    with opener(args.record, 'rt') as stream:
        record = json.load(stream)
    summary = analyze(record)
    stored = record.get('summary')
    if stored is not None and json.loads(json.dumps(summary)) != stored:
        raise SystemExit('saved summary differs from recomputation')
    print(json.dumps({k: v for k, v in summary.items() if k != 'windows'}, indent=2))
    if not summary['passed']:
        raise SystemExit('one or more gates failed')


if __name__ == '__main__':
    main()
