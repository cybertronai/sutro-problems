"""Energy protocol and gate tests against a simulated GPU (no hardware needed)."""

import copy
import json
import multiprocessing
import threading
import time
import unittest

import numpy as np

from mnist.code import energy

IDLE_W = 40.0
TASK_W = 50.0          # above idle while the fake task runs
TASK_UNIT_S = 0.002    # -> about 100 mJ above idle per unit
REFERENCE_W = 160.0    # 8.4 J/TFLOP at 19 TFLOP/s, like a healthy A100
REFERENCE_UNIT_S = 2 * 4096 ** 3 / 19e12


class FakeSensor:
    """Sampler-process view of FakeGpu: reads the shared load value."""

    def __init__(self, load, background, power_scale, intruders):
        self.load, self.background, self.power_scale, self.intruders = load, background, power_scale, intruders

    def __call__(self):
        return self

    def power_w(self):
        return (IDLE_W + self.load.value + self.background.value) * self.power_scale

    def extras(self):
        return [self.load.value]

    def foreign_processes(self):
        return list(self.intruders)

    def close(self):
        pass


class FakeGpu:
    """Board power is idle plus the load of whatever workload is running."""

    extra_columns = ['load_w']

    def __init__(self, power_scale=1.0, counter_scale=1.0, intruders=()):
        self.power_scale, self.counter_scale = power_scale, counter_scale
        context = multiprocessing.get_context('spawn')
        self.shared_load = context.Value('d', 0.0, lock=False)
        self.background = context.Value('d', 0.0, lock=False)  # untrusted side work
        self.lock = threading.Lock()
        self.joules = 0.0
        self.since = time.perf_counter()
        self.intruders = list(intruders)

    @property
    def load(self):
        return self.shared_load.value

    def set_load(self, watts):
        with self.lock:
            now = time.perf_counter()
            self.joules += (IDLE_W + self.shared_load.value + self.background.value) * (now - self.since)
            self.since = now
            self.shared_load.value = watts

    def set_background(self, watts):
        load = self.load
        self.set_load(load)  # settle the energy integral at the old background level
        with self.lock:
            self.background.value = watts

    def energy_mj(self):
        self.set_load(self.load)
        return int(self.joules * 1000 * self.counter_scale)

    def sensor(self):
        return FakeSensor(self.shared_load, self.background, self.power_scale, self.intruders)

    def foreign_processes(self):
        return list(self.intruders)

    def describe(self):
        return {'name': 'NVIDIA A100-SXM4-40GB (simulated)'}


class FakeWorkload:
    def __init__(self, gpu, watts, unit_s):
        self.gpu, self.watts, self.unit_s = gpu, watts, unit_s
        self.name, self.scope = 'fake', 'simulated'

    def run(self):
        if self.gpu.load != self.watts:
            self.gpu.set_load(self.watts)
        # Busy-wait: sleep() overshoots by ~40% on macOS, which would distort J/TFLOP.
        deadline = time.perf_counter() + self.unit_s
        while time.perf_counter() < deadline:
            pass

    def synchronize(self):
        self.gpu.set_load(0.0)

    def verify(self):
        return {'passed': True}

    def prepare(self):
        pass

    def finish(self):
        pass


def fast_plan(**overrides):
    plan = energy.Plan(task_rounds=2, settle_s=0.05, idle_s=0.3, active_s=0.6, reference_s=0.4,
                       sample_interval_s=0.01, process_check_interval_s=0.1, calibration_s=0.05)
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def measure(gpu, plan=None):
    task = FakeWorkload(gpu, TASK_W, TASK_UNIT_S)
    reference = FakeWorkload(gpu, REFERENCE_W, REFERENCE_UNIT_S)
    return energy.run_protocol(gpu, task, reference, plan or fast_plan(), log=lambda _: None)


class IntegrationTests(unittest.TestCase):
    def test_exact_trapezoid_with_interpolated_endpoints(self):
        times = np.array([0.0, 1.0, 2.0])
        power = np.array([0.0, 10.0, 10.0])
        self.assertAlmostEqual(energy.integrate_power(times, power, 0.5, 2.0), 3.75 + 10.0)
        with self.assertRaises(ValueError):
            energy.integrate_power(times, power, -0.1, 1.0)


class HealthyHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = measure(FakeGpu())

    def test_passes_and_recovers_known_energy(self):
        summary = self.record['summary']
        self.assertTrue(summary['passed'], json.dumps(summary['gates'], indent=1))
        expected_mj = TASK_W * summary['task']['wall_ms']  # W x ms = mJ
        self.assertAlmostEqual(summary['task']['net_mj'] / expected_mj, 1.0, delta=0.05)
        self.assertAlmostEqual(summary['task']['net_mj_counter'] / expected_mj, 1.0, delta=0.05)
        self.assertAlmostEqual(summary['task']['idle_w'], IDLE_W, delta=1.0)
        ref = [w for w in summary['windows'] if w['kind'] == 'reference']
        self.assertEqual(len(ref), 2)
        for window in ref:
            self.assertAlmostEqual(window['net_j_per_tflop']['power'], 8.4, delta=1.0)

    def test_summary_recomputes_from_raw_record(self):
        record = json.loads(json.dumps(self.record))
        self.assertEqual(json.loads(json.dumps(energy.analyze(record))), record['summary'])

    def test_plan_tampering_is_detected(self):
        record = copy.deepcopy(self.record)
        record['plan']['gates']['round_spread_max_rel'] = 10.0
        with self.assertRaises(ValueError):
            energy.analyze(record)

    def test_counter_disagreement_fails_meter_gate(self):
        record = copy.deepcopy(self.record)
        for row in record['intervals']:
            for edge in ('start', 'end'):
                row[edge]['energy_mj'] = int(row[edge]['energy_mj'] * 1.2)
        self.assertEqual(energy.analyze(record)['gates']['meters_agree']['status'], 'fail')

    def test_baseline_drift_in_sham_fails_residual_gate(self):
        record = copy.deepcopy(self.record)
        sham = next(r for r in record['intervals'] if r['name'].endswith('-sham'))
        a, b = sham['start']['time_s'], sham['end']['time_s']
        for row in record['trace']:
            if a <= row[0] <= b:
                row[1] += 20.0
        summary = energy.analyze(record)
        self.assertEqual(summary['gates']['sham_residual']['status'], 'fail')
        self.assertFalse(summary['passed'])

    def test_missing_verification_fails(self):
        record = copy.deepcopy(self.record)
        record['verification']['after'] = {'passed': False}
        self.assertFalse(energy.analyze(record)['passed'])


class FaultyHostTests(unittest.TestCase):
    def test_scaled_down_sensor_aborts_after_first_reference(self):
        gpu = FakeGpu(power_scale=0.01, counter_scale=0.01)
        started = time.perf_counter()
        with self.assertRaises(energy.Aborted) as caught:
            measure(gpu)
        self.assertIn('reference', str(caught.exception))
        self.assertLess(time.perf_counter() - started, 3.0)

    def test_saved_partial_record_explains_failure(self):
        gpu = FakeGpu(power_scale=0.5, counter_scale=0.5)
        plan = fast_plan()
        task = FakeWorkload(gpu, TASK_W, TASK_UNIT_S)
        reference = FakeWorkload(gpu, REFERENCE_W, REFERENCE_UNIT_S)
        captured = {}
        original = energy.analyze

        def spy(record):
            captured['record'] = record
            return original(record)

        energy.analyze = spy
        try:
            with self.assertRaises(energy.Aborted):
                energy.run_protocol(gpu, task, reference, plan, log=lambda _: None)
        finally:
            energy.analyze = original
        summary = captured['record']['summary']
        self.assertEqual(summary['gates']['reference']['status'], 'fail')
        self.assertFalse(summary['passed'])

    def test_other_gpu_process_aborts(self):
        gpu = FakeGpu(intruders=[{'pid': 4242, 'used_gpu_memory': 1 << 30}])
        with self.assertRaises(energy.Aborted):
            measure(gpu)


class DutyCycleTests(unittest.TestCase):
    def test_diagnostic_reports_each_duty_cycle(self):
        record = measure(FakeGpu(), fast_plan(task_rounds=1, duty_cycles=[0.5], burst_period_s=0.05))
        diagnostic = record['summary']['duty_cycle_diagnostic']
        self.assertEqual([d['duty_cycle'] for d in diagnostic], [0.5])
        # The simulated board draws nothing extra between bursts, so per-unit energy is
        # unchanged. A sampler thread in this process read about 0.65x here (GIL starvation).
        self.assertAlmostEqual(diagnostic[0]['net_mj'] / record['summary']['task']['net_mj'], 1.0, delta=0.1)
        self.assertAlmostEqual(diagnostic[0]['net_mj_counter'] / record['summary']['task']['net_mj'], 1.0, delta=0.1)



class CheatingSubmission(FakeWorkload):
    """Pauses side work while it is being measured and resumes it whenever it is not.

    Work done while idle is being measured inflates the baseline and lowers the
    submission's apparent net energy.
    """

    BACKGROUND_W = 30.0

    def __init__(self, gpu):
        super().__init__(gpu, TASK_W, TASK_UNIT_S)
        self.frozen = False
        self.freezes = 0

    def run(self):
        if self.gpu.background.value:
            self.gpu.set_background(0.0)
        super().run()

    def synchronize(self):
        super().synchronize()
        if not self.frozen:
            self.gpu.set_background(self.BACKGROUND_W)

    def freeze(self):
        self.frozen = True
        self.freezes += 1
        self.gpu.set_background(0.0)

    def thaw(self):
        self.frozen = False


class ControlAndQuiesceTests(unittest.TestCase):
    def test_harness_overhead_is_subtracted(self):
        gpu = FakeGpu()
        task = FakeWorkload(gpu, TASK_W + 10.0, TASK_UNIT_S)       # learner + 10 W of input staging
        control = FakeWorkload(gpu, 10.0, TASK_UNIT_S)             # input staging alone
        reference = FakeWorkload(gpu, REFERENCE_W, REFERENCE_UNIT_S)
        record = energy.run_protocol(gpu, task, reference, fast_plan(task_rounds=1, control=True),
                                     log=lambda _: None, control=control)
        summary = record['summary']
        self.assertTrue(summary['passed'], json.dumps(summary['gates'], indent=1))
        self.assertEqual([w['kind'] for w in record['windows']], ['reference', 'task', 'sham', 'control', 'reference'])
        expected = TASK_W * summary['task']['wall_ms']
        self.assertAlmostEqual(summary['task']['net_mj'] / expected, 1.0, delta=0.05)
        self.assertAlmostEqual(summary['task']['harness_mj'] / (10.0 * summary['task']['wall_ms']), 1.0, delta=0.1)

    def test_control_requires_workload(self):
        gpu = FakeGpu()
        with self.assertRaises(ValueError):
            measure(gpu, fast_plan(control=True))

    def _cheat(self, quiesce):
        gpu = FakeGpu()
        cheat = CheatingSubmission(gpu)
        reference = FakeWorkload(gpu, REFERENCE_W, REFERENCE_UNIT_S)
        record = energy.run_protocol(gpu, cheat, reference, fast_plan(task_rounds=1), log=lambda _: None,
                                     quiesce=cheat if quiesce else None)
        return record['summary'], cheat

    def test_freezing_defeats_idle_window_work(self):
        summary, cheat = self._cheat(quiesce=True)
        self.assertTrue(summary['passed'], json.dumps(summary['gates'], indent=1))
        honest = TASK_W * summary['task']['wall_ms']
        self.assertAlmostEqual(summary['task']['net_mj'] / honest, 1.0, delta=0.05)
        self.assertGreaterEqual(cheat.freezes, 2)

    def test_without_freezing_idle_work_hides_energy(self):
        summary, _ = self._cheat(quiesce=False)
        honest = TASK_W * summary['task']['wall_ms']
        self.assertLess(summary['task']['net_mj'], 0.7 * honest)


if __name__ == '__main__':
    unittest.main()
