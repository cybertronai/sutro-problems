"""Minimal ctypes binding to NVML for board power, energy and process checks.

KernelBot's image has no pynvml, but GPU containers ship the driver's
libnvidia-ml.so.1. Only the calls the energy protocol needs are bound.
"""
import ctypes
import os
import time

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_TEMPERATURE_GPU = 0


class NvmlError(RuntimeError):
    pass


class _Utilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class _ProcessInfo(ctypes.Structure):
    # nvmlProcessInfo_v2_t / v3 layout
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]


class Nvml:
    def __init__(self, index=0):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2(), 'nvmlInit_v2')
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), 'nvmlDeviceGetCount_v2')
        self.device_count = count.value
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)),
                    'nvmlDeviceGetHandleByIndex_v2')

    def _check(self, code, name):
        if code != NVML_SUCCESS:
            self.lib.nvmlErrorString.restype = ctypes.c_char_p
            raise NvmlError(f'{name} failed: {self.lib.nvmlErrorString(code).decode()} ({code})')

    def _uint(self, function, *args):
        value = ctypes.c_uint()
        self._check(getattr(self.lib, function)(self.handle, *args, ctypes.byref(value)), function)
        return value.value

    def _string(self, function, size, device=True):
        buffer = ctypes.create_string_buffer(size)
        args = (self.handle, buffer, size) if device else (buffer, size)
        self._check(getattr(self.lib, function)(*args), function)
        return buffer.value.decode()

    def power_w(self):
        return self._uint('nvmlDeviceGetPowerUsage') / 1000

    def energy_mj(self):
        value = ctypes.c_ulonglong()
        self._check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)),
                    'nvmlDeviceGetTotalEnergyConsumption')
        return value.value

    def temperature_c(self):
        return self._uint('nvmlDeviceGetTemperature', NVML_TEMPERATURE_GPU)

    def pstate(self):
        return self._uint('nvmlDeviceGetPerformanceState')

    def utilization_percent(self):
        util = _Utilization()
        self._check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)),
                    'nvmlDeviceGetUtilizationRates')
        return util.gpu

    def compute_processes(self):
        function = getattr(self.lib, 'nvmlDeviceGetComputeRunningProcesses_v3', None) \
            or self.lib.nvmlDeviceGetComputeRunningProcesses_v2
        count = ctypes.c_uint(0)
        code = function(self.handle, ctypes.byref(count), None)
        if code == NVML_SUCCESS:
            return []
        if code != NVML_ERROR_INSUFFICIENT_SIZE:
            self._check(code, 'nvmlDeviceGetComputeRunningProcesses')
        rows = (_ProcessInfo * (count.value + 8))()
        count = ctypes.c_uint(len(rows))
        self._check(function(self.handle, ctypes.byref(count), rows), 'nvmlDeviceGetComputeRunningProcesses')
        return [{'pid': rows[i].pid, 'used_gpu_memory': rows[i].usedGpuMemory} for i in range(count.value)]

    def describe(self):
        return {
            'name': self._string('nvmlDeviceGetName', 96),
            'uuid': self._string('nvmlDeviceGetUUID', 96),
            'vbios': self._string('nvmlDeviceGetVbiosVersion', 32),
            'driver': self._string('nvmlSystemGetDriverVersion', 80, device=False),
            'power_limit_w': self._uint('nvmlDeviceGetPowerManagementLimit') / 1000,
            'default_power_limit_w': self._uint('nvmlDeviceGetPowerManagementDefaultLimit') / 1000,
            'device_count': self.device_count,
        }

    def close(self):
        self.lib.nvmlShutdown()


class FakeNvml:
    """Constant 50 W board for local dry runs of the evaluator plumbing (MNIST_EVAL_FAKE_NVML=1)."""

    device_count = 1

    def power_w(self):
        return 50.0

    def energy_mj(self):
        return int(time.perf_counter() * 50_000)

    def temperature_c(self):
        return 40

    def pstate(self):
        return 0

    def utilization_percent(self):
        return 0

    def compute_processes(self):
        return []

    def describe(self):
        return {'name': 'fake (dry run)', 'device_count': 1}

    def close(self):
        pass


def open_nvml():
    return FakeNvml() if os.environ.get('MNIST_EVAL_FAKE_NVML') == '1' else Nvml()


class Sensor:
    """Picklable; opens its own NVML session inside the sampler process."""

    def __init__(self, max_processes):
        self.max_processes = max_processes

    def __call__(self):
        self.nvml = open_nvml()
        return self

    def power_w(self):
        return self.nvml.power_w()

    def extras(self):
        return [self.nvml.temperature_c(), self.nvml.pstate(), self.nvml.utilization_percent()]

    def foreign_processes(self):
        return excess_processes(self.nvml, self.max_processes)

    def close(self):
        self.nvml.close()


def excess_processes(nvml, max_processes):
    """Containers may report every context under one PID alias, so count contexts instead of matching PIDs."""
    rows = nvml.compute_processes()
    return rows if len(rows) > max_processes else []
