"""GPU worker processes for the MNIST energy evaluator.

Role "submission" imports the untrusted learner. Role "harness" never does:
it runs the matmul telemetry reference and the input-staging control. Workers
receive only training images, training labels and test images; test labels,
draw seeds and the label permutation stay in the parent evaluator.

Contract with the learner: custom_kernel receives the same three input tensors
on every call (contents change between calls), so learners may capture CUDA
graphs. Staging a draw into those tensors is harness overhead; the control
window measures it so the parent can subtract it.
"""
import os
import signal
import time

import numpy as np
import torch

DEVICE = os.environ.get('MNIST_EVAL_DEVICE', 'cuda')  # 'cpu' only for local dry runs of the plumbing


def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


class Inputs:
    def __init__(self, train_shape, test_shape):
        self.train_x = torch.empty(train_shape, dtype=torch.float32, device=DEVICE)
        self.train_y = torch.empty(train_shape[0], dtype=torch.int64, device=DEVICE)
        self.test_x = torch.empty(test_shape, dtype=torch.float32, device=DEVICE)

    def as_tuple(self):
        return self.train_x, self.train_y, self.test_x

    def load(self, draw):
        self.train_x.copy_(draw[0], non_blocking=True)
        self.train_y.copy_(draw[1], non_blocking=True)
        self.test_x.copy_(draw[2], non_blocking=True)


def _to_device(draw):
    return tuple(torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE) for a in draw)


def _validate_output(out, test_count):
    if not isinstance(out, torch.Tensor) or out.device.type != DEVICE:
        raise TypeError(f'custom_kernel must return a {DEVICE} tensor')
    if out.shape != (test_count,):
        raise ValueError(f'custom_kernel returned shape {tuple(out.shape)}, expected ({test_count},)')


class Worker:
    def __init__(self, role):
        self.role = role
        self.inputs = None
        self.staged = []
        self.outputs = None
        self.kernel = None
        if role == 'submission':
            from submission import custom_kernel
            self.kernel = custom_kernel
        if DEVICE == 'cuda':
            torch.cuda.init()
        sync()

    def ensure_inputs(self, draw):
        train_shape, test_shape = draw[0].shape, draw[2].shape
        if self.inputs is None:
            self.inputs = Inputs(train_shape, test_shape)
        elif tuple(self.inputs.train_x.shape) != tuple(train_shape) or tuple(self.inputs.test_x.shape) != tuple(test_shape):
            raise ValueError('all draws in one evaluation must share shapes')

    # -- commands -------------------------------------------------------
    def predict(self, draw):
        """One fresh, unstaged draw: warm-up, accuracy and verification calls."""
        self.ensure_inputs(draw)
        self.inputs.load(_to_device(draw))
        sync()
        start = time.perf_counter()
        out = self.kernel(self.inputs.as_tuple())
        sync()
        elapsed = time.perf_counter() - start
        _validate_output(out, draw[2].shape[0])
        return out.to(torch.int64).cpu().numpy(), elapsed

    def stage(self, draws):
        self.ensure_inputs(draws[0])
        self.staged = [_to_device(d) for d in draws]
        self.outputs = torch.zeros((len(draws), draws[0][2].shape[0]), dtype=torch.int64, device=DEVICE)
        self.no_op = torch.zeros(draws[0][2].shape[0], dtype=torch.int64, device=DEVICE)
        sync()
        self.cursor = 0
        return len(self.staged)

    def run(self, count):
        """Stage the next draw, call the learner, keep its output for spot checks."""
        staged, outputs, inputs, kernel = self.staged, self.outputs, self.inputs, self.kernel
        cursor = self.cursor
        for _ in range(count):
            index = cursor % len(staged)
            inputs.load(staged[index])
            outputs[index].copy_(kernel(inputs.as_tuple()), non_blocking=True)
            cursor += 1
        sync()
        self.cursor = cursor
        return count

    def control(self, count):
        """The same staging and output copy with no learner."""
        staged, outputs, inputs = self.staged, self.outputs, self.inputs
        cursor = self.cursor
        for _ in range(count):
            index = cursor % len(staged)
            inputs.load(staged[index])
            outputs[index].copy_(self.no_op, non_blocking=True)
            cursor += 1
        sync()
        self.cursor = cursor
        return count

    def calibrate(self, seconds):
        sync()
        count, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            self.run(1)
            count += 1
        return count, time.perf_counter() - start

    def staged_outputs(self):
        return self.outputs.cpu().numpy()

    def reference_prepare(self, dim):
        exponent = dim.bit_length() - 1
        if dim & (dim - 1) or exponent % 2:
            raise ValueError('reference dimension must be a power of four')
        torch.backends.cuda.matmul.allow_tf32 = False
        value = 2.0 ** -(exponent // 2)  # every product entry is exactly 1
        self.ref = [torch.full((dim, dim), value, dtype=torch.float32, device=DEVICE) for _ in range(2)]
        self.ref.append(torch.empty_like(self.ref[0]))
        torch.mm(self.ref[0], self.ref[1], out=self.ref[2])
        sync()
        return True

    def reference_run(self, count):
        a, b, c = self.ref
        for _ in range(count):
            torch.mm(a, b, out=c)
        sync()
        return count

    def reference_finish(self):
        exact = bool((self.ref[2] == 1).all())
        self.ref = None
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return exact


def main(connection, role, cwd):
    # The evaluator freezes this process with SIGSTOP. A stopped process in a process
    # group that becomes orphaned triggers SIGHUP to the whole group, which would
    # include KernelBot's runner; a session of our own keeps any such signal here.
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.chdir(cwd)
    try:
        worker = Worker(role)
        connection.send(('ready', os.getpid()))
    except BaseException as error:
        connection.send(('error', repr(error)))
        return
    while True:
        command, args = connection.recv()
        if command == 'stop':
            connection.send(('ok', None))
            return
        try:
            connection.send(('ok', getattr(worker, command)(*args)))
        except BaseException as error:
            connection.send(('error', repr(error)))
