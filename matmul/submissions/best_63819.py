"""Verify the frozen 63,819 16x16 artifact; not a constructor or search replay."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from matmul import score_16x16
from matmul.submissions.best_66178 import _prove

EXPECTED_SCORE = 63819
EXPECTED_SHA256 = 'e8adc50783a64088e6864ecfccb7e18a74814e24cfd6e1f9e3045e61fb37f2df'
EXPECTED_OPERATIONS = {'copy': 1932, 'mul': 4096, 'add': 3840}
EXPECTED_READ_COSTS = {'copy': 20639, 'mul': 18477, 'add': 20256, 'output': 4447}
IR_PATH = Path(__file__).with_suffix('.ir')


def verify():
    data = IR_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise AssertionError('frozen artifact SHA-256 mismatch')
    ir = data.decode('utf-8')
    score = score_16x16(ir)
    operations, read_costs = _prove(ir)
    if score != EXPECTED_SCORE or sum(read_costs.values()) != EXPECTED_SCORE:
        raise AssertionError('official or independent score mismatch')
    if operations != EXPECTED_OPERATIONS:
        raise AssertionError('operation counts mismatch')
    if read_costs != EXPECTED_READ_COSTS:
        raise AssertionError('read-cost breakdown mismatch')
    return score


if __name__ == '__main__':
    print(f'{IR_PATH.name}: score={verify():,}, sha256={EXPECTED_SHA256}')
    print('Frozen artifact verified: official symbolic scorer and independent exact polynomial proof.')
    print('No constructor or search replay is provided.')
    print(f'operations: {EXPECTED_OPERATIONS}')
    print(f'read costs: {EXPECTED_READ_COSTS}')
