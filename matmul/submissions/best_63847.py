"""Verify the frozen 63,847 16x16 artifact; not a constructor or search replay."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from matmul import score_16x16
from matmul.submissions.best_66178 import _prove

EXPECTED_SCORE = 63847
EXPECTED_SHA256 = '30651a3d04111d1e7f799eb7c246c856c2ceefa9a1bce16239f71de0930e46f5'
EXPECTED_OPERATIONS = {'copy': 1844, 'mul': 4096, 'add': 3840}
EXPECTED_READ_COSTS = {'copy': 20315, 'mul': 19075, 'add': 20066, 'output': 4391}
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
