"""Replay and verify the 64,074 16x16 submission (standard library only)."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from matmul import score_16x16
from matmul.matmul import _parse
from matmul.submissions.best_66178 import _prove

EXPECTED_SHA256 = 'b5ca6fb67c61825927e6978c13e16a6a432748c32f6c0f1f52a5dbefb3f2e7dd'
BASE_SHA256 = '9d94114a87fecd30168fbcf63931bbc98a50778984a11fe0c3b16940218bcf11'
IR_PATH = Path(__file__).with_suffix('.ir')
EXPECTED_OPERATIONS = {'copy': 1748, 'mul': 4096, 'add': 3840}
EXPECTED_READ_COSTS = {'copy': 20047, 'mul': 18832, 'add': 20751, 'output': 4444}


def decode(ir):
    """Identify values by input index or arithmetic definition, ignoring copies."""
    inputs, operations, outputs = _parse(ir)
    memory = dict(zip(inputs, range(len(inputs))))
    steps = []
    next_value = len(inputs)
    for op, args in operations:
        dest = args[0]
        sources = args[1:] if op == 'copy' or len(args) == 3 else args
        values = tuple(memory[a] for a in sources)
        value = values[0] if op == 'copy' else next_value
        next_value += op != 'copy'
        steps.append((op, dest, values, value))
        memory[dest] = value
    return inputs, steps, [memory[a] for a in outputs]


def generate_best_64074():
    base = IR_PATH.with_name('best_64431.ir').read_bytes()
    assert hashlib.sha256(base).hexdigest() == BASE_SHA256
    inputs, steps, outputs = decode(base.decode())
    # Rows are [boundary in the original 9,440 instructions, input ID, dest].
    captures = json.loads(IR_PATH.with_suffix('.json').read_text())
    assert len(captures) == 244
    assert captures == sorted(captures, key=lambda row: row[0])
    inserted = {}
    for boundary, value, dest in captures:
        assert 0 <= boundary <= len(steps) and 0 <= value < 512
        assert 1 <= dest <= 638
        inserted.setdefault(boundary, []).append(('copy', dest, (value,), value))
    memory = dict(zip(inputs, range(len(inputs))))
    replicas = {v: {a} for a, v in memory.items()}
    lines = [','.join(map(str, inputs))]
    for boundary in range(len(steps) + 1):
        group = inserted.get(boundary, [])
        if boundary < len(steps):
            group = group + [steps[boundary]]
        for op, dest, values, value in group:
            # All reads precede the write, including a read from dest itself.
            sources = [min(replicas[v]) for v in values]
            lines.append(op + ' ' + ','.join(map(str, [dest, *sources])))
            if dest in memory:
                replicas[memory[dest]].remove(dest)
            memory[dest] = value
            replicas.setdefault(value, set()).add(dest)
    lines.append(','.join(str(min(replicas[v])) for v in outputs))
    return '\n'.join(lines) + '\n'


def verify():
    data = IR_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    ir = data.decode()
    assert generate_best_64074() == ir
    score = score_16x16(ir)
    operations, read_costs = _prove(ir)
    assert score == 64074 == sum(read_costs.values())
    assert operations == EXPECTED_OPERATIONS
    assert read_costs == EXPECTED_READ_COSTS
    inputs, ops, outputs = _parse(ir)
    assert max(inputs + outputs + [a for _, args in ops for a in args]) == 638
    return score


if __name__ == '__main__':
    print(f'{IR_PATH.name}: score={verify():,}, sha256={EXPECTED_SHA256}')
    print('Exact symbolic outputs, independent polynomial proof, and byte-exact replay passed.')
    print(f'operations: {EXPECTED_OPERATIONS}')
    print(f'read costs: {EXPECTED_READ_COSTS}')
