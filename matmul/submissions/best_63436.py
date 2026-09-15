"""Verify all 256 exact matrix outputs and the 63,436 weighted-read score."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
EXPECTED_SCORE=63436
EXPECTED_SHA256='30ef2350bc861be0dbbb045844bd26a19fe30906cb9a7443c8d9373642a7de68'
EXPECTED_OPERATIONS={'copy': 2134, 'mul': 4096, 'add': 3840}
EXPECTED_READ_COSTS={'copy': 20595, 'mul': 18369, 'add': 20173, 'output': 4299}
DEPENDENCIES={'matmul/matmul.py': 'cb701af7e34aa330492a76e43483c7953bb55a883c95398524bcc551a15fd4d9', 'matmul/__init__.py': 'b81bd729b9013020e40278c36956069bc54eda91a152595233b9355eabe60853', 'matmul/submissions/best_66178.py': '56045f1436cfce083c570069917e9242bfa11f2df91b27bd1a64cbca3c086284'}

def verify():
    for name,expected in DEPENDENCIES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    from matmul import score_16x16
    from matmul.submissions.best_66178 import _prove
    data=Path(__file__).with_suffix('.ir').read_bytes()
    assert hashlib.sha256(data).hexdigest()==EXPECTED_SHA256
    score=score_16x16(data.decode());operations,reads=_prove(data.decode())
    assert score==EXPECTED_SCORE==sum(reads.values())
    assert dict(operations)==EXPECTED_OPERATIONS and dict(reads)==EXPECTED_READ_COSTS
    return score

if __name__=='__main__':
    print(f'Verified score={verify():,}; all 256 outputs match exact integer polynomials.')
    print(f'SHA-256: {EXPECTED_SHA256}')
    print(f'Operations: {EXPECTED_OPERATIONS}')
    print(f'Read costs: {EXPECTED_READ_COSTS}')
