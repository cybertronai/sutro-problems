"""Exact-output, score, and frozen-artifact contracts for the 63,847 record."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from matmul.submissions import best_63847 as record


class Record63847Tests(unittest.TestCase):
    def test_exact_outputs_cost_and_artifact(self):
        self.assertEqual(record.verify(), 63847)

    def test_hash_gate_rejects_changed_artifact(self):
        changed = record.IR_PATH.read_bytes() + b'\n'
        self.assertEqual(record.score_16x16(changed.decode()), 63847)
        with patch.object(record.Path, 'read_bytes', return_value=changed):
            with self.assertRaisesRegex(AssertionError, 'SHA-256'):
                record.verify()

    def test_official_score_gate(self):
        with patch.object(record, 'score_16x16', return_value=63848):
            with self.assertRaisesRegex(AssertionError, 'score mismatch'):
                record.verify()

    def test_independent_proof_gates(self):
        counts = dict(record.EXPECTED_OPERATIONS)
        counts['copy'] += 1
        costs = dict(record.EXPECTED_READ_COSTS)
        costs['copy'] += 1
        redistributed = dict(costs)
        redistributed['mul'] -= 1
        for operations, read_costs, message in [
            (counts, record.EXPECTED_READ_COSTS, 'operation counts'),
            (record.EXPECTED_OPERATIONS, costs, 'score mismatch'),
            (record.EXPECTED_OPERATIONS, redistributed, 'read-cost breakdown'),
        ]:
            with self.subTest(message=message):
                with patch.object(record, '_prove', return_value=(operations, read_costs)):
                    with self.assertRaisesRegex(AssertionError, message):
                        record.verify()

    def test_cli_without_site_packages_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-S', str(record.IR_PATH.with_suffix('.py'))],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        self.assertIn('score=63,847', result.stdout)
        self.assertIn(record.EXPECTED_SHA256, result.stdout)


if __name__ == '__main__':
    unittest.main()
