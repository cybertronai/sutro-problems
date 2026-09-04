"""Exact-output and frozen-artifact contracts for the 64,431 submission."""
import unittest
from unittest.mock import patch
from matmul.submissions import best_64431


class Record64431Tests(unittest.TestCase):
    def test_exact_outputs_cost_and_artifact(self):
        # Two exact polynomial representations check all 256 outputs. The
        # verifier also checks paid source/exit reads and the frozen hash.
        self.assertEqual(best_64431.verify(), 64431)

    def test_hash_gate_rejects_changed_artifact(self):
        changed = best_64431.generate_best_64431() + '\n'
        self.assertEqual(best_64431.score_16x16(changed), 64431)
        with patch.object(best_64431, 'generate_best_64431', return_value=changed):
            with self.assertRaises(AssertionError):
                best_64431.verify()


if __name__ == '__main__':
    unittest.main()
