"""Frozen record, exact dual, and standalone verification contracts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matmul.submissions import best_63553 as record


class Record63553Tests(unittest.TestCase):
    def test_exact_outputs_score_and_allocation_bound(self):
        result = record.verify()
        self.assertEqual(result["score"], 63553)
        self.assertEqual(result["lower_bound"], "63553")

    def test_hash_gates_reject_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("IR_PATH", "CERTIFICATE_PATH"):
                with self.subTest(artifact=name):
                    changed = Path(directory) / name
                    changed.write_bytes(getattr(record, name).read_bytes() + b"\n")
                    with (
                        patch.object(record, name, changed),
                        self.assertRaisesRegex(AssertionError, "SHA-256"),
                    ):
                        record.verify()

    def test_short_form_reads_and_same_instruction_address_reuse(self):
        intervals, _arithmetic, gaps = record._trace("1,2;mul 3,1,2;add 3,2;3")
        self.assertEqual(intervals, [(-1, 0, 1), (-1, 1, 2), (0, 1, 1), (1, 2, 1)])
        self.assertEqual(gaps, 3)
        unread, _, _ = record._trace("1,2;copy 3,1;3")
        self.assertEqual(unread[1], (-1, 0, 0))

    def test_arithmetic_trace_erases_copies_but_preserves_dependencies(self):
        direct = record._trace("1,2;mul 3,1,2;3")[1]
        copied = record._trace("1,2;copy 4,1;copy 5,4;mul 3,5,2;3")[1]
        changed = record._trace("1,2;add 3,1,2;3")[1]
        self.assertEqual(direct, copied)
        self.assertNotEqual(direct, changed)

    def test_exact_dual_includes_unpriced_tiers_and_rejects_invalid_prices(self):
        intervals, _, gaps = record._trace("1,2;mul 3,1,2;3")
        certificate = {"modeled_tiers": 1, "capacity_prices": [[1, 0, 1, 1]]}
        self.assertEqual(record._dual_bound(intervals, gaps, certificate), 4)
        for rows in (
            [[1, 0, -1, 1]], [[1, 0, 1, 0]], [[2, 0, 1, 1]],
            [[1, gaps, 1, 1]], [[1, 0, 1, 1], [1, 0, 1, 1]],
        ):
            with self.subTest(prices=rows), self.assertRaises(ValueError):
                record._dual_bound(intervals, gaps,
                                   {"modeled_tiers": 1, "capacity_prices": rows})

    def test_cli_without_site_packages_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", str(record.IR_PATH.with_suffix(".py"))],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        self.assertIn("score=63,553", result.stdout)
        self.assertIn(record.EXPECTED_SHA256, result.stdout)
        self.assertIn("exact rational dual = 63,553", result.stdout)


if __name__ == "__main__":
    unittest.main()
