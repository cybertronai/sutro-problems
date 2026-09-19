"""Exact semantics, residence lifetimes, and dual checks for the 670 record."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matmul.submissions import best_670 as record


class Record670Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = record.generate_best_670()
        cls.report, cls.versions = record._prove(cls.ir)
        cls.certificate = json.loads(record.CERTIFICATE_PATH.read_bytes())

    def test_exact_witness_and_certificate(self):
        report = record.verify()
        self.assertEqual(report["score"], 670)
        self.assertEqual(report["operations"], {"mul": 64, "add": 48, "copy": 19})
        self.assertEqual(report["costs"], {"mul": 321, "add": 185, "copy": 95, "output": 69})
        self.assertEqual(report["versions"], 163)
        self.assertEqual(report["max_address"], 37)
        self.assertEqual(report["peak_liveness"], 36)
        self.assertEqual(report["fixed_trace_lower_bound"], "670")
        self.assertEqual(report["dual_inequalities"], 2119)
        self.assertTrue(report["higher_tiers_verified"])

    def test_hash_gates_reject_equivalent_modified_files(self):
        for attribute in ("IR_PATH", "CERTIFICATE_PATH"):
            with self.subTest(artifact=attribute), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "changed"
                path.write_bytes(getattr(record, attribute).read_bytes() + b"\n")
                with patch.object(record, attribute, path):
                    with self.assertRaisesRegex(AssertionError, "SHA-256"):
                        record.verify()
                    if attribute == "IR_PATH":
                        with self.assertRaisesRegex(AssertionError, "SHA-256"):
                            record.generate_best_670()

    def test_official_score_gate(self):
        with patch.object(record, "score_4x4", return_value=671):
            with self.assertRaisesRegex(AssertionError, "score mismatch"):
                record.verify()

    def test_independent_score_gate(self):
        changed = dict(self.report, score=671)
        with patch.object(record, "_prove", return_value=(changed, self.versions)):
            with self.assertRaisesRegex(AssertionError, "score mismatch"):
                record.verify()

    def test_wrong_output_order_and_arithmetic_are_rejected(self):
        lines = self.ir.splitlines()
        # The archived file has a trailing blank line; replace the output line itself.
        output_index = max(index for index, line in enumerate(lines) if line.strip())
        outputs = lines[output_index].split(",")
        outputs[0], outputs[1] = outputs[1], outputs[0]
        lines[output_index] = ",".join(outputs)
        wrong_outputs = "\n".join(lines)
        wrong_arithmetic = self.ir.replace("mul ", "add ", 1)
        for ir in (wrong_outputs, wrong_arithmetic):
            with self.subTest(ir=ir[:80]):
                with self.assertRaisesRegex(AssertionError, "symbolic output mismatch"):
                    record._prove(ir)
                with self.assertRaisesRegex(ValueError, "correctness failed"):
                    record.score_4x4(ir)

    def test_read_before_write_half_open_lifetimes(self):
        report, versions = record._prove("1,2;copy 1,1;mul 1,1,2;1", n=1)
        self.assertEqual(report["score"], 5)
        self.assertEqual(report["peak_liveness"], 2)
        self.assertEqual(versions, [
            record.Version(0, 1, 1), record.Version(0, 2, 1),
            record.Version(1, 2, 1), record.Version(2, 3, 1),
        ])
        # Each source occurrence is a paid read, and an unread temporary is empty.
        _, versions = record._prove("1,2;add 3,1,1;mul 1,1,2;1", n=1)
        self.assertEqual(versions[0], record.Version(0, 2, 3))
        self.assertEqual(versions[2], record.Version(1, 1, 0))

    def test_invalid_dual_sign_constraint_and_bound(self):
        for key, index, value, message in (
            ("beta", 0, "1", "nonpositive"),
            ("alpha", 0, "100000", "dual inequality"),
            ("bound", None, "669", "lower bound"),
        ):
            certificate = copy.deepcopy(self.certificate)
            if index is None:
                certificate[key] = value
            else:
                certificate[key][index] = value
            with self.subTest(key=key), self.assertRaisesRegex(AssertionError, message):
                record.check_certificate(self.versions, certificate, 670)

    def test_invalid_clique_members(self):
        for clique, message in (([0, 0], "duplicate"), ([163], "index"), ([True], "index")):
            certificate = copy.deepcopy(self.certificate)
            certificate["cliques"][0] = clique
            with self.subTest(clique=clique), self.assertRaisesRegex(ValueError, message):
                record.check_certificate(self.versions, certificate, 670)

    def test_touching_lifetimes_are_not_a_clique(self):
        versions = [record.Version(0, 1, 1), record.Version(1, 2, 1)]
        certificate = {
            "tiers": 2, "alpha": ["0", "0"], "beta": ["0", "0"],
            "cliques": [[0, 1]], "bound": "0",
        }
        with self.assertRaisesRegex(AssertionError, "not simultaneously live"):
            record.check_certificate(versions, certificate, 0)

    def test_higher_tiers_checked_even_when_finite_constraints_pass(self):
        versions = [record.Version(0, 1, 0)]
        certificate = {
            "tiers": 1, "alpha": ["1"], "beta": ["-1"],
            "cliques": [[0]], "bound": "0",
        }
        with self.assertRaisesRegex(AssertionError, "higher-tier"):
            record.check_certificate(versions, certificate, 0)

    def test_exact_fraction_dual(self):
        versions = [record.Version(0, 1, 1)]
        certificate = {
            "tiers": 1, "alpha": ["3/2"], "beta": ["-1/2"],
            "cliques": [[0]], "bound": "1",
        }
        report = record.check_certificate(versions, certificate, 1)
        self.assertEqual(report["fixed_trace_lower_bound"], "1")

    def test_cli_without_site_packages_or_assertions_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-O", "-S", str(record.IR_PATH.with_suffix(".py"))],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        self.assertIn("score=670", result.stdout)
        self.assertIn(record.EXPECTED_SHA256, result.stdout)
        self.assertIn('"fixed_trace_lower_bound": "670"', result.stdout)
        self.assertIn("not a lower bound for all programs", result.stdout)


if __name__ == "__main__":
    unittest.main()
