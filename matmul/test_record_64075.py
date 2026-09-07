"""Exact correctness, byte-exact reproduction, and capture semantics."""
import unittest
from unittest.mock import patch

from matmul.submissions import best_64075 as record


class Record64075Tests(unittest.TestCase):
    def test_exact_outputs_cost_artifact_and_replay(self):
        self.assertEqual(record.verify(), 64075)

    def test_hash_gate_rejects_changed_artifact(self):
        changed = record.IR_PATH.read_bytes() + b'\n'
        self.assertEqual(record.score_16x16(changed.decode()), 64075)
        with patch.object(record.Path, 'read_bytes', return_value=changed):
            with self.assertRaises(AssertionError):
                record.verify()

    def test_replay_gate_rejects_changed_reproduction(self):
        changed = record.generate_best_64075() + '\n'
        with patch.object(record, 'generate_best_64075', return_value=changed):
            with self.assertRaises(AssertionError):
                record.verify()

    def test_original_logical_instructions_are_preserved(self):
        base = record.IR_PATH.with_name('best_64431.ir').read_text()
        inputs, steps, outputs = record.decode(base)
        new_inputs, new_steps, new_outputs = record.decode(record.generate_best_64075())
        self.assertEqual(inputs, new_inputs)
        self.assertEqual(outputs, new_outputs)
        position = 0
        captures = 0
        for step in new_steps:
            if position < len(steps) and step == steps[position]:
                position += 1
            else:
                op, _, values, value = step
                self.assertEqual(op, 'copy')
                self.assertEqual(values, (value,))
                self.assertLess(value, 512)
                captures += 1
        self.assertEqual(position, len(steps))
        self.assertEqual(captures, 245)

    def test_decode_read_before_write_and_copy_identity(self):
        _, steps, outputs = record.decode('1,2;copy 3,1;mul 1,1,2;add 1,3;1')
        self.assertEqual(steps, [
            ('copy', 3, (0,), 0),
            ('mul', 1, (0, 1), 2),
            ('add', 1, (2, 0), 3),
        ])
        self.assertEqual(outputs, [3])


if __name__ == '__main__':
    unittest.main()
