"""Check portable dependency pinning without training, downloads or paid jobs."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import reproduce

HERE = Path(__file__).resolve().parent


class ReproductionTests(unittest.TestCase):
    def test_original_protocol_and_source_hashes(self):
        protocol = reproduce.run.check_protocol()
        self.assertEqual(protocol['generator_sha256'], reproduce.GENERATOR_SHA256)
        self.assertEqual(Path(reproduce.run.generator.__file__).name, 'frozen_data.py')

    def test_archived_inputs_are_allowlisted_and_intact(self):
        manifest = json.loads((HERE / 'draw_manifest.json').read_text())
        self.assertEqual(len(manifest['draws']), 11)
        for draw in manifest['draws']:
            path = HERE / draw['archive']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), draw['archive_sha256'])
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(set(z.files), {'train_images','train_labels','test_images'})
                for name in z.files:
                    self.assertEqual(reproduce.frozen_data.array_hash(z[name]),draw['arrays'][name]['sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
