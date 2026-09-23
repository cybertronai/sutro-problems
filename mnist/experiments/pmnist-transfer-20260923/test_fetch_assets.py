"""Offline tests for asset integrity, safe extraction, and no-op defaults."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import fetch_assets as assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.root = self.folder / "study"
        self.root.mkdir()

    def bundle(self, members, expected=None):
        """members: [(name, bytes or TarInfo)]; expected: {path: content}."""
        archive = self.folder / "checkpoints.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name, value in members:
                if isinstance(value, tarfile.TarInfo):
                    output.addfile(value)
                else:
                    header = tarfile.TarInfo(name)
                    header.size = len(value)
                    output.addfile(header, io.BytesIO(value))
        if expected is None:
            expected = {name: value for name, value in members if isinstance(value, bytes)}
        spec = {"url": archive.as_uri(), "sha256": assets.file_hash(archive),
                "bytes": archive.stat().st_size,
                "members": [{"path": path, "sha256": hashlib.sha256(content).hexdigest(),
                             "bytes": len(content)} for path, content in expected.items()]}
        return archive, spec

    def test_no_arguments_has_no_side_effects(self):
        with patch.object(assets, "fetch_transfer") as transfer, \
             patch.object(assets, "fetch_mnist") as mnist, \
             patch.object(assets, "fetch_checkpoints") as checkpoints, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(assets.main([]), 0)
        self.assertIn("--data", output.getvalue())
        transfer.assert_not_called()
        mnist.assert_not_called()
        checkpoints.assert_not_called()

    def test_extracts_only_verified_files_and_is_idempotent(self):
        contents = {"results/all/official/mnist.pt": b"first-checkpoint",
                    "results/all/transfer/kmnist/draw00.pt": b"second-checkpoint"}
        archive, spec = self.bundle(list(contents.items()))
        self.assertEqual(assets.extract_checkpoints(archive, self.root, spec), 2)
        for path, value in contents.items():
            self.assertEqual((self.root / path).read_bytes(), value)
        self.assertEqual(assets.extract_checkpoints(archive, self.root, spec), 0)
        self.assertFalse(list(self.root.glob(".checkpoint-*")))

    def test_archive_hash_or_size_failure_installs_nothing(self):
        archive, spec = self.bundle([("results/model.pt", b"payload")])
        for change in ({"sha256": "0" * 64}, {"bytes": spec["bytes"] + 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                assets.extract_checkpoints(archive, self.root, dict(spec, **change))
            self.assertFalse((self.root / "results").exists())

    def test_member_hash_or_size_failure_installs_nothing(self):
        archive, spec = self.bundle([("results/good.pt", b"first"),
                                     ("results/bad.pt", b"second")])
        spec["members"][1]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            assets.extract_checkpoints(archive, self.root, spec)
        self.assertFalse((self.root / "results").exists())

    def test_extra_missing_and_duplicate_members_are_rejected(self):
        scenarios = [
            ([("results/a.pt", b"a"), ("results/extra.pt", b"x")], {"results/a.pt": b"a"}),
            ([("results/a.pt", b"a")], {"results/a.pt": b"a", "results/b.pt": b"b"}),
            ([("results/a.pt", b"a"), ("results/a.pt", b"a")], {"results/a.pt": b"a"}),
        ]
        for members, expected in scenarios:
            with self.subTest(members=members):
                archive, spec = self.bundle(members, expected)
                with self.assertRaises(ValueError):
                    assets.extract_checkpoints(archive, self.root, spec)
                self.assertFalse((self.root / "results").exists())

    def test_archive_traversal_and_links_are_rejected(self):
        for path in ("../escape.pt", "/absolute.pt", "results/../escape.pt",
                     "results\\escape.pt", "./results/a.pt"):
            with self.subTest(path=path):
                archive, spec = self.bundle([(path, b"bad")], {"results/a.pt": b"a"})
                with self.assertRaises(ValueError):
                    assets.extract_checkpoints(archive, self.root, spec)
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                header = tarfile.TarInfo("results/a.pt")
                header.type, header.linkname = kind, "../../escape.pt"
                archive, spec = self.bundle([("results/a.pt", header)], {"results/a.pt": b"a"})
                with self.assertRaises(ValueError):
                    assets.extract_checkpoints(archive, self.root, spec)
        self.assertFalse((self.folder / "escape.pt").exists())
        self.assertFalse((self.root / "results").exists())

    def test_existing_conflict_is_preserved_and_other_files_not_installed(self):
        archive, spec = self.bundle([("results/new.pt", b"new"),
                                     ("results/existing.pt", b"expected")])
        target = self.root / "results/existing.pt"
        target.parent.mkdir()
        target.write_bytes(b"local-user-content")
        with self.assertRaises(ValueError):
            assets.extract_checkpoints(archive, self.root, spec)
        self.assertEqual(target.read_bytes(), b"local-user-content")
        self.assertFalse((self.root / "results/new.pt").exists())

    def test_existing_destination_symlink_is_rejected(self):
        archive, spec = self.bundle([("results/model.pt", b"payload")])
        outside = self.folder / "outside"
        outside.mkdir()
        (self.root / "results").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            assets.extract_checkpoints(archive, self.root, spec)
        self.assertFalse(list(outside.iterdir()))

    def test_checkpoint_fetch_skips_network_if_all_members_are_verified(self):
        archive, spec = self.bundle([("results/model.pt", b"payload")])
        assets.extract_checkpoints(archive, self.root, spec)
        (self.root / "assets-manifest.json").write_text(json.dumps({"schema_version": 1,
                                                                  "checkpoints": spec}))
        with patch.object(assets, "download_verified") as download, redirect_stdout(io.StringIO()):
            assets.fetch_checkpoints(self.root)
        download.assert_not_called()

    def test_checkpoint_fetch_rejects_existing_conflict_before_network(self):
        archive, spec = self.bundle([("results/model.pt", b"payload")])
        target = self.root / "results/model.pt"
        target.parent.mkdir()
        target.write_bytes(b"wrong")
        (self.root / "assets-manifest.json").write_text(json.dumps({"checkpoints": spec}))
        with patch.object(assets, "download_verified") as download, self.assertRaises(ValueError):
            assets.fetch_checkpoints(self.root)
        download.assert_not_called()

    def test_download_checks_hash_and_never_replaces_a_conflict(self):
        target = self.root / "asset.gz"
        content = b"test-download-content"
        expected = hashlib.sha256(content).hexdigest()
        with patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(content)), \
             redirect_stdout(io.StringIO()):
            assets.download_verified("https://example.test/asset.gz", target, expected,
                                     expected_bytes=len(content))
        self.assertEqual(target.read_bytes(), content)
        with patch.object(assets.urllib.request, "urlopen") as network, self.assertRaises(ValueError):
            assets.download_verified("https://example.test/asset.gz", target, "0" * 64)
        network.assert_not_called()
        self.assertEqual(target.read_bytes(), content)
        bad = self.root / "bad.gz"
        with patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(content)), \
             self.assertRaises(ValueError):
            assets.download_verified("https://example.test/bad.gz", bad, "0" * 64)
        self.assertFalse(bad.exists())
        self.assertFalse(list(self.root.glob(".download-*")))

    def test_data_contract_uses_all_four_mnist_checksums_and_five_pools(self):
        self.assertEqual(len(assets.MNIST_MD5), 4)
        self.assertEqual(assets.MNIST_MD5["t10k-labels-idx1-ubyte.gz"],
                         "ec29112dd5afa0611ce80d1b7f02629c")
        folder = self.root / "data/mnist"
        with patch.object(assets, "download_verified") as download:
            assets.fetch_mnist(folder)
        self.assertEqual(download.call_count, 4)
        for call, (filename, md5) in zip(download.call_args_list, assets.MNIST_MD5.items()):
            self.assertEqual(call.args, (assets.MNIST_URL + filename, folder / filename, md5))
            self.assertEqual(call.kwargs, {"algorithm": "md5"})
        suite = self.root / "vendor/aminist21"
        (suite / "datasets").mkdir(parents=True)
        (suite / "datasets/manifest.json").write_text("{}")
        module = SimpleNamespace(__file__=str(suite / "aminist21/data.py"),
                                 manifest=lambda: {"datasets": [{"id": name} for name in assets.TRANSFER_NAMES]})
        with patch.object(module, "fetch", create=True) as fetch, \
             patch.object(assets.importlib, "import_module", return_value=module), \
             patch.object(assets.sys, "path", list(assets.sys.path)):
            assets.fetch_transfer(suite)
        fetch.assert_called_once_with(suite.resolve() / "data", names=assets.TRANSFER_NAMES)


if __name__ == "__main__":
    unittest.main()
