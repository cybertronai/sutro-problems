"""Fetch checksum-verified review assets. Never launches training or cloud jobs.

From a clean clone, install requirements-review.txt, then run:
    python fetch_assets.py --data --checkpoints
No arguments prints help and performs no downloads. Large files stay untracked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import stat
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
TRANSFER_NAMES = ("kmnist", "emnist_letters_aj", "qmnist_recovered",
                  "fashion_mnist", "cifar10")
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BLOCK_SIZE = 1024 * 1024


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str, algorithm: str = "sha256",
                 expected_bytes: int | None = None) -> bool:
    """Return False if absent; fail rather than replacing conflicting files."""
    if path.is_symlink():
        raise ValueError(f"Refusing a symbolic-link destination: {path}")
    if not path.exists():
        return False
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Destination is not a regular file: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Size mismatch for {path}; existing file was not changed")
    if file_hash(path, algorithm) != expected:
        raise ValueError(f"Checksum mismatch for {path}; existing file was not changed")
    return True


def download_verified(url: str, target: Path, expected: str, *,
                      algorithm: str = "sha256",
                      expected_bytes: int | None = None) -> Path:
    """Stream into a temporary file, check it, and atomically install if absent."""
    target = Path(target)
    if _verify_file(target, expected, algorithm, expected_bytes):
        print(f"Verified {target.name}", flush=True)
        return target
    import certifi
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pmnist-review/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".download-", dir=target.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                size = 0
                for block in iter(lambda: response.read(BLOCK_SIZE), b""):
                    size += len(block)
                    if expected_bytes is not None and size > expected_bytes:
                        raise ValueError(f"Download exceeds the manifest size: {target.name}")
                    stream.write(block)
        _verify_file(temporary, expected, algorithm, expected_bytes)
        # Unlike replace/rename, link refuses to overwrite a file created meanwhile.
        try:
            os.link(temporary, target)
        except FileExistsError:
            _verify_file(target, expected, algorithm, expected_bytes)
        print(f"Downloaded and verified {target.name}", flush=True)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_mnist(folder: Path) -> None:
    for filename, expected_md5 in MNIST_MD5.items():
        download_verified(MNIST_URL + filename, Path(folder) / filename,
                          expected_md5, algorithm="md5")


def fetch_transfer(suite_root: Path) -> None:
    """Reuse the bundled suite's release-manifest and SHA-256 checks."""
    suite_root = Path(suite_root).resolve()
    if not (suite_root / "datasets/manifest.json").is_file():
        raise ValueError(f"Bundled dataset manifest not found under {suite_root}")
    sys.path.insert(0, str(suite_root))
    module = importlib.import_module("aminist21.data")
    if not Path(module.__file__).resolve().is_relative_to(suite_root):
        raise ValueError("A different aminist21 package is already imported; use a fresh process")
    available = {row["id"] for row in module.manifest()["datasets"]}
    if not set(TRANSFER_NAMES).issubset(available):
        raise ValueError("Dataset manifest does not include all five study tasks")
    module.fetch(suite_root / "data", names=TRANSFER_NAMES)


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path


def _safe_destination(root: Path, relative: str) -> Path:
    path = _relative_path(relative)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing a symbolic link in the destination: {current}")
        if current != root / path and current.exists() and not current.is_dir():
            raise ValueError(f"Destination parent is not a directory: {current}")
    return root / path


def _checkpoint_spec(manifest: dict) -> dict:
    if manifest.get("schema_version", 1) != 1:
        raise ValueError("Unsupported asset-manifest schema")
    spec = manifest["checkpoints"]
    if not isinstance(spec.get("bytes"), int) or spec["bytes"] <= 0:
        raise ValueError("Checkpoint archive requires a positive byte count")
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
        raise ValueError("Invalid checkpoint archive SHA-256")
    members = {}
    for member in spec["members"]:
        path = member["path"]
        if _relative_path(path).suffix != ".pt" or path in members:
            raise ValueError(f"Invalid or duplicate checkpoint path: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", member.get("sha256", "")):
            raise ValueError(f"Invalid member SHA-256: {path}")
        if "bytes" in member and (not isinstance(member["bytes"], int) or member["bytes"] < 0):
            raise ValueError(f"Invalid member byte count: {path}")
        members[path] = member
    if not members:
        raise ValueError("Checkpoint manifest has no members")
    return dict(spec, members_by_path=members)


def extract_checkpoints(archive: Path, root: Path, spec: dict) -> int:
    """Validate the complete archive in staging before installing listed files.

Never calls tarfile.extract/extractall. Rejects extra/missing/duplicate members,
links, traversal, nonregular files, and conflicting existing destinations.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = _checkpoint_spec({"checkpoints": spec})
    _verify_file(Path(archive), spec["sha256"], expected_bytes=spec["bytes"])
    expected = spec["members_by_path"]
    directories = {str(parent) for name in expected for parent in
                   PurePosixPath(name).parents if str(parent) != "."}
    for name, entry in expected.items():
        _verify_file(_safe_destination(root, name), entry["sha256"],
                     expected_bytes=entry.get("bytes"))
    with tempfile.TemporaryDirectory(prefix=".checkpoint-stage-", dir=root) as temporary:
        staging = Path(temporary)
        seen, seen_directories = set(), set()
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                name = member.name
                _relative_path(name)
                if member.isdir():
                    if name not in directories or name in seen_directories:
                        raise ValueError(f"Unexpected or duplicate archive directory: {name}")
                    seen_directories.add(name)
                    continue
                if not member.isfile() or member.issparse():
                    raise ValueError(f"Archive member is not a plain regular file: {name}")
                if name not in expected or name in seen:
                    raise ValueError(f"Unexpected or duplicate archive member: {name}")
                seen.add(name)
                entry = expected[name]
                if "bytes" in entry and member.size != entry["bytes"]:
                    raise ValueError(f"Archive member size mismatch: {name}")
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Archive member cannot be read: {name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, BLOCK_SIZE)
                _verify_file(target, entry["sha256"], expected_bytes=entry.get("bytes"))
        if seen != set(expected):
            raise ValueError(f"Archive is missing members: {sorted(set(expected) - seen)}")
        # Recheck every destination before installing any staged checkpoint.
        absent = []
        for name, entry in expected.items():
            destination = _safe_destination(root, name)
            if not _verify_file(destination, entry["sha256"], expected_bytes=entry.get("bytes")):
                absent.append(name)
        for name in absent:
            destination = _safe_destination(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging / name, destination)
            except FileExistsError:
                entry = expected[name]
                _verify_file(destination, entry["sha256"], expected_bytes=entry.get("bytes"))
        return len(absent)


def fetch_checkpoints(root: Path = ROOT, manifest_path: Path | None = None) -> None:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "assets-manifest.json"
    spec = _checkpoint_spec(json.loads(manifest_path.read_text()))
    complete = True
    for name, entry in spec["members_by_path"].items():
        exists = _verify_file(_safe_destination(root, name), entry["sha256"],
                              expected_bytes=entry.get("bytes"))
        complete = complete and exists
    if complete:
        print(f"Verified all {len(spec['members_by_path'])} existing checkpoints", flush=True)
        return
    # Keep the archive only while restoring it; the extracted models remain.
    with tempfile.TemporaryDirectory(prefix=".checkpoint-download-", dir=root) as temporary:
        archive = download_verified(spec["url"], Path(temporary) / "checkpoints.tar.gz",
                                    spec["sha256"], expected_bytes=spec["bytes"])
        count = extract_checkpoints(archive, root, spec)
    print(f"Restored {count} checkpoints; all manifest hashes verified", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="store_true",
                        help="Download/verify the five transfer pools and four canonical MNIST files")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Download/verify the checkpoint release and safely restore its files")
    parser.add_argument("--suite-root", type=Path,
                        default=Path(os.environ.get("AMINIST21_ROOT", ROOT / "vendor/aminist21")),
                        help="Bundled suite root (default: AMINIST21_ROOT or vendor/aminist21)")
    parser.add_argument("--mnist-dir", type=Path,
                        default=Path(os.environ.get("PMNIST_MNIST_DIR", ROOT / "data/mnist")),
                        help="MNIST destination (default: PMNIST_MNIST_DIR or data/mnist)")
    args = parser.parse_args(argv)
    if not args.data and not args.checkpoints:
        parser.print_help()
        return 0
    if args.data:
        fetch_transfer(args.suite_root)
        fetch_mnist(args.mnist_dir)
    if args.checkpoints:
        fetch_checkpoints()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
        raise SystemExit(f"Asset download failed: {exc}") from exc
