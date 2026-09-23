"""Pinned official download sources and checked IDX / HTTP range readers."""
from __future__ import annotations
import gzip, io, json, struct, urllib.request, zipfile
from pathlib import Path
import numpy as np
from .common import file_hash

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_MEMBERS = tuple(
    f"gzip/emnist-letters-{split}-{kind}-idx{dim}-{dtype}.gz"
    for split in ("train", "test")
    for kind, dim, dtype in (("images", 3, "ubyte"), ("labels", 1, "ubyte"))
)
KMNIST_BASE = "https://codh.rois.ac.jp/kmnist/dataset/kmnist/"
QMNIST_BASE = "https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
USPS_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
SOURCES = {
    **{
        f"kmnist-{split}-{kind}.npz": {"url": KMNIST_BASE + f"kmnist-{split}-{kind}.npz"}
        for split in ("train", "test") for kind in ("imgs", "labels")
    },
    "usps.bz2": {"url": USPS_BASE + "usps.bz2", "md5": "ec16c51db3855ca6c91edd34d0e9b197"},
    "usps.t.bz2": {"url": USPS_BASE + "usps.t.bz2", "md5": "8ea070ee2aca1ac39742fdd1ef5ed118"},
    "qmnist-test-images-idx3-ubyte.gz": {
        "url": QMNIST_BASE + "qmnist-test-images-idx3-ubyte.gz",
        "md5": "1394631089c404de565df7b7aeaf9412",
    },
    "qmnist-test-labels-idx2-int.gz": {
        "url": QMNIST_BASE + "qmnist-test-labels-idx2-int.gz",
        "md5": "5b5b05890a5e13444e108efe57b788aa",
    },
}
# Content hashes are pinned after the initial download from the above official
# sources. For EMNIST these identify extracted gzip members, not the full ZIP.
PINNED_SHA256 = {
    "emnist-letters-test-images-idx3-ubyte.gz": "50f0e93d75b99b463e9760b8345866f22587fa2c2fba472730a5ba09c693412d",
    "emnist-letters-test-labels-idx1-ubyte.gz": "924955c31fe7fd809b7303b5c0141282955f4b8390c74cdfd5d80435e0f4b4cb",
    "emnist-letters-train-images-idx3-ubyte.gz": "6288f418a917e007bf5fddb823a2dc7a5c6a35031401b93292a539cb12bd881f",
    "emnist-letters-train-labels-idx1-ubyte.gz": "9759af91ea6bccdf07a6040fd085861f2ec88e37f7c3f5760f6feb57106fbe5c",
    "kmnist-test-imgs.npz": "692d824896bec8f3c357ebf7a0fe904ed67ac632ce953d0b58f7fbef7bb27d87",
    "kmnist-test-labels.npz": "660f13a4705431ea2a6391c41ecbe384cd0d932c595a101ec4f3b8989e785046",
    "kmnist-train-imgs.npz": "11d69e17c8fd294905864aef177fd10c2185e5a3655f4da03d72ea5a64c17a72",
    "kmnist-train-labels.npz": "a325c877035602cb9b9ba54f4194a2a1b5f718026a463db5b5e0cdc124c03720",
    "qmnist-test-images-idx3-ubyte.gz": "43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
    "qmnist-test-labels-idx2-int.gz": "9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
    "usps.bz2": "3771e9dd6ba685185f89867b6e249233dd74652389f263963b3b741e994b034f",
    "usps.t.bz2": "a9c0164e797d60142a50604917f0baa604f326e9a689698763793fa5d12ffc4e",
}
EXPECTED_COUNTS = {
    "mnist_medium": {"train": 10000, "test": 10000},
    "mnist_official": {"test": 10000},
    "kmnist": {"train": 60000, "test": 10000},
    "emnist_letters_aj": {"train": 48000, "test": 8000},
    "usps": {"train": 7291, "test": 2007},
    "qmnist_test10k": {"test": 10000},
    "qmnist_test50k": {"test": 50000},
}


def _verify(path: Path, source: dict) -> None:
    expected_md5 = source.get("md5")
    if expected_md5 and file_hash(path, "md5") != expected_md5:
        raise ValueError(f"Invalid source MD5: {path}")
    expected_sha = PINNED_SHA256.get(path.name)
    if expected_sha and file_hash(path) != expected_sha:
        raise ValueError(f"Invalid source SHA256: {path}")


def _download(raw: Path, filename: str) -> Path:
    source = SOURCES[filename]
    destination = raw / filename
    if destination.exists():
        _verify(destination, source)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {filename}", flush=True)
    try:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "sutro-transfer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        # Check the expected filename's hashes while content still has .part.
        if source.get("md5") and file_hash(temporary, "md5") != source["md5"]:
            raise ValueError(f"Invalid source MD5: {filename}")
        if PINNED_SHA256.get(filename) and file_hash(temporary) != PINNED_SHA256[filename]:
            raise ValueError(f"Invalid source SHA256: {filename}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class _HTTPRangeReader(io.RawIOBase):
    """Seekable, range-checked input so only selected EMNIST members transfer."""

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "sutro-transfer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith("bytes 0-0/"):
                raise ValueError("EMNIST server must support HTTP byte ranges")
            self.size = int(content_range.split("/")[-1])
            self.etag = response.headers.get("ETag")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        if stop <= self.position:
            return b""
        expected_range = f"bytes {self.position}-{stop - 1}/{self.size}"
        headers = {"Range": f"bytes={self.position}-{stop - 1}", "User-Agent": "sutro-transfer/1.0"}
        if self.etag:
            headers["If-Match"] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Incorrect HTTP range response from EMNIST server")
            data = response.read()
        if len(data) != stop - self.position:
            raise ValueError("Truncated EMNIST range")
        self.position = stop
        return data


def _download_emnist(raw: Path) -> None:
    missing = [member for member in EMNIST_MEMBERS if not (raw / Path(member).name).exists()]
    if missing:
        print("Downloading the four EMNIST Letters members using official ZIP byte ranges", flush=True)
        with _HTTPRangeReader(EMNIST_URL) as remote, zipfile.ZipFile(remote) as archive:
            archive_info = {"url": EMNIST_URL, "bytes": remote.size, "etag": remote.etag, "members": {}}
            for member in EMNIST_MEMBERS:
                info = archive.getinfo(member)
                archive_info["members"][member] = {"crc32": f"{info.CRC:08x}", "bytes": info.file_size}
                if member in missing:
                    path = raw / Path(member).name
                    temporary = path.with_suffix(path.suffix + ".part")
                    try:
                        # ZipFile verifies the member's CRC32 while decompressing.
                        temporary.write_bytes(archive.read(member))
                        if PINNED_SHA256.get(path.name) and file_hash(temporary) != PINNED_SHA256[path.name]:
                            raise ValueError(f"Invalid EMNIST SHA256: {member}")
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            (raw / "emnist-archive.json").write_text(json.dumps(archive_info, indent=2) + "\n")
    for member in EMNIST_MEMBERS:
        _verify(raw / Path(member).name, {})


def _idx(path: Path) -> np.ndarray:
    data = gzip.decompress(path.read_bytes())
    if len(data) < 8 or data[:2] != b"\0\0" or data[2] not in (8, 12) or not 1 <= data[3] <= 3:
        raise ValueError(f"Invalid IDX header: {path}")
    dtype = np.dtype("u1" if data[2] == 8 else ">i4")
    ndim = data[3]
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    array = np.frombuffer(data, dtype=dtype, offset=4 + 4 * ndim)
    if array.size != np.prod(shape):
        raise ValueError(f"Invalid IDX length: {path}")
    return array.reshape(shape)

