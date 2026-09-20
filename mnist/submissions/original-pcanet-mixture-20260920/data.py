"""Verified official MNIST in original IDX order; no shuffled tier cache."""
import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request

import numpy as np

SOURCES = (
    ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873', 60000, True),
    ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432', 60000, False),
    ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3', 10000, True),
    ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c', 10000, False),
)


def official(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, manifest = [], {}
    for name, expected, count, images in SOURCES:
        path = directory / name
        if not path.exists():
            urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/' + name, path)
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != expected:
            raise ValueError('MNIST checksum mismatch: ' + name)
        payload = gzip.decompress(raw)
        header = (2051, count, 28, 28) if images else (2049, count)
        size = 4 * len(header)
        assert struct.unpack('>' + 'I' * len(header), payload[:size]) == header
        assert len(payload) == size + count * (784 if images else 1)
        a = np.frombuffer(payload, dtype=np.uint8, offset=size)
        a = a.reshape(count, 28, 28).astype(np.float32) / np.float32(255) if images else a.astype(np.int64)
        arrays.append(a)
        manifest[name] = {'md5': expected, 'sha256': hashlib.sha256(raw).hexdigest(),
                          'array_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                          'shape': list(a.shape), 'dtype': str(a.dtype)}
    return arrays, manifest
