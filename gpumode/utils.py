"""Shared helpers for the MNIST-medium time leaderboard evaluator.

Kept deliberately small: seeding, an A100-sized L2 flush, the integer accuracy
rule, the timer-plausibility gate and the child-process network guard.
"""

from __future__ import annotations

import math
import random

HARNESS_VERSION = "sutro-mnist-medium-time/1.2.0"
# 1.2.0: draws are released as z = Q W (x - mu) -- exact PCA whitening onto the
# top release_dims principal directions of the draw's own training rows, then a
# per-draw secret Haar rotation. See DESIGN.md "Linear release (harness 1.2.0)".

# A100 (40 GB and 80 GB) has a 40 MB L2. Writing 256 MB evicts it several times
# over and costs under a millisecond at ~1.5 TB/s. KernelBot's AMD harness uses
# a 64 GiB buffer because MI355X has 256 MB of LLC; copying that number here
# would add seconds per call and would not fit a 40 GB A100 at all.
L2_FLUSH_BYTES = 256 * 1024 * 1024


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    try:
        import torch

        torch.manual_seed(seed % (2**63))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**63))
    except ImportError:  # pragma: no cover
        pass


def combine(a: int, b: int) -> int:
    """KernelBot's Cantor pairing of a public case seed with the secret seed.

    A large secret seed makes the combined seed large, so the public seed in
    task.yml leaks nothing usable about the draw a submission will receive.
    """
    return int(a + (a + b) * (a + b + 1) // 2)


def required_correct(total: int, error_bp: int) -> int:
    """ceil(total * (10000 - error_bp) / 10000) in exact integer arithmetic."""
    if not 0 <= error_bp <= 10000:
        raise ValueError("error_bp must be between 0 and 10000")
    numerator = total * (10000 - error_bp)
    return -(-numerator // 10000)


def stats(durations_ns: list[float]) -> dict:
    """KernelBot's Stats fields plus a median, all in nanoseconds."""
    runs = len(durations_ns)
    if runs == 0:
        raise ValueError("no durations")
    mean = sum(durations_ns) / runs
    if runs > 1:
        variance = sum((x - mean) ** 2 for x in durations_ns) / (runs - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    ordered = sorted(durations_ns)
    middle = runs // 2
    median = ordered[middle] if runs % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {
        "runs": runs,
        "mean": mean,
        "std": std,
        "err": std / math.sqrt(runs),
        "best": float(min(durations_ns)),
        "worst": float(max(durations_ns)),
        "median": median,
    }


# The parent's clock is the only one a submission cannot reach. A call whose
# reported device time is below this fraction of the parent-measured span (after
# subtracting the calibrated inter-process overhead) is rejected: scaling every
# clock inside the child by a constant no longer helps, because the parent's
# number does not scale with it.
PARENT_CLOCK_FRACTION = 0.25


def timing_plausible(
    device_ms: float,
    child_wall_ms: float,
    parent_wall_ms: float,
    overhead_ms: float = 0.0,
) -> str | None:
    """Return None when the clocks agree, else a human-readable reason.

    Three clocks bracket every call: CUDA events (or ``perf_counter`` on the CPU
    path) inside the child, the child's ``perf_counter``, and the parent's
    ``perf_counter`` around the whole staging-plus-call round trip. Only the
    parent's clock is out of the submission's reach, so it supplies both bounds
    that matter: the child may not claim more time than the parent saw, and it
    may not claim dramatically less either.

    ``overhead_ms`` is the inter-process cost (staging the draw, the pipe round
    trip, reading the predictions back) measured *before* the submission was
    imported, so it cannot be inflated by the submission to create slack.
    """
    if not (device_ms == device_ms) or device_ms < 0:  # NaN or negative
        return f"device time {device_ms!r} is not a positive number"
    if device_ms > child_wall_ms + 0.5:
        return (
            f"device time {device_ms:.3f} ms exceeds the child's wall clock "
            f"{child_wall_ms:.3f} ms"
        )
    # Relative below a millisecond, so a fast call cannot hide half its work in
    # a fixed slack term (reference-kernels#161).
    if device_ms < 0.5 * child_wall_ms - min(0.5, 0.25 * child_wall_ms):
        return (
            f"device time {device_ms:.3f} ms is less than half the child's wall clock "
            f"{child_wall_ms:.3f} ms; work is running outside the timed window"
        )
    if child_wall_ms > parent_wall_ms + 5.0:
        return (
            f"child wall clock {child_wall_ms:.3f} ms exceeds the parent's "
            f"{parent_wall_ms:.3f} ms"
        )
    budget_ms = parent_wall_ms - overhead_ms
    if device_ms < PARENT_CLOCK_FRACTION * budget_ms - 0.5:
        return (
            f"device time {device_ms:.3f} ms is under {PARENT_CLOCK_FRACTION:g} of the "
            f"{budget_ms:.3f} ms the parent measured for this call (round trip "
            f"{parent_wall_ms:.3f} ms minus {overhead_ms:.3f} ms of calibrated overhead); "
            "the clocks inside the submission's process are not trusted"
        )
    return None


class NetworkDisabled(OSError):
    """Raised inside a submission process that tries to reach the network."""


class DatasetFileDenied(OSError):
    """Raised inside a submission process that tries to open a dataset file."""


# Audit events that mean "this process is talking to the network". An audit
# hook cannot be removed once installed and fires inside the C implementation,
# so unlike a monkeypatch it survives ``importlib.reload(socket)`` and reaching
# past ``socket.socket`` to ``_socket.socket``.
NETWORK_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sendto",
        "urllib.Request",
        "ftplib.connect",
        "smtplib.connect",
    }
)

# Filename fragments that only a submission looking for a copy of the public
# dataset on the evaluator's disk would open. A tripwire, not a sandbox.
DATASET_FRAGMENTS = (
    "idx3-ubyte",
    "idx1-ubyte",
    "mnist",
    "fashion",
    "emnist",
    "cifar",
    "t10k",
)


def install_network_guard() -> None:
    """Deny network access and dataset-file reads inside the submission's process.

    Installed before the submission module is imported. Two layers: an audit
    hook (permanent, C-level, covers ``_socket`` and a reloaded ``socket``) and
    the obvious monkeypatches, which give a clearer error message.

    It is still not a sandbox. A subprocess is a fresh interpreter without the
    hook, so ``curl`` in a shell escapes it; a hosted run must also deny egress
    at the container level (``block_network=True`` on the Modal function).
    """
    import sys

    def audit(event, args):
        if event in NETWORK_EVENTS:
            raise NetworkDisabled(f"network access is disabled inside submissions ({event})")
        if event == "open" and args:
            name = args[0]
            if isinstance(name, (str, bytes)):
                text = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
                lowered = text.lower()
                if any(fragment in lowered for fragment in DATASET_FRAGMENTS):
                    raise DatasetFileDenied(
                        f"opening dataset files is not allowed inside submissions: {text}"
                    )

    sys.addaudithook(audit)

    import socket

    def deny(*_args, **_kwargs):
        raise NetworkDisabled("network access is disabled inside submissions")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    try:
        import urllib.request

        urllib.request.urlopen = deny
        urllib.request.urlretrieve = deny
    except ImportError:  # pragma: no cover
        pass


def system_info() -> dict:
    """Versions participants need in order to reproduce a ranked run."""
    info = {"harness": HARNESS_VERSION}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "none"
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            major, minor = torch.cuda.get_device_capability(0)
            info["capability"] = f"{major}.{minor}"
            # _cuda_getDriverVersion needs the lazy CUDA init to have run, and
            # the accessor moved between torch versions; report why it failed
            # rather than a bare "unknown" (D10 promises the driver version).
            try:
                torch.cuda.init()
            except Exception:  # pragma: no cover - CPU dry runs never get here
                pass
            for accessor in (
                lambda: torch.cuda.driver_version(),
                lambda: torch._C._cuda_getDriverVersion(),
            ):
                try:
                    info["driver"] = str(accessor())
                    break
                except Exception as error:  # pragma: no cover - version dependent
                    info["driver"] = f"unknown ({type(error).__name__}: {error})"[:120]
        else:
            info["device"] = "cpu"
            info["device_count"] = 0
    except ImportError:  # pragma: no cover
        info["torch"] = "missing"
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    import platform
    import sys

    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    return info
