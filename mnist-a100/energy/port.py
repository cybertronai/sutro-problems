"""Port the cutoff study's MLPs to the mnist-a100 API, unchanged but for the signature.

    python energy/port.py      # from mnist-a100/: writes energy/entries/<configuration>.py

The study's files (mnist/experiments/release-cutoffs-20260925/mlp_timing/submissions) are
popcorn3 entries: `custom_kernel(data)` with `train_x, train_y, test_x = data`. Each port drops
the two #!POPCORN lines and takes the three tensors as arguments of `mlp`; nothing else changes.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1] / "mnist/experiments/release-cutoffs-20260925/mlp_timing/submissions"
PORTS = ("mlp-k1-w1024-s100-b512",      # eager, the study's pick for 200 labels (difficulty 1)
         "mlpg-k4-w256-s800-b512",      # graph-captured, its pick for 532 labels (difficulty 2)
         "mlp-k16-w1024-s400-b512")     # eager, its pick for 532 labels (difficulty 2)
HEADER = "def custom_kernel(data):\n    train_x, train_y, test_x = data\n"


def port(text):
    lines = [line for line in text.splitlines(keepends=True) if not line.startswith("#!POPCORN")]
    body = "".join(lines).lstrip("\n")
    if body.count(HEADER) != 1:
        raise ValueError("expected exactly one custom_kernel(data) header")
    return body.replace(HEADER, "def mlp(train_x, train_y, test_x):\n")


def main():
    for name in PORTS:
        target = HERE / "entries" / (name.replace("-", "_") + ".py")
        target.write_text(port((STUDY / f"{name}.py").read_text()))
        print(f"{target.relative_to(HERE.parent)}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
