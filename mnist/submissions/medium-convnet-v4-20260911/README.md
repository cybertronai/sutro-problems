# MNIST-medium: 4% error target

The complete report is in [report.md](report.md). It records the frozen
three-ConvNet learner, eleven 10,000-train/10,000-test draws, exact scalar-v4
model scores, A100 timing and idle-adjusted NVML energy.

- Accuracy: **97.8% ± 0.1 pp**; error: **2.2% ± 0.1 pp**.
- Theoretical: **4,900,000 ms**, **12,000 mJ**, **8.9 mm²**; scoring: **11 s**.
- A100: **13,000 ms**, **780,000 mJ** per complete training-and-prediction task.

Use `reproduce.py` to prepare a fresh source-only reproduction tree.
The detailed commands, numerical conventions and measurement boundaries are in
the report. `audit.py --skip-input-files --output /tmp/medium-audit.json`
checks published evidence without the regenerable dataset archives.

[Separate ambiguities](ambiguities.md) · [Visible session](session.md)
