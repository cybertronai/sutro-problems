# MNIST-small: higher accuracy and compact scoring

This exploratory study tests 55/60/65/70/75% accuracy goals using a fixed-training-loop FP32 neural network, and implements a compact affine-loop intermediate language for exact v4 cost aggregation. It is a follow-up to the original 1NN submission. The original study measured CPU execution and theoretical costs; it did not perform MLP A100 trials.

The official MNIST-small target is now **60%**, requiring **360/600** correct. Every seed at 300 epochs and above exceeded it. The original 1NN result, **308/600 (51%)**, is below the current requirement. The study meets the accuracy requirement, and its tables preserve the original CPU/model measurement scope. One of 15 runs exceeded 65%; none reached 70% or 75%. Compact scoring remains practical for programs representing billions of primitive instructions. Execution time uses ms, energy uses mJ, area uses mm², and time to score uses s, with two significant figures. Area displays divide the exact native µm² totals by 10⁶; the 1NN baseline is 0.0060 mm² and H32 networks use 0.020 mm².

A separate [H32 / 300 epochs / seed 101 A100 submission](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/) carries the frozen candidate forward and reports its GPU verification and measurement status. A100 entries here remain labeled “Not measured in this study”; other configurations and seeds remain unmeasured on A100.

The model-cost, scoring-time, and CPU-execution comparison tables each include accuracy, exact counts by seed, and the number of seeds meeting the current target.

Read `report.md` for results and reproduction, `IL.md` for language/scorer semantics, `study-protocol.md` for the selection protocol, and `ambiguities.md` for remaining decisions. Exact results and frozen predictions are checked in beside the source.

The HTML renderer reuses the original submission's report style. After changing report prose, rebuild from the repository root:

```bash
S=mnist/experiments/accuracy-il-20260911
python "$S/write_report.py"
python -m pip install -r mnist/submissions/1nn-v4-20260911/requirements-report.txt
python "$S/build_pages.py"
```

Publishing requires committing and pushing the generated `docs/submissions/accuracy-il-20260911/` directory to `main`.

[Published report](https://cybertronai.github.io/sutro-problems/docs/submissions/accuracy-il-20260911/) · [Ambiguities](https://cybertronai.github.io/sutro-problems/docs/submissions/accuracy-il-20260911/ambiguities.html)
