# MNIST-small: higher accuracy and compact scoring

This exploratory study tests 55/60/65/70/75% accuracy goals using a fixed-training-loop FP32 neural network, and implements a compact affine-loop intermediate language for exact v4 cost aggregation. It is a follow-up to the original 1NN submission, with no new MLP A100 measurements yet.

Every seed at 300 epochs and above exceeded 60%. One of 15 runs exceeded 65%; none reached 70% or 75%. Compact scoring remains practical for programs representing billions of primitive instructions. Displayed time uses ps and energy uses fJ, with two significant figures.

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
