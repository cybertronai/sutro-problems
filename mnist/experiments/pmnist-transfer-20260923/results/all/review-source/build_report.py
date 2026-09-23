"""Build the study summary solely from completed, audited experiment records."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {"kmnist":"KMNIST", "emnist_letters_aj":"EMNIST letters A–J",
          "qmnist_recovered":"QMNIST recovered", "fashion_mnist":"Fashion-MNIST",
          "cifar10":"CIFAR-10"}


def read(path):
    return json.loads(path.read_text())


def main():
    result = ROOT/"results/all"
    original = read(result/"official/scores.json")
    transfer = read(result/"transfer/scores.json")
    comparison = read(result/"comparison/comparison.json")
    execution = read(result/"execution.json")
    records = [read(result/"official/mnist.json")]
    records += [read(result/"transfer"/row["metadata_path"])
                for row in read(result/"transfer/prediction-manifest.json")["entries"]]
    seconds = sum(row["adapter_wall_seconds"] for row in records)
    train_seconds = sum(row["metadata"]["training_seconds"] for row in records)
    # Workers requested A100-40GB but Modal can provision 80GB parts. Use the
    # higher 80GB published rate for this conservative report, without altering
    # the immutable launch record's requested-resource estimate.
    rate = 0.000694 + 4*0.0000131 + 8*0.00000222
    gpu_names = sorted({row["hardware"]["gpu"] for row in records})
    elapsed = execution["elapsed_seconds"]
    (result/"cost-estimate.json").write_text(json.dumps({
        "requested_gpu":"A100-40GB", "reported_gpu_names":gpu_names,
        "conservative_rate_usd_per_worker_second":rate,
        "pricing_basis":"A100-80GB GPU + four physical CPU cores + 8 GiB RAM",
        "source":"https://modal.com/pricing", "checked_date":"2026-09-23",
        "adapter_seconds":seconds, "app_seconds":elapsed,
        "adapter_time_estimate_usd":seconds*rate,
        "four_workers_full_app_time_estimate_usd":elapsed*4*rate,
        "caveat":"Conservative resource estimates, not a provider invoice. Launch record used requested 40GB GPU rate. Image builds and smoke separate."
    },indent=2)+"\n")
    table = []
    for row in comparison["datasets"]:
        new = row["candidate"]
        dense = row["references"]["mlp256-sgd"]
        conv = row["references"]["cnn32-ensemble3"]
        table.append(f"| {LABELS[row['dataset']]} | {new['mean_accuracy_percent']:.2f} ± {new['sample_sd_pp']:.2f} | "
                     f"{dense['mean_accuracy_percent']:.2f} | {conv['mean_accuracy_percent']:.2f} | "
                     f"{dense['candidate_minus_reference_mean_pp']:+.2f} |")
    text = f"""# Fixed-permutation MNIST and transfer of the Ladder training procedure

Research and experiment date: 2026-09-23. Results are local; this report has not been published or pushed.

## Main findings

The strongest eligible published result verified in this search is **99.431% mean accuracy** for **Ladder AMLP [2,2]** (Pezeshki et al., ICML 2016), corresponding to 0.569% error with standard error 0.010 percentage points over ten runs. This is effectively tied with the original Ladder result of 99.43%. This is the best source-supported result found, not a claim that an exhaustive, continuously updated leaderboard exists. [Final paper, Table 2](https://proceedings.mlr.press/v48/pezeshki16.pdf), [original Ladder paper](https://arxiv.org/pdf/1507.02672).

Our fresh PyTorch implementation achieved **{original['accuracy_percent']:.2f}%** on the canonical **60,000-train / 10,000-test** split: **{original['errors']} errors**, versus the paper's mean 56.9 errors. Its accuracy differs from the published mean by **{original['accuracy_gap_from_published_pp']:+.3f} percentage points**. This is one prespecified seed, with no test-guided tuning; it is not the paper's ten-run result or its uncertainty estimate.

The same frozen recipe was trained from scratch on **five of the fifteen** Aminist 21 transfer tasks, using all **eleven fixed draws** per selected task. Every draw contains 10,000 training and 10,000 query examples at 9×9 resolution. These are procedure-transfer results, not evaluation of a pretrained MNIST digit classifier on unrelated classes.

| Transfer task | Ladder AMLP, mean ± SD | Existing MLP256 | Existing CNN ensemble | Ladder − MLP256, pp |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

The strong original pMNIST score did **not** make this frozen recipe the strongest transfer procedure. It improved on MLP256 for KMNIST, EMNIST A–J and CIFAR-10, but was worse for recovered QMNIST and Fashion-MNIST. The CNN ensemble remained ahead on all five. This is evidence that success on original MNIST alone does not predict the ordering of these procedures across the selected transfer tasks; it does not establish that the Ladder architecture could never do better after separately authorized tuning.

All values are accuracy percentages; SD is the sample SD across the eleven draws. References use the exact same draws and query labels. The CNN ensemble sees the original ordered 9×9 grid; our Ladder receives a fixed permutation and uses no spatial prior. These comparisons are between complete procedures, with different parameter counts and compute, not controlled architecture-only or equal-budget comparisons. All six reference recipes, paired differences, and per-draw values are retained in the comparison files.

## Benchmark interpretation

One PCG64 permutation with seed `20260923` is applied to all examples, identically at train and test: 784 features for canonical MNIST and 81 for the transfer suite. The learner gets the full feature vector and no inverse permutation or original coordinate map. A dense model can absorb a fixed feature permutation into its input weights, so the relevant literature calls this **permutation-invariant MNIST**. This is distinct from causal/sequential pMNIST and continual-learning suites of multiple permutations. Gaussian noise is coordinatewise; no rotations, translations, elastic distortions, spatial convolution, or external examples are used.

The five transfer datasets were selected for breadth before training or inspecting query results: Japanese characters, Latin letters, recovered digits, clothing, and objects. They retain the suite's curated pools, classes, preprocessing, draw seeds `20261101`–`20261111`, and 10k/10k disjoint-within-draw sampling. The suite's pools can repartition original releases, so these are not official-test results for KMNIST/Fashion-MNIST/CIFAR-10. Draws may overlap; some task families share ancestry. Reported SDs are descriptive, not confidence intervals. **This is an incomplete fifteen-task benchmark, with complete eleven-draw coverage of each selected task.**

## Method and reproducibility

The encoder is `input → 1000 → 500 → 250 → 250 → 250 → 10`, with batch normalization and ReLU hidden units. Each decoder coordinate has its own `3 → 2 → 2 → 1` leaky-ReLU combinator. The published full-label settings are noise SD 0.3 at every level, input reconstruction coefficient 2000 and zero at other levels, and combinator weight initialization SD 0.025. Adam uses learning rate 0.002, batch size 100, 100 constant-rate epochs plus 50 decay epochs. Final checkpoints are the last fixed epoch. [Supplementary Tables 4–5](https://proceedings.mlr.press/v48/pezeshki16-supp.pdf).

No authors' AMLP implementation was found. This is a paper-based PyTorch reimplementation informed by the original public Ladder code. Numerical/random-number implementation, Adam's epsilon placement, zero combinator biases, and independent reconstruction-stream shuffling are documented adaptation choices. The schedule follows the paper. Train-only clean batch-statistic calibration precedes evaluation. CUDA graph warmup is erased from model, optimizer, and random state before recorded training; arithmetic is FP32 with TF32 disabled.

Each of the 56 fits starts with fresh weights and optimizer state. Query labels are absent from GPU payloads. All 55 transfer prediction archives and the canonical MNIST predictions were saved before scoring. The evaluator checked source snapshots, all completed epoch counts, checkpoint/logit/prediction hashes, and agreement between saved logits and classes before reading official MNIST test labels. The suite scorer additionally rechecked dataset hashes, draw indices, disjoint native-image identities and matching input hashes. Correctness tests cover coordinate-permutation equivalence of the objective/gradients, clean inference, nodewise combinators and normalization calibration.

The fit workers reported **{seconds/60:.1f} GPU-worker minutes** including adapter setup/calibration/inference, of which **{train_seconds/60:.1f} minutes** were training. The entire app lasted **{elapsed/60:.1f} minutes**, with at most four workers requesting A100-40GB. Actual reported hardware: **{', '.join(gpu_names)}**. Using the higher published A100-80GB rate conservatively, adapter time corresponds to approximately **${seconds*rate:.2f}**, and charging four complete workers for the app's full elapsed time gives **${elapsed*4*rate:.2f}**; neither is a provider invoice. This corrects the launch record's estimate based on the requested 40GB rate. A separate synthetic GPU smoke test is recorded. The app had a 3,000-second absolute deadline and no user-code retries. Runtime is not energy consumption.

### Running and rescoring

The controller requires Python 3.11, NumPy, Modal 1.5.5 and the adjacent `aminist-21-validation` checkout (or `AMINIST21_ROOT`). That checkout supplies checksum-verified `data-v1` pools, sampling, and evaluation. The remote image is pinned by digest in `run_study.py`, with PyTorch 2.5.1/CUDA 12.4 and NumPy 2.2.6. Each record contains actual software/hardware versions. Local verification used PyTorch 2.2.2.

```bash
python -m pytest -q test_model.py
modal run run_study.py --mode smoke
modal run run_study.py --mode all
python audit_results.py
python score_study.py
python compare_results.py --scores results/all/transfer/scores.json --output results/all/comparison --label 'Ladder AMLP'
python build_report.py
```

Launch records prevent accidental reruns into an existing results directory. Archive the current results first before an intentional new run; do not replace completed predictions or mix source revisions. The `results/all/source` snapshot is the exact executed source. The canonical MNIST compressed files are checksum-verified from the neighboring original-MNIST experiment's cache. Reproduction elsewhere must supply those four standard files or adapt the cache location without changing their contents.

## Files

- [Literature audit and excluded claims](research/literature.md)
- [Detailed model protocol and porting choices](research/model-protocol.md)
- [Transfer dataset inventory](research/transfer-inventory.md)
- [Canonical pMNIST score](results/all/official/scores.json)
- [Transfer scores and per-draw confusion matrices](results/all/transfer/scores.json)
- [Transfer score CSV](results/all/transfer/scores.csv)
- [All-six-reference comparison](results/all/comparison/comparison.md)
- [Comparison data and provenance](results/all/comparison/comparison.json)
- [Independent audit](results/all/audit.json)
- [Frozen source snapshot](results/all/source/)
- [All-predictions verification record](results/all/all-predictions-verified.json)
- [Execution record](results/all/execution.json)
- [Conservative cost estimate](results/all/cost-estimate.json)
- [Stopped-app verification](results/lifecycle.json)
"""
    (ROOT/"README.md").write_text(text)
    print(ROOT/"README.md")


if __name__ == "__main__":
    main()
