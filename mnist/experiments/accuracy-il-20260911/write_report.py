"""Build the human-readable feasibility report from frozen result artifacts."""
from decimal import Decimal
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(value):
    mantissa, exponent = f'{value:.1e}'.split('e')
    return f'{mantissa} × 10{str(int(exponent)).translate(SUP)}'


def display(value):
    return format(Decimal(f'{value:.2g}'), 'f')


def label(row):
    return f"H{row['width']} · {row['epochs']:,} epochs"


def accuracy_label(row):
    counts = ' / '.join(str(value) for value in row['correct_by_seed'])
    passed = sum(value >= 360 for value in row['correct_by_seed'])
    return f"{row['mean_accuracy_percent']:.2g}% mean; {counts}; {passed}/3 pass"


def main():
    accuracy = json.loads((HERE/'accuracy_results.json').read_text())
    scoring = json.loads((HERE/'scoring_results.json').read_text())
    scores = {r['config_id']: r for r in scoring['configurations']}
    rows = sorted(accuracy['configurations'], key=lambda r: (r['width'], r['epochs']))
    baseline = scoring['baseline']
    lines = ['# Higher accuracy, practical scoring', '',
    '**MNIST-small · exploratory results · 11 September 2026**', '',
    '**The official MNIST-small accuracy target is now 60%, and this study meets that accuracy requirement.** A 32-hidden-unit network trained for 300 epochs cleared it with all three predeclared seeds. These results remain an exploratory study, not a complete new A100 submission. A 65% target was reached by only one run; 70% and 75% were not reached. This finite search does not establish an upper limit on accuracy.', '',
    '**Scoring the training algorithm is practical with a compact intermediate language.** The tested neural networks represent up to 24 billion primitive instructions, yet exact cost aggregation takes about '+display(max(r['median_static_score_seconds'] for r in scores.values()))+' s. Numerical training and accuracy verification are separate. These MLPs have no measured A100 runtime or energy yet.', '',
    '[TOC]', '',
    '## One display convention', '',
    'Execution times use **milliseconds (ms)**, energies use **millijoules (mJ)**, and **time to score uses seconds (s)**, with **two significant figures**. Prediction counts, model dimensions, and exact target thresholds retain their integer values. Raw JSON preserves full measurements and exact integer cost totals.', '',
    'Energy and time use matching milli prefixes: **E(mJ) = P(W) × t(ms)**. At 1 W, their numerical values are equal. Model and measured execution share ms and mJ; host scoring work is shown separately in s. The raw scorer retains exact internal ps ticks and fJ counts, converted only for display.', '',
    '## Higher accuracy results', '',
    'Each count below is correct predictions out of the same 600 test examples, in seed order 101 / 102 / 103. The shortlist and all 15 prediction arrays were frozen before test evaluation. Means and sample standard deviations describe seed variation on this fixed test set, not population uncertainty.', '',
    '| Configuration | Correct / 600, by seed | Mean accuracy (rounded) | Seed SD (percentage points) |',
    '| --- | --- | ---: | ---: |']
    for row in rows:
        counts = ' / '.join(str(v) for v in row['correct_by_seed'])
        lines.append(f"| {label(row)} | {counts} | {row['mean_accuracy_percent']:.2g}% | {row['sample_sd_percentage_points']:.2g} |")
    lines += ['', 'The original 1NN baseline scored **308/600 (51%)**, below the current **60%** requirement of **360/600** correct. It remains a historical measurement reference. All neural-network configurations use learning rate 0.2 and minibatches of 30. H denotes hidden-layer width.', '',
    '| Target | Required correct / 600 | Runs meeting it | Interpretation |',
    '| ---: | ---: | ---: | --- |']
    for target, interpretation in [(55,'Reached by every tested run.'),(60,'Reached by every seed at 300 epochs and above.'),
                                   (65,'One exploratory run; not reliable across seeds.'),(70,'Not reached in this search.'),(75,'Not reached in this search.')]:
        count = sum(r['correct']*100 >= target*r['total'] for r in accuracy['runs'])
        lines.append(f'| {target}% | {target*6} | {count}/15 | {interpretation} |')
    lines += ['', '**Target checks use exact counts, not rounded display percentages.** The single run above 65% was H32 / 1,000 epochs / seed 102, with 405/600 correct (about 68%). Its other two seeds scored 383/600 and 376/600. Choosing that seed after viewing the test result would need separate validation. The best training-validation configuration was H32 / 10,000 epochs; its three test results were 381/600, 389/600, and 387/600.', '',
    '## Complete-task model costs', '',
    'Every MLP score includes explicit scratch initialization, dataset tape operations, pixel transformation, one-hot target construction, initial weight writes, all training updates, inference, and output selection. Costs use the same pinned Dally v4 conventions as the 1NN baseline. Area is occupied scratch-cell area with the declared fixed placement.', '',
    'Accuracy columns show the rounded mean, exact correct counts out of 600 in seed order 101 / 102 / 103, and the number of seeds meeting the current 60% requirement. Pass/fail uses the exact 360/600 threshold.', '',
    '| Configuration | Accuracy (mean; correct / 600 by seed; ≥60%) | Model time (ms) | Model energy (mJ) | Area (µm²) |',
    '| --- | --- | ---: | ---: | ---: |',
    f"| Original 1NN | 51%; 308/600; below 60% | {display(baseline['time_ps']/1e9)} | {display(baseline['energy_fj']/1e12)} | {sci(baseline['area_um2_occupied_cells'])} |"]
    for row in rows:
        r=scores[row['config_id']]
        lines.append(f"| {label(row)} | {accuracy_label(row)} | {display(r['time_ps']/1e9)} | {display(r['energy_fj']/1e12)} | {sci(r['area_um2_occupied_cells'])} |")
    lines += ['', 'All three seeds of each configuration have identical model costs: only the seed-dependent literal bits differ. The learner uses separately rounded FP32 multiplication and addition with ascending reduction order. The cost model charges memory reads and writes; it is not a hardware power simulator.', '',
    '## Cost-evaluation work', '',
    'These timings include schema, address-bound and initialization checks, placement, exact access histograms, integer cost sums, and canonical program hashing. They exclude JSON loading, file output, numerical training, and accuracy verification. Each timing is the median of five complete scoring calls on the same host and Python environment.', '',
    '| Configuration | Accuracy (mean; correct / 600 by seed; ≥60%) | Expanded instructions | Compact JSON bytes | Time to score (s) |',
    '| --- | --- | ---: | ---: | ---: |',
    f"| Original 1NN | 51%; 308/600; below 60% | {sci(baseline['total_instructions'])} | {sci(baseline['program_json_bytes'])} | {display(baseline['median_static_score_seconds'])} |"]
    for row in rows:
        r=scores[row['config_id']]
        lines.append(f"| {label(row)} | {accuracy_label(row)} | {sci(r['total_instructions'])} | {sci(r['program_json_bytes'])} | {display(r['median_static_score_seconds'])} |")
    lines += ['', 'At fixed width H32, increasing training from 100 to 10,000 epochs multiplies the training cost by 100 while the static scoring time stays nearly constant. The epoch loop changes repetition count, not accessed addresses. The compact representation does not forgive the repeated work: each occurrence contributes its full v4 cost.', '',
    'The original 1NN interpreter took about **35 s** while also executing each FP32 instruction. That is a different workload from static cost evaluation. The new number is not an end-to-end verification speedup. A separate scaling stress test, without an accuracy claim, also scored a 41-billion-instruction program; raw measurements use a separately recorded Python/NumPy environment.', '',
    '## Measured execution and comparison boundaries', '',
    'For context, the CPU reference actually performed training and inference. The table gives the range across the three final seeds. Timing begins after input transformation, one-hot conversion, and parameter initialization; it excludes the independent ordered-reduction checks. Those operations are included in the theoretical IL costs above. CPU energy was not measured.', '',
    '| Configuration | Accuracy (mean; correct / 600 by seed; ≥60%) | CPU reference training + inference time (ms) | A100 time / energy |',
    '| --- | --- | ---: | --- |']
    for row in rows:
        elapsed=[r['cpu_reference_train_and_infer_seconds']*1e3 for r in accuracy['runs'] if r['config_id']==row['config_id']]
        low, high = display(min(elapsed)), display(max(elapsed))
        interval = low if low == high else f'{low}–{high}'
        lines.append(f"| {label(row)} | {accuracy_label(row)} | {interval} | Not measured |")
    lines += ['', 'The already measured **1NN** comparison remains a historical reference: its **308/600 (51%)** accuracy is below the current **60%** target.', '',
    '| Quantity | Dally model | A100 measured |', '| --- | ---: | ---: |',
    '| Time (ms) | 1.7 | 0.0069 |', '| Energy (mJ) | 0.0019 | 0.52 |', '',
    'A100 values are GPU-resident steady-state complete-task throughput and idle-adjusted NVML energy, including training memorization. Host transfer, compilation, warm-up, and idle baseline selection have different boundaries. See the original submission for raw trials and baseline sensitivity. No A100 values have been extrapolated to the MLPs.', '',
    '## The proposed intermediate language', '',
    '`sutro-affine-v4/0.1` stores a fixed scratch layout, nested constant-bound loops, affine addresses, and ordinary v4 instructions. A dot product is a loop of `mul` and `add`; training is loops around explicitly represented forward, backward, and update operations. There is no free matrix-multiply operation or caller-supplied cost certificate.', '',
    'For each primitive operand, the scorer computes the exact histogram of concrete addresses across its enclosing loops. It combines address progressions with discrete convolution. An epoch index absent from an address simply multiplies the count. Reads and writes remain separate, aliased operands are charged repeatedly, and `select` charges both candidate values. Per-address counts are then multiplied by the pinned distance-dependent costs.', '',
    'The prototype deliberately restricts programs to fixed control flow and affine addressing. It checks all address bounds and proves sources initialized before use, including unchosen selections. Difficult initialization proofs are rejected when the proof budget is exhausted. Correctness, training/test separation, and accuracy still require an independent semantic evaluator.', '',
    '## Verification evidence', '',
    '- **Original 1NN:** the compact program expands byte-for-byte to the original 220 MB v4 trace. Exact model time, energy, opcode counts, and all 6,014 per-address read/write counts agree.',
    '- **General scorer:** 11 tests cover independent enumeration, negative strides, overlapping addresses, aliases, selection, initialization, bounds, tape semantics, overflow, and a small multi-batch MLP.',
    '- **MLP lowering:** two complete small training/inference programs were expanded and executed in the original interpreter with explicit comparison predicates. Every learned parameter bit, output prediction, instruction count, and model score matched the ordered reference.',
    '- **Full training arithmetic:** the validation-best H32 / 10,000-epoch / seed-101 run was independently repeated with explicit ordered FP32 reductions. All final parameter bits and all 600 output score vectors matched. The comparison took about **140,000 ms**.',
    '- **Every final run:** all 600 final score vectors matched explicit ordered reductions. Canonical dataset hashes, source hashes, prediction arrays, selection plans, and chronology are saved.', '',
    'The full 24-billion-instruction MLP trace was not expanded and interpreted. The evidence combines independent scorer checks, small end-to-end lowering checks, source review, and a full ordered numerical training check. Official acceptance of the IL and a complete MLP A100 submission remain future work.', '',
    '## Reproduce and inspect', '',
    'Run from the repository root with Python 3.11. The accuracy study records NumPy 2.4.6; the separate legacy scaling file records its own environment. Use a fresh directory for accuracy reruns so the frozen published evidence is preserved.', '',
    '```bash', 'S=mnist/experiments/accuracy-il-20260911',
    'python -m pip install -r "$S/study-requirements.txt"',
    'python -m unittest discover -s "$S" -p test_il.py -v',
    'python "$S/validate_mlp_il.py" --output /tmp/mnist-mlp-validation.json',
    'python "$S/score_study.py" --output /tmp/mnist-cost-rerun',
    'python "$S/accuracy_study.py" --phase search --output /tmp/mnist-accuracy-rerun',
    'python "$S/accuracy_study.py" --phase extend --output /tmp/mnist-accuracy-rerun',
    'python "$S/accuracy_study.py" --phase final --output /tmp/mnist-accuracy-rerun',
    'python "$S/verify_ordered_training.py" --output /tmp/mnist-accuracy-rerun',
    'python "$S/accuracy_study.py" --phase evaluate --output /tmp/mnist-accuracy-rerun', '```', '',
    'The first scoring command uses the published frozen shortlist and emits regenerated IL programs plus repeated timing samples. Training and scoring need no external model weights. Reproduction of the validation protocol intentionally follows the recorded second-stage extension.', '',
    '**Documents:** [Language specification](il.html) · [Study protocol](protocol.html) · [Ambiguities and remaining work](ambiguities.html) · [Visible session export](session.html).', '',
    '**Exact evidence:** [Accuracy results](accuracy_results.json) · [Cost results](scoring_results.json) · [Frozen predictions](frozen_predictions.json) · [Full training check](ordered_training_verification.json) · [MLP lowering check](mlp_validation.json) · [IL checks](il-validation.json) · [Trace identity](il-expansion-validation.json) · [Scaling stress test](il-scaling.json).', '',
    '**Source:** [Learner](accuracy_study.py) · [IL scorer](il.py) · [MLP lowering](mlp_il.py) · [Scoring driver](score_study.py) · [Repository directory](https://github.com/cybertronai/sutro-problems/tree/main/mnist/experiments/accuracy-il-20260911).', '',
    '**Related:** [Original submission and A100 measurements](../1nn-v4-20260911/) · [MNIST task](https://github.com/cybertronai/sutro-problems/blob/main/mnist/README.md#mnist-small).', '']
    (HERE/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
