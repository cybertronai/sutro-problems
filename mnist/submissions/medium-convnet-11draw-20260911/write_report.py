"""Generate the submission report and accuracy plot from frozen evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cost(value):
    value = f'{value:.2g}'
    if 'e' in value:
        mantissa, exponent = value.split('e')
        superscript = str(int(exponent)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
        return f'{mantissa} × 10{superscript}'
    return value


def main():
    accuracy = json.loads((HERE / 'accuracy.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    benchmark = json.loads((HERE / 'benchmark/measurements.json').read_text())
    summary = benchmark['summary']
    mean = accuracy['mean_accuracy_percent']
    sd = accuracy['sample_standard_deviation_pp']
    wall = summary['wall_ms_per_task_median']
    energy = summary['idle_adjusted_energy_mj_per_task_median']
    gpu = summary['cuda_event_ms_per_task_median']
    gross = summary['gross_energy_mj_per_task_median']
    meets = accuracy['meets_accuracy_target']
    result = 'meets' if meets else 'does not meet'
    trial_lines = [
        '| Trial | Wall time (ms) | Idle-adjusted energy (mJ) | Idle before (W) | Idle after (W) |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for trial in benchmark['trials']:
        trial_lines.append(f'| {trial["trial"] + 1} | {cost(trial["wall_ms_per_task"])} | '
                           f'{cost(trial["idle_adjusted_energy_mj_per_task"])} | '
                           f'{cost(trial["idle_before"]["average_power_w"])} | '
                           f'{cost(trial["idle_after"]["average_power_w"])} |')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'medium-convnet-11draw-20260911', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 4.0), constrained_layout=True)
    values = [row['accuracy_percent'] for row in accuracy['draws']]
    ax.axhspan(mean - sd, mean + sd, color='#0d7d84', alpha=.12, label='Mean ± sample SD')
    ax.axhline(mean, color='#0d7d84', linewidth=1.5, label=f'Mean {mean:.1f}%')
    ax.axhline(98, color='#a85b1e', linestyle='--', linewidth=1.5, label='98% mean requirement')
    ax.scatter(range(11), values, color='#103c47', s=40, zorder=3)
    ax.set_xticks(range(11), [f'{i:02d}' for i in range(11)])
    ax.set_xlabel('Independent dataset draw')
    ax.set_ylabel('Ensemble accuracy (%)')
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(f'Fresh training on every draw: {mean:.1f}% ± {sd:.1f} pp', loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=.15)
    ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=3,
              bbox_to_anchor=(0, -.19))
    fig.savefig(HERE / 'accuracy-by-draw.svg', metadata={'Date': None})
    plt.close(fig)
    svg = HERE / 'accuracy-by-draw.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')

    text = [
        '# MNIST-medium: three ConvNets across 11 datasets', '',
        f'**The frozen ensemble reached {mean:.1f}% ± {sd:.1f} percentage points across 11 independently sampled datasets.** '
        f'It {result} the **98% mean accuracy requirement** with {accuracy["total_correct"]:,} / 66,000 correct predictions. '
        f'The unrounded total is {abs(accuracy["margin_correct"]):,} predictions {"above" if meets else "below"} the inclusive 64,680 threshold.', '',
        f'A100 measurements include fresh training of all three members and prediction: **{cost(wall)} ms** and '
        f'**{cost(energy)} mJ** idle-adjusted GPU-board energy per complete task, using medians of three trials on draw 00.', '',
        '> **Submission status:** The accuracy requirement and A100 measurements are established. '
        'The exact Dally v4 translation and its theoretical time, energy, area, and scoring runtime remain incomplete. '
        'This is a submitted attempt for review, not a claim of complete benchmark compliance. '
        'The separate problems report and scoring-feasibility audit explain the missing work.', '',
        '[TOC]', '',
        '## Submission entry', '',
        '| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        f'| {mean:.1f}% ± {sd:.1f} pp | — | — | — | — | {cost(wall)} | {cost(energy)} |', '',
        'Accuracy is mean ± sample standard deviation of dataset-level accuracies. The target applies to the unrounded mean; '
        'it does not require every draw, or mean minus SD, to exceed 98%. Costs use two significant figures. '
        'Dashes mean unavailable; accuracy-evaluation duration is not substituted for time to score.', '',
        '## Accuracy on the 11 frozen draws', '',
        '![Per-dataset ensemble accuracy, with the mean, sample SD band, and 98% requirement](accuracy-by-draw.svg)', '',
        'Both training and test subsets are resampled independently for every draw from the original 60,000 MNIST training images. '
        'Each draw has 6,000 training rows and 6,000 disjoint test rows, resized to 9 × 9 with the canonical competition-v2 area-resize procedure. '
        'Different draws may overlap. The integer dataset seeds are fixed in the protocol; they are not timestamps.', '',
        '| Draw | Dataset seed | Accuracy | Correct / total | Predictions |',
        '| ---: | ---: | ---: | ---: | --- |',
    ]
    for row in accuracy['draws']:
        i = row['draw_index']
        text.append(f'| {i:02d} | {row["dataset_seed"]} | {row["accuracy_percent"]:.1f}% | '
                    f'{row["correct"]:,} / 6,000 | [Array](predictions/draw-{i:02d}.npy) · [Manifest](draws/draw-{i:02d}.json) |')
    text += ['',
        f'The exact aggregate is **{accuracy["total_correct"]:,} / 66,000**. '
        '`accuracy.json` retains the full-precision mean, sample SD, exact rational variance, and per-member diagnostic counts. '
        'SD uses the 11 percentages with denominator 10 (`ddof=1`); it is not SD across the three members and not a standard error. '
        'All 11 draws are retained; none was discarded or selected after evaluation.', '',
        '## Frozen learner', '',
        'Each member uses three padded 3 × 3 convolution layers of width 64, each followed by BatchNorm and GELU, with no pooling. '
        'A flattened 5,184-element feature vector feeds a 256-unit GELU head, dropout 0.2, and ten output logits. '
        'There are 1,404,618 trainable parameters per member. The three FP32 raw-logit arrays are averaged in FP64; '
        'argmax breaks ties toward the smallest digit.', '',
        'Training runs for exactly 71 epochs with AdamW (learning rate 0.001, weight decay 0.001, '
        'betas 0.9 and 0.999, epsilon 1e-8). The cosine schedule retains its original 100-epoch horizon and a 2% learning-rate floor. '
        'Minibatches contain 128 examples, retaining the final partial batch of 112. Each epoch receives a seeded shuffle. '
        'Normalization uses only that draw’s training-image mean and population standard deviation.', '',
        'Mild affine augmentation applies with probability 0.5: rotation within ±8°, translation within ±0.35 pixels per axis, '
        'and inverse sampling scale from 0.94 to 1.06. Bilinear sampling uses zero padding and `align_corners=False`, '
        'before normalization. Test images are normalized only and do not affect training statistics or optimizer updates.', '',
        'Member seeds 101, 102, and 103 are held fixed across draws. Every member starts with fresh weights, optimizer, scheduler, '
        'and normalization state. The final 71st epoch is used without validation, early stopping, checkpoint selection, or tuning on these draws. '
        'Per-epoch training diagnostics are recorded but do not select the returned model.', '',
        '## Development history and label isolation', '',
        'The architecture and 71-epoch stopping rule came from the preceding ConvNet search using a training-only validation split. '
        'That study’s final single-dataset test also measured a diagnostic three-model ensemble. Choosing that ensemble for this new attempt '
        'was informed by its disclosed historical result; it is not presented as a choice made before all historical test observations. '
        'The new 11 seeds and complete procedure were frozen before this evaluation. No historical weights or learned state were reused.', '',
        f'- Protocol frozen: `{protocol["frozen_at_utc"][:19].replace("T", " ")} UTC`.',
        f'- All 11 prediction artifacts frozen: `{accuracy["prediction_frozen_at_utc"][:19].replace("T", " ")} UTC`.',
        f'- Evaluator/evidence freeze: `{accuracy["evaluation_frozen_at_utc"][:19].replace("T", " ")} UTC`.',
        f'- Raw labels opened for scoring: `{accuracy["raw_labels_opened_at_utc"][:19].replace("T", " ")} UTC`.', '',
        'Trusted preparation reads the original label pool only to extract permitted training labels. It does not construct per-draw test labels. '
        'Each remote fit receives only its own draw’s training images, training labels, and test images. Raw IDX files, other draws’ archives, '
        'and pretrained checkpoints are not mounted. The separate evaluator verifies every prediction, logit, manifest, source hash, '
        'ensemble calculation, and retained checkpoint before deriving any test-label vector.', '',
        'An independent audit regenerated all 11 index sets and 22 resized image arrays, checked source equivalence to the earlier learner, '
        'and independently recomputed aggregate counts and sample SD. All predictions, logits, histories, and model-state hashes are retained; '
        'checkpoint binaries are retained for draw 00 only. Repeating the learner regenerates the other weights.', '',
        '## Complete-task A100 measurements', '',
        'The benchmark uses draw 00 (dataset seed 20261001), chosen before measuring performance. It calls the unchanged frozen learner '
        'sequentially for seeds 101, 102, and 103, then averages their logits in FP64 and returns predictions. '
        'One complete warmup precedes three measured fresh tasks in one A100 allocation. Each measured invocation includes model initialization, '
        'optimizer initialization, normalization, all training, the per-epoch training diagnostics, test inference, CPU↔GPU transfers, '
        'tensor hashing, checkpoint serialization, and ensemble construction. No trained parameters are reused between tasks.', '',
        '| Measured quantity | Median per complete task |',
        '| --- | ---: |',
        f'| Wall time on A100 (ms), primary runtime | {cost(wall)} |',
        f'| CUDA-event elapsed time (ms) | {cost(gpu)} |',
        f'| Idle-adjusted GPU-board energy (mJ), primary energy | {cost(energy)} |',
        f'| Gross GPU-board energy (mJ) | {cost(gross)} |', '',
        f'Across the three trials, wall time ranges from {cost(summary["wall_ms_per_task_min"])} to '
        f'{cost(summary["wall_ms_per_task_max"])} ms, and idle-adjusted energy from '
        f'{cost(summary["idle_adjusted_energy_mj_per_task_min"])} to '
        f'{cost(summary["idle_adjusted_energy_mj_per_task_max"])} mJ. '
        'The entry reports medians; the per-trial spread and both idle baselines remain visible below.', '',
        *trial_lines, '',
        'The wall timer excludes the bounding NVML API calls. A separately retained counter interval drives energy subtraction. '
        'Container startup, dependency imports, input delivery to the container, and result upload to this machine are excluded. '
        'CUDA-event elapsed time includes host-induced gaps between GPU work; it is not kernel-active time. '
        'The GPU and its memory are measured by NVML; host CPU and network energy are excluded.', '',
        'Before and after each trial, the benchmark waits three seconds for settling and samples idle energy for three seconds. '
        'Idle-adjusted energy is the gross counter delta minus the average of those two idle powers times the counter interval. '
        'The raw trials retain both baseline choices, gross energy, timing intervals, temperatures, clocks, and output-identity checks. '
        'Reference comparisons are verification only: the learner does not receive or consume reference learned parameters or answers.', '',
        'The pinned image uses PyTorch 2.5.1+cu124 with CUDA 12.4, cuDNN 9.1, and NumPy 2.2.6. '
        'Execution uses deterministic PyTorch/cuDNN, FP32, no autocast, and no TF32. The raw benchmark record identifies the exact GPU, '
        'driver, NVML library, and measurement environment. Results describe this implementation on this allocation, not an optimized lower bound.', '',
        f'Measured GPU: **{benchmark["hardware"]["gpu"]}**. Driver: `{benchmark["hardware"]["driver"]}`. '
        f'NVML: `{benchmark["hardware"]["nvml"]}`. All warmup and measured model-state hashes, FP32 logits, '
        'FP64 ensemble values, and final predictions matched draw 00 exactly.', '',
        'The container requests four CPU cores. The host CPU model was not recorded; because host work is included in '
        'the elapsed task, reproductions on another host may differ even with the same GPU. The GPU power limit and clocks '
        'were left at platform defaults.', '',
        '## Theoretical scoring is still open', '',
        'There is no complete v4 program for this learner. GELU, BatchNorm, AdamW, cross-entropy, augmentation, and FP64 ensemble arithmetic '
        'need explicit lowering to the restricted primitive set; GPU reductions and fused arithmetic need a defined numerical correspondence. '
        'A compact tensor-loop IL can describe repeated work, but every operator must expand to valid instructions with charged memory accesses. '
        'The feasibility audit documents the proposed representation and validation path. It does not claim an implemented translator.', '',
        'Theoretical model time, energy, area, and time to score are therefore left unavailable. '
        'Parameter count and GPU allocation do not establish occupied scratch-cell area. '
        'Any changed arithmetic needs its own accuracy evaluation before it can claim this result. '
        'The report intentionally does not equate a native PyTorch FLOP estimate with an exact model score.', '',
        '## Reproduction and evidence', '',
        'The accompanying README gives commands to prepare isolated inputs, run the 33 fits, freeze and evaluate predictions, '
        'repeat the A100 measurements, and rebuild these pages. Contributors: Yaroslav Bulatov (requirements), '
        'Codex (implementation, experiments, auditing, and reporting). No W&B runs were created.', '',
        '- [Reproduction instructions](README.md)',
        '- [Exact accuracy results](accuracy.json) · [Independent audit](audit.json)',
        '- [Frozen protocol](protocol.json) · [Draw manifest](draw_manifest.json)',
        '- [Prediction freeze](prediction_manifest.json) · [Evaluator freeze](evaluation_freeze.json)',
        '- [Learner source](learner.py) · [Source-equivalence audit](learner_equivalence.json)',
        '- [Raw A100 measurements](benchmark/measurements.json) · [Independent A100 audit](benchmark/audit.json) · [Benchmark source](gpu_benchmark.py)',
        '- [Detailed scoring-feasibility audit](scoring-feasibility.md)',
        '- [Separate ambiguities and problems](ambiguities.html)',
        '- [Human-readable session export](session.html)',
        '- [Previous ConvNet development study](../medium-convnet-20260911/)', '',
    ]
    (HERE / 'report.md').write_text('\n'.join(text))
    print(HERE / 'report.md')


if __name__ == '__main__':
    main()
