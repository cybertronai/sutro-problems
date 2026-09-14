"""Build English documentation and freeze a small, portable evidence package.

Does not launch Modal, open a PR, commit, push, or publish GitHub Pages.
"""
import hashlib
import html
import json
import math
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TITLE = 'MNIST-small: energy-aware panel MLP'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sig(value):
    if value==0:
        return '0'
    digits = 1-math.floor(math.log10(abs(value)))
    return f'{value:.{digits}f}' if digits>=0 else f'{round(value,digits):,.0f}'


def export_accuracy():
    source = HERE/'generated/fresh11'
    destination = HERE/'evidence/accuracy'
    if not (source/'accuracy.json').exists():
        source = destination
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('plan.json','predictions_frozen.json','accuracy.json'):
        (destination/name).write_bytes((source/name).read_bytes())
    accuracy = json.loads((source/'accuracy.json').read_text())
    for row in accuracy['draws']:
        for field in ('manifest_file','prediction_file'):
            path = destination/row[field]
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((source/row[field]).read_bytes())
    if (HERE/'generated/input_provenance.json').exists():
        (HERE/'evidence/input_provenance.json').write_bytes((HERE/'generated/input_provenance.json').read_bytes())
    if (HERE/'generated/verification.json').exists():
        (HERE/'evidence/submission_verification.json').write_bytes((HERE/'generated/verification.json').read_bytes())
    return accuracy


def main():
    accuracy = export_accuracy()
    costs = json.loads((HERE/'costs.json').read_text())
    gpu = json.loads((HERE/'evidence/gpu/results.json').read_text())
    comparison = json.loads((HERE/'evidence/gpu_comparison.json').read_text())
    selected = costs['optimized']
    primary = gpu['summary']['energy']
    acc = f'{accuracy["mean_percent"]:.1f}% +/- {accuracy["sample_sd_pp"]:.1f} pp'
    metric_row = f'| {acc} | {sig(selected["time_ms"])} | {sig(selected["energy_mj"])} | {sig(selected["area_mm2"])} | {sig(selected["time_to_score_wall_median"])} | {sig(primary["cuda_ms"]["median"])} | {sig(primary["adjusted_mj"]["median"])} |'
    metrics = {'tier':'small','specification':'historical: 600/600 and single-core-with-tape',
               'current_specification_qualified':False,'current_spatial_grid_metrics':None,
               'accuracy':accuracy,'theoretical':selected,'gpu':primary,
               'headline_variant':'energy','model_variant':'optimized','gpu_primary_run':'ap-5pImGSZ90Y9PNI1vVQf9TO'}
    (HERE/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    table = ['| Accuracy | Time (ms) | Energy (mJ) | Area (mm2) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) |',
             '|---|---:|---:|---:|---:|---:|---:|',metric_row]
    lines = [f'# {TITLE}', '',
        '**Historical specification: 600/600 examples and single-core-with-tape scoring.**', '',
        'Upstream main now requires 1,000/1,000 examples, 67% mean accuracy and spatial-computer '
        'grid scoring. This study was measured under the previous specification and is NOT a '
        'current qualifying submission. Its single-core numbers must not be placed in spatial-grid '
        'columns. The current-specification metrics remain unmeasured.', '',
        '## Summary', '',
        'A fixed 9-32-10 ReLU MLP is optimized through operand reuse, persistent rectangular panels '
        'and memory placement across all seven matrix products. Architecture, initialization, '
        'ordered FP32 arithmetic, learning rate, batch size and training budget are unchanged. '
        'The primary variant saves 14.14% modeled energy and 17.95% measured idle-adjusted A100 '
        'energy against the original MLP. It takes 1.31% more modeled time but 20.34% less A100 time.', '',
        *table, '',
        f'Accuracy: **{accuracy["correct"]}/6600**, above the historical requirement of3960/6600, across eleven fresh '
        'predeclared datasets. SD is the sample standard deviation across datasets, not across '
        'training seeds. Costs use two significant figures here; exact values remain in JSON.', '',
        '## Contributors and Prior Work', '',
        'OpenCode: panel implementation, local search, CPU/IL verification, A100 experiments, '
        'submission packaging and documentation. SecurityQQ: research direction and submission. '
        'No legal name is inferred from the GitHub handle.', '',
        'The original learner and measurement protocol are from the repository MLP60 submission '
        '(Yaroslav Bulatov and Codex). Scheduling ideas come from the matmul4x4 work of Juraj '
        'Selep and the matmul16x16 panel/capture work credited to Cosmin, sjbaebae and '
        'SecurityQQ/OpenCode. Their literal v0 scores and schedules are not presented as this '
        'submission\'s v4 measurements. No W&B run was created.', '',
        '## Frozen Learner', '',
        '- Tier: MNIST-small only,600 train and600 test examples,3x3 images.',
        '- Network:9 inputs,32 ReLU hidden units,10 output scores;650 FP32 parameters.',
        '- Initialization: NumPy PCG64 seed101, uniform bounds1/sqrt(fan-in), zero biases.',
        '- Normalization: float32(x)*4-0.5. Loss: squared error against one-hot targets.',
        '- SGD:300 epochs, batch30, learning rate0.2, fixed cyclic supplied-row order, no shuffle.',
        '- Fresh parameters for every draw/invocation; no pretrained weights, ensembles or cross-draw state.',
        '- Every output reduction starts at+0 and accumulates in ascending K, with separate FP32 multiply/add.',
        '- Predictions are argmax scores with first-class tie breaking.',
        '- The learner reads only train_images, train_labels and test_images; never test_labels.', '',
        '## Seven Matrix Products', '',
        '| Product | Shape | Selected execution |', '|---|---|---|',
        '| X @ W1 |30x9 times9x32|Cache minibatch X; retain9x4 W1 panels across30 rows|',
        '| H @ W2 |30x32 times32x10|Stage one H operand and reuse across ten classes|',
        '| D2 @ old W2.T |30x10 times10x32|Retain10x4 transposed-W2 panels across30 rows|',
        '| X.T @ D1 |9x30 times30x32|Reuse X cache; retain30x4 D1 panels across nine features|',
        '| H.T @ D2 |32x30 times30x10|Stage H values across ten classes; stream weight updates|',
        '| Q @ W1 |30-query groups|Reuse query cache and9x4 W1 panels|',
        '| H_query @ W2 |30-query groups|Reuse hidden operands across ten classes|', '',
        'All valid results are consumed once. The v4 instruction counts for multiply, add, subtract, '
        'compare and select equal the original baseline. The extra work is captures/copies and '
        'scratch initialization, traded against fewer expensive operand reads. Transposes are '
        'strided views, and no full gradient matrices are materialized.', '',
        'Shared extra scratch is401 words:10 accumulators, one staged operand, a120-word maximum '
        'retained panel and a270-word minibatch/query cache. Total scratch is20,691 words '
        '(82,764bytes), area0.020691mm2 before rounding. All D1 values consume old W2 before '
        'weight updates. D2 storage is reused for inference scores only after training finishes.', '',
        '## Model Scoring', '',
        'The selected executable uses the repository\'s compact affine-v4 representation and '
        'the same pinned single-core-with-tape scorer as the original MLP submission. It includes '
        'all initialization, input normalization, target construction, training, output work, '
        'capture writes and placement shifts. Every expanded leaf is a v4 primitive. '
        'recv/send conventions and occupied-cell area follow the inherited scorer.', '',
        'Energy per scratch access is max(50,2*Manhattan_distance)fJ. Read time is '
        'max(250,4*distance)ticks; write time is max(250,2*distance)ticks, each tick0.2ps. '
        'Reads and writes count. Arithmetic operations themselves are not separately charged. '
        'Area covers peak initialized scratch only, excluding processor, routing, instruction '
        'storage and tape. Canonical600/600 input sizes are used for these costs.', '',
        'Time to score is the recorded host median of five score(document) calls after a warmup. '
        'It includes validation, placement, address histograms, cost summation and hashing, '
        'but excludes generation, JSON I/O, search and numerical replay. Fresh reproduction '
        'updates timing samples without changing the exact model totals.', '',
        'Search covered1284 initial distinct configurations and826 joint-inference refinement '
        'evaluations, all local static costs. It tested asymmetric dimensions, staging sides, '
        'persistent panels, layouts and inference grouping. The primary objective was full-task '
        'energy, then time, then space. This was not a global optimality proof or accuracy search.', '',
        '## Fresh Eleven-Draw Evaluation', '',
        'The submission evaluation was performed AFTER freezing the final algorithm and BEFORE '
        'examining these eleven test results. It uses new seeds2026091301 through2026091311, '
        'not the earlier exploratory draw set. The plan stores source/config hashes and a timestamp. '
        'All eleven predictions were written and hashed before a separate scoring phase opened '
        'test-label slices. No tuning or retraining followed that scoring.', '',
        'Sampling follows the repository generator: independently permute the original60,000 '
        'training examples for each seed; take600 training examples at offset0 and600 test '
        'examples at offset6000. Sampling is without replacement within a draw, train/test '
        'indices are disjoint, and different draws may overlap. Training randomness is fixed '
        'to seed101. Pixel preprocessing uses float32 division by255 and repository box-area '
        'resize to3x3, then clipping to[0,1]. Manifests record actual input bytes and source indices.', '',
        '| Dataset seed | Learner seed | Correct / total | Accuracy |', '|---|---:|---:|---:|']
    for row in accuracy['draws']:
        lines.append(f'| {row["seed"]} |101| {row["correct"]}/600 | {row["accuracy_percent"]:.1f}% |')
    lines += ['', 'The primary CPU panel learner and the original CPU learner matched parameters, '
        'scores and predictions bit-for-bit on every fresh draw. The reported variation is '
        'dataset variation, not training-seed variation. All eleven draws are included.', '',
        '## A100 Protocol and Results', '',
        'Primary measurements used one NVIDIA A100-SXM4-40GB. The original eight Triton kernel '
        'definitions are unchanged. All variants ran freshly in one container with three cyclic '
        'rounds: baseline/energy/no_slowdown, energy/no_slowdown/baseline, '
        'no_slowdown/baseline/energy. Each occupied each trial position once.', '',
        'Every graph replay resets the model, normalizes inputs, constructs targets, trains '
        'all300epochs and writes600predictions plus6000scores. CUDA runtime graph enumeration '
        'confirmed24,004 kernel nodes per variant: four per minibatch plus four setup/inference '
        'launches. No extra per-minibatch prefetch launch is hidden.', '',
        'Time is measured by CUDA events and synchronized wall time. Each active interval targets '
        '10seconds. Before and after it, allow3seconds settling and measure idle consumption '
        'for3seconds. Adjusted energy equals the NVML cumulative counter difference minus '
        'mean paired idle power times the actual NVML interval, divided by graph replays. '
        'Raw counters, temperatures, clocks, throttling and one-sided idle corrections are saved.', '',
        '| Variant | CUDA time (ms) | Adjusted energy (mJ) | Gross energy (mJ) | CUDA change | Adjusted-energy change |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ('baseline','energy','no_slowdown'):
        value = gpu['summary'][mode]
        delta = comparison['median_changes_percent'][mode]
        lines.append(f'| {mode} | {sig(value["cuda_ms"]["median"])} | {sig(value["adjusted_mj"]["median"])} | '
                     f'{sig(value["gross_mj"]["median"])} | {delta["cuda_ms"]:+.2f}% | {delta["adjusted_mj"]:+.2f}% |')
    lines += ['', 'The energy profile wins time and both energy measures in all three matched rounds. '
        'The name no_slowdown refers only to its model-selection constraint; it is slower on '
        'A100 and is retained as a negative comparison, not a second headline submission.', '',
        'GPU kernels use SIMD/register broadcasts for panel reuse. The X cache is real270-word '
        'cross-kernel storage, filled inside a single-CTA hidden kernel with a barrier. '
        'The energy update kernel computes disjoint W1/W2/bias outputs with no repeated valid '
        'results. Candidate CTA grids differ; original baseline grids and arithmetic flags '
        'are retained. GPU parallel inference materializes all600hidden rows between two '
        'launches, unlike the serial model\'s30-row reuse. Dally distances do not map to GPU '
        'virtual addresses, and model area is not GPU memory.', '',
        'GPU validation covers all650parameter words,6000scores and600predictions after eager '
        'execution, captured replay, each timed round, changed queries and changed training '
        'labels before/after timing, and canonical restoration. All match the independent '
        'ordered-FP32 CPU reference. PTX has no floating FMA/MAD; SASS was not separately '
        'disassembled in this experiment.', '',
        'Excluded from steady-state metrics: host/device transfers, allocation, JIT, graph '
        'capture, validation, startup and CPU energy. These can still incur cloud charges. '
        'Board-energy telemetry and idle subtraction have uncertainty; three trials on one '
        'device are not a population confidence interval.', '',
        '## Hardware and Software', '', '```json',json.dumps({'hardware':gpu['hardware'],'versions':gpu['versions']},indent=2),'```','',
        'CPU scoring/evaluation used Python3.14.7, NumPy2.4.1 and macOS15.5 ARM64. '
        'The canonical GPU input arrays are recovered from the already tracked input tape and '
        'checked against the canonical manifest. This avoids silently accepting platform-dependent '
        'resize-bit differences. The tape contains no test labels. Fresh draws retain their own '
        'manifests; exact resize-byte reproduction across other BLAS/CPU environments is not guaranteed.', '',
        '## Verification and Limitations', '',
        'Actual expanded-IL tests cover full minibatches, multiple batches/epochs, inference '
        'group tails and changed inputs/labels. Every parameter, score, prediction and per-address '
        'read/write count matches reference execution. A full300-epoch CPU run matches the '
        'original after each epoch. The approximately641million-instruction whole IL is scored '
        'statically, not fully interpreted. This evidence is not a claim of full instruction-by-instruction '
        'execution or official acceptance of the prototype affine representation.', '',
        'Two initial GPU attempts failed before timing due to Triton compilation restrictions. '
        'A preliminary completed run used a shared buffer that misaligned D2 relative to the '
        'original allocation. It is preserved but not primary. The final run restored separate '
        'aligned buffers and repeated all three variants. The two completed runs used different '
        'GPU allocations, so their absolute differences do not isolate alignment effects. '
        'The headline uses only the final aligned run, not a pooled or best-of selection.', '',
        '## Reproduction', '',
        'The package imports only its own files and existing tracked repository modules. '
        'No untracked research directory, generated archive, saved model or external workspace '
        'is required by the learner or model generator. Commands are listed in README.md. '
        'Data/model caches are generated locally and excluded from the proposed PR; small '
        'prediction artifacts and exact measurement evidence are included.', '',
        '## Artifacts', '',
        '- [Reproduction instructions](README.md)',
        '- [Requirement checklist and caveats](REVIEW_CHECKLIST.md)',
        '- [Submission verification results](evidence/submission_verification.json)',
        '- [Full-precision headline metrics](metrics.json)',
        '- [Fresh11 accuracy and per-draw counts](evidence/accuracy/accuracy.json)',
        '- [Predeclared evaluation plan](evidence/accuracy/plan.json)',
        '- [Frozen prediction manifest](evidence/accuracy/predictions_frozen.json)',
        '- [Exact theoretical scores](costs.json)',
        '- [Executable optimized IL](optimized.il.json)',
        '- [Primary A100 raw counters and validation](evidence/gpu/results.json)',
        '- [Measured GPU source snapshot](evidence/gpu/runner.py)',
        '- [GPU run history](evidence/gpu_run_history.json)',
        '- [Canonical input provenance](evidence/input_provenance.json)',
        '- [Primary Modal run](https://modal.com/apps/vargapowercouple/main/ap-5pImGSZ90Y9PNI1vVQf9TO)']
    formatted = []
    in_code = False
    for line in lines:
        if line.startswith('```'):
            in_code = not in_code
        elif not in_code and not line.startswith('- ['):
            line = re.sub(r'\b(all|each|every|with|of|after|before|at|through|plus|and|over|from|seed|seeds|batch|epochs|trains|writes|produces|times|takes|targets|by|required|original|fixed|maximum|across|retain|retains|offset|used)(?=\d)',r'\1 ',line)
            line = re.sub(r'(?<=\d)(?=(?:training|query|queries|examples|predictions|scores|parameters|epochs|minibatches|hidden|rows|columns|features|seconds|words|bytes|units|ms|mJ|fJ|ticks|ps|pp|mm2|draws)\b)',' ',line)
            line = re.sub(r'\)(?=(?:fJ|ticks|ps)\b)',') ',line)
        formatted.append(line)
    report = '\n'.join(formatted)+'\n'
    (HERE/'report.md').write_text(report)
    historical_row = f'| 2026-09-12 | {acc} | {sig(primary["adjusted_mj"]["median"])} | {sig(primary["cuda_ms"]["median"])} | \u2014 | \u2014 | [Panel-cached MLP (600/600)](submissions/mlp-panels-v4-20260911/report.md) |'
    (HERE/'table-row.md').write_text(historical_row+'\n')
    page = ROOT/'docs/submissions/mlp-panels-v4-20260911/index.html'
    page.parent.mkdir(parents=True,exist_ok=True)
    page.write_text(render_html(report))
    files = {}
    for path in sorted(HERE.rglob('*')):
        rel = path.relative_to(HERE)
        if not path.is_file() or any(part in ('generated','__pycache__','gpu_measured') for part in rel.parts):
            continue
        if path.name=='artifact_manifest.json':
            continue
        files[str(rel)] = sha(path)
    (HERE/'artifact_manifest.json').write_text(json.dumps({'algorithm':'sha256','files':files},indent=2)+'\n')
    print(metric_row)


def render_html(markdown):
    prefix = '../../../mnist/submissions/mlp-panels-v4-20260911/'
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda match:
            '<a href="'+(match[2] if match[2].startswith(('https://','http://','#')) else prefix+match[2])+'">'+match[1]+'</a>',text)
        text = re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    body,paragraph = [],[]
    in_code,in_table,in_list = False,False,False
    for line in markdown.splitlines()+['']:
        if line.startswith('```'):
            if paragraph:
                body.append('<p>'+inline(' '.join(paragraph))+'</p>'); paragraph=[]
            body.append('</code></pre>' if in_code else '<pre><code>')
            in_code = not in_code
            continue
        if in_code:
            body.append(html.escape(line)+'\n'); continue
        if not line.startswith('|') and in_table:
            body.append('</tbody></table></div>'); in_table=False
        if not line.startswith('- ') and in_list:
            body.append('</ul>'); in_list=False
        if not line or line.startswith(('#','|','- ')):
            if paragraph:
                body.append('<p>'+inline(' '.join(paragraph))+'</p>'); paragraph=[]
        if line.startswith('|'):
            cells = [part.strip() for part in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?',cell) for cell in cells):
                continue
            if not in_table:
                body.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in cells)+'</tr></thead><tbody>')
                in_table=True
            else:
                body.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
        elif line.startswith('#'):
            level = len(line)-len(line.lstrip('#'))
            body.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>')
        elif line.startswith('- '):
            if not in_list:
                body.append('<ul>'); in_list=True
            body.append('<li>'+inline(line[2:])+'</li>')
        elif line:
            paragraph.append(line)
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+\
        '<title>'+TITLE+'</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#202428;background:#fff;max-width:1100px;margin:40px auto;padding:0 20px}h1,h2{line-height:1.2}h2{margin-top:2em}a{color:#1557a0}code{font-size:.9em}pre{padding:16px;background:#f4f6f8;overflow:auto}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f4f6f8}@media(max-width:600px){body{margin:20px auto;font-size:15px}}</style></head><body><main>'+''.join(body)+'</main></body></html>\n'


if __name__=='__main__':
    main()
