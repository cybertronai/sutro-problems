"""Run an MNIST energy problem through KernelBot's own evaluation pipeline.

Builds the task with KernelBot's make_task_definition / build_task_config and
executes run_config, either locally (plumbing dry run: CPU tensors and a fake
NVML) or on Modal with KernelBot's runner image on an A100-80GB, the GPU its
leaderboards use. Needs a KernelBot checkout:

    git clone https://github.com/gpu-mode/kernelbot /path/to/kernelbot
    export KERNELBOT_SRC=/path/to/kernelbot/src

    # local dry run (Python with torch, numpy, pyyaml)
    python mnist/kernelbot/kernelbot_run.py --local --submission reference.py \\
        --set error_bp=3000 draws=2 rounds=1 active_s=2 idle_s=1 settle_s=1 reference_s=1

    # Modal (pyyaml and modal installed locally)
    python mnist/kernelbot/kernelbot_run.py --submission submissions/pca_qda.py --output results/pca-qda.json

Overrides given with --set apply to both the test and benchmark cases.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(os.environ.get('KERNELBOT_SRC', ''), ''))


def build_config(problem, submission, mode, overrides):
    from libkernelbot.consts import SubmissionMode
    from libkernelbot.task import build_task_config, make_task_definition
    definition = make_task_definition(HERE / problem / 'task.yml')
    for case in definition.task.tests + definition.task.benchmarks:
        case.update(overrides)
    source = (HERE / submission).read_text()
    return build_task_config(definition.task, source, arch=None, mode=SubmissionMode(mode))


def summarize(result, output, record=None):
    runs = {}
    if record is not None and output:
        path = Path(output).with_name(Path(output).stem + '-record.json')
        path.write_bytes(record)
    for name, eval_result in (result.runs or {}).items():
        run = eval_result.run
        if run is None:
            runs[name] = {'compilation_or_setup_failed': True}
            continue
        values = dict(run.result)
        record = values.pop('energy.record', None)
        if record and output:
            path = Path(output).with_name(Path(output).stem + f'-{name}-record.json')
            path.write_bytes(zlib.decompress(base64.b64decode(record)))
            values['energy.record_saved_to'] = str(path)
        runs[name] = {'exit_code': run.exit_code, 'passed': run.passed, 'duration_s': run.duration,
                      'result': values, 'stderr_tail': (run.stderr or '')[-3000:]}
    summary = {'success': result.success, 'error': result.error, 'system': result.system.__dict__, 'runs': runs}
    if output:
        Path(output).write_text(json.dumps(summary, indent=2, default=str) + '\n')
    return summary


def run_local(config):
    from libkernelbot import run_eval
    run_eval.make_system_info = run_eval.SystemInfo  # the real probe reads Linux-only /proc/cpuinfo
    run_config = run_eval.run_config
    os.environ['MNIST_EVAL_DEVICE'] = 'cpu'
    os.environ['MNIST_EVAL_FAKE_NVML'] = '1'
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        try:
            return run_config(config)
        finally:
            os.chdir(previous)


def direct_eval(config):
    """Diagnostics only: run eval.py once in the given mode with stderr streamed to the Modal logs."""
    import subprocess
    import tempfile
    work = tempfile.mkdtemp()
    for name, content in config['sources'].items():
        Path(work, name).write_text(content)
    cases = config['tests'] if config['mode'] == 'test' else config['benchmarks'][-1:]
    Path(work, 'cases.txt').write_text('\n'.join('; '.join(f'{k}: {v}' for k, v in c.items()) for c in cases))
    read, write = os.pipe()
    process = subprocess.run(['python3', 'eval.py', config['mode'], 'cases.txt'], cwd=work, pass_fds=[write],
                             env={**os.environ, 'POPCORN_FD': str(write)})
    os.close(write)
    return {'returncode': process.returncode, 'result': os.fdopen(read).read()}


def run_and_collect(config):
    """KernelBot's modal_run_config, plus the raw energy record, which is too large for its result pipe."""
    from modal_runner import modal_run_config
    record = Path('energy-record.json')
    record.unlink(missing_ok=True)  # containers can be reused
    result = modal_run_config(config)
    return result, record.read_bytes() if record.exists() else None


def run_modal(config, direct=False):
    import modal
    sys.path.insert(0, os.path.join(os.environ['KERNELBOT_SRC'], 'runners'))
    from modal_runner import MODAL_RUN_TIMEOUT_SECONDS, PCH_MOUNT, app, cuda_image, modal_run_config, pch_volume
    function = app.function(gpu='A100-80GB', image=cuda_image, name='mnist_energy_a100_80gb', serialized=True,
                            restrict_modal_access=True, timeout=MODAL_RUN_TIMEOUT_SECONDS, retries=0,
                            volumes={PCH_MOUNT: pch_volume.with_mount_options(read_only=True)})(
        direct_eval if direct else run_and_collect)
    with modal.enable_output(), app.run():
        return function.remote(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problem', default='medium_5pct')
    parser.add_argument('--submission', required=True, help='path relative to mnist/kernelbot')
    parser.add_argument('--mode', default='leaderboard', choices=['test', 'benchmark', 'leaderboard'])
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=INT')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--direct', action='store_true',
                        help='Modal diagnostics: run eval.py once in --mode, outside run_config, with stderr streamed')
    parser.add_argument('--output', help='summary JSON path; energy records are written beside it')
    args = parser.parse_args()
    if not os.environ.get('KERNELBOT_SRC'):
        parser.error('set KERNELBOT_SRC to a KernelBot checkout\'s src directory')
    if args.output and Path(args.output).exists():
        parser.error('output exists; choose a new path')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    overrides = {key: int(value) for key, value in (item.split('=', 1) for item in args.set)}
    config = build_config(args.problem, args.submission, args.mode, overrides)
    if args.direct:
        result = run_modal(config, direct=True)
        print(result['result'])
        print('returncode', result['returncode'])
        return
    record = None
    if args.local:
        result = run_local(config)
    else:
        result, record = run_modal(config)
    summary = summarize(result, args.output, record)
    for name, run in summary['runs'].items():
        shown = {k: v for k, v in run.items() if k != 'stderr_tail'}
        print(f'== {name}\n{json.dumps(shown, indent=2, default=str)}')
        if not run.get('passed'):
            print(run.get('stderr_tail', ''))
    if summary['error']:
        print('error:', summary['error'])


if __name__ == '__main__':
    main()
