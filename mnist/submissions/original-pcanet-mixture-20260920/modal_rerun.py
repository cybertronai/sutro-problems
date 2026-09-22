"""One isolated A100, full-split checks and energy; no persistent resources."""
from pathlib import Path
import json
import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.from_registry('pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel')
         .pip_install('numpy==2.1.2', 'nvidia-ml-py==13.610.43', 'ninja==1.11.1.1')
         .env({'MAX_JOBS': '2', 'TORCH_CUDA_ARCH_LIST': '8.0'})
         .add_local_file(HERE / 'model.py', '/workspace/model.py')
         .add_local_file(HERE / 'cuda_kernel.py', '/workspace/cuda_kernel.py')
         .add_local_file(HERE / 'data.py', '/workspace/data.py')
         .add_local_file(HERE / 'validate.py', '/workspace/validate.py'))
app = modal.App('sutro-pcanet-k100-validation')


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=32768,
              timeout=1800, max_containers=1, min_containers=0,
              scaledown_window=2, retries=0)
def execute():
    import subprocess
    result = subprocess.run(['python', '-u', '/workspace/validate.py',
                             '--raw', '/tmp/raw', '--output', '/tmp/results'],
                            cwd='/workspace', text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    files = {p.name: p.read_bytes() for p in Path('/tmp/results').glob('*') if p.is_file()}
    files['run.log'] = result.stdout.encode()
    files['execution.json'] = json.dumps({'returncode': result.returncode}).encode()
    return result.returncode, files


@app.local_entrypoint()
def main(output: str = 'generated/rerun'):
    out = HERE / output
    out.mkdir(parents=True, exist_ok=False)
    code, files = execute.remote()
    for name, data in files.items():
        (out / name).write_bytes(data)
    assert code == 0, f'Rerun failed; see {out / "run.log"}'
    print('Saved', out)
