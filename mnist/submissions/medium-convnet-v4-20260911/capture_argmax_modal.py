"""Supplementary same-source regeneration of the final 10,000-row argmax PTX.

This compiles the frozen Backend.argmax specialization in the pinned A100
environment. It does not claim to recover PTX from the original timed launch,
and it loads no training data, labels, learned parameters or predictions.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
    'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
CAPTURE_SOURCES = ['capture_argmax_modal.py', 'ordered_backend/ops.py',
                   'ensemble.py', 'modal_run.py']
image = (modal.Image.from_registry(IMAGE_REF).apt_install('gcc')
         .pip_install('numpy==2.2.6', 'nvidia-ml-py==12.560.30'))
for name in CAPTURE_SOURCES:
    image = image.add_local_file(str(HERE/name), '/root/submission/'+name)
app = modal.App('sutro-mnist-medium-supplemental-final-argmax')


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=4096,
              timeout=180, startup_timeout=300, retries=0)
def capture(expected_sources, frozen_software, protocol_sha256, benchmark_sha256):
    import platform
    import re
    import subprocess
    import sys
    import numpy as np
    import torch
    import triton
    sys.path.insert(0, '/root/submission/ordered_backend')
    from ops import Backend, KW

    actual_sources = {name: hashlib.sha256((Path('/root/submission')/name).read_bytes()).hexdigest()
                      for name in CAPTURE_SOURCES}
    assert actual_sources == expected_sources, 'Capture/frozen source identity changed'
    software = dict(torch=str(torch.__version__), triton=str(triton.__version__),
                    numpy=np.__version__, cuda=torch.version.cuda, image=IMAGE_REF)
    assert software == frozen_software, (software, frozen_software)
    hardware = dict(name=torch.cuda.get_device_name(0),
                    uuid=str(torch.cuda.get_device_properties(0).uuid),
                    compute_capability=list(torch.cuda.get_device_capability(0)))
    assert 'A100' in hardware['name'] and hardware['compute_capability'] == [8, 0]
    assert KW == {'num_warps': 4, 'enable_fp_fusion': False}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    rng = np.random.default_rng(20260911)
    random_values = rng.standard_normal((10000, 10), dtype=np.float32)
    ties = rng.integers(-3, 4, (10000, 10)).astype(np.float32)
    ties[0] = 0
    ties[1] = -4
    ties[2] = -5; ties[2, [2, 7]] = 6
    ties[3] = np.array([-0., 0.] * 5, dtype=np.float32)
    ties[4] = 0; ties[4, [3, 9]] = np.nextafter(np.float32(0), np.float32(1))
    ties[5] = -np.finfo(np.float32).max; ties[5, [1, 8]] = np.finfo(np.float32).max
    ties[6] = -7; ties[6, 9] = 8
    ties[-1] = -9; ties[-1, [4, 9]] = 10
    backend = Backend(); checks = []
    for name, values in [('finite_random', random_values), ('ties_and_boundary_rows', ties)]:
        assert values.shape == (10000, 10) and values.dtype == np.float32 and np.isfinite(values).all()
        inputs = torch.from_numpy(values).cuda()
        outputs = torch.empty(10000, dtype=torch.int64, device='cuda')
        # This is exactly the frozen call used by Ensemble.invoke: B=10000,
        # BLOCK=128, four warps, FP32 input and signed-int64 output.
        backend.argmax(inputs, outputs)
        torch.cuda.synchronize()
        actual = outputs.cpu().numpy()
        expected = values.argmax(axis=1).astype(np.int64)
        np.testing.assert_array_equal(actual, expected)
        checks.append(dict(name=name, rows=10000, all_predictions_match=True,
            input_shape=list(values.shape), input_dtype=str(values.dtype),
            output_shape=list(actual.shape), output_dtype=str(actual.dtype),
            input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            first_eight_predictions=actual[:8].tolist(), last_prediction=int(actual[-1])))
    assert len(backend.compiled) == 1, 'Unexpected additional specialization'
    name, kernel = next(iter(backend.compiled.items()))
    ptx = kernel.asm['ptx']
    audit = dict(fp32_fma_or_mad=bool(re.search(r'\b(?:fma|mad)(?:\.[a-z0-9]+)*\.f32\b', ptx)),
                 flush_to_zero=bool(re.search(r'\.ftz(?:\.|\b)', ptx)),
                 tensor_core=bool(re.search(r'\b(?:mma|wmma|wgmma)\.', ptx)))
    assert not any(audit.values()), audit
    triton_root = Path(triton.__file__).resolve().parent
    compiler_files = {name: hashlib.sha256((triton_root/name).read_bytes()).hexdigest()
                      for name in ('compiler/compiler.py', 'backends/nvidia/compiler.py')}
    ptxas = triton_root/'backends/nvidia/bin/ptxas'
    compiler = dict(python=platform.python_version(), source_sha256=compiler_files,
                    kernel_name=kernel.name, specialization={'B':10000, 'BLOCK':128},
                    launch_grid=[79], launch_options=KW,
                    ptx_version=re.search(r'^\.version\s+([^\n]+)', ptx, re.M).group(1),
                    ptx_target=re.search(r'^\.target\s+([^\n]+)', ptx, re.M).group(1),
                    metadata=json.loads(json.dumps(kernel.metadata._asdict(), default=str)))
    if ptxas.is_file():
        compiler['bundled_ptxas_sha256'] = hashlib.sha256(ptxas.read_bytes()).hexdigest()
        compiler['bundled_ptxas_version'] = subprocess.check_output([str(ptxas), '--version'], text=True).strip()
    result = dict(schema_version=1, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Separate same-source regeneration of the final 10000-row argmax specialization; not PTX captured from the original timed invocation. No full learner or timing was rerun.',
        source_sha256=actual_sources, protocol_sha256=protocol_sha256,
        original_benchmark_result_sha256=benchmark_sha256, software=software, hardware=hardware,
        compiler=compiler, checks=checks, all_passed=True, instruction_audit=audit,
        ptx_file=name+'.ptx', ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
        dataset_loaded=False, learned_parameters_loaded=False, original_predictions_loaded=False)
    assert actual_sources == {name: hashlib.sha256((Path('/root/submission')/name).read_bytes()).hexdigest()
                              for name in CAPTURE_SOURCES}
    # Return plain JSON so the receiving CPU environment need not import Triton
    # to unpickle compiler-specific metadata classes such as GPUTarget.
    return json.dumps(result, allow_nan=False), ptx


@app.local_entrypoint()
def main():
    protocol_path = HERE/'protocol.json'
    benchmark_path = HERE/'benchmark/results/draw-00.json'
    protocol = json.loads(protocol_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    sources = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in CAPTURE_SOURCES}
    for name in CAPTURE_SOURCES[1:]:
        assert sources[name] == protocol['source_sha256'][name] == benchmark['source_sha256'][name]
    output = HERE/'supplemental-argmax'
    assert not output.exists() or not any(output.iterdir()), 'Refusing to overwrite retained supplemental evidence'
    result_text, ptx = capture.remote(sources, benchmark['software'],
        hashlib.sha256(protocol_path.read_bytes()).hexdigest(), hashlib.sha256(benchmark_path.read_bytes()).hexdigest())
    result = json.loads(result_text)
    assert sources == {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in CAPTURE_SOURCES}
    assert result['all_passed'] and result['source_sha256'] == sources
    assert hashlib.sha256(ptx.encode()).hexdigest() == result['ptx_sha256']
    output.mkdir(parents=True, exist_ok=True)
    (output/result['ptx_file']).write_text(ptx)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'all_passed':True, 'output':str(output), 'ptx_sha256':result['ptx_sha256'],
                      'scope':'Supplementary same-source regeneration; original timed-run evidence remains unchanged.'}))
