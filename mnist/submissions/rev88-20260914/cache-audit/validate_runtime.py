"""One-worker independent numerical/API/CLI validation of cache_runtime."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json

import modal

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image=(modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
       .env({'PYTHONPATH':'/workspace/rev88/cache-audit:/workspace/rev88'}))
if modal.is_local():
    image=image.add_local_file(BASE/'learner.py','/workspace/rev88/learner.py')
    image=image.add_local_file(HERE/'cache_runtime.py','/workspace/rev88/cache-audit/cache_runtime.py')
app=modal.App('sutro-rev88-cache-runtime-validation-20260914')


@app.function(image=image,gpu='A100-40GB',cpu=4,memory=16384,max_containers=1,
              min_containers=0,buffer_containers=0,scaledown_window=2,timeout=600,retries=0)
def validate(config):
    import os
    import subprocess
    import sys
    import tempfile
    import numpy as np
    import cache_runtime as runtime
    import torch
    torch.set_num_threads(4)
    numerical=runtime.validate_cuda_replay(config)
    assert numerical['passed']
    generator=np.random.Generator(np.random.PCG64(9140914))
    arrays={'train_images':generator.random((19,1,9,9),dtype=np.float32),
            'train_labels':generator.integers(0,10,19,dtype=np.int64),
            'test_images':generator.random((7,1,9,9),dtype=np.float32)}
    tiny={**config,'epochs':2,'batch_size':8}
    task=runtime.prepare_task(**arrays,config=tiny)
    task.run();predictions,api=task.outputs()
    task.run();repeat_predictions,repeat=task.outputs()
    assert np.array_equal(predictions,repeat_predictions)
    for key in ('final_parameter_sha256','final_velocity_sha256','scores_sha256'):
        assert api[key]==repeat[key],key
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        np.savez(root/'input.npz',**arrays)
        (root/'config.json').write_text(json.dumps(tiny))
        command=[sys.executable,'/workspace/rev88/cache-audit/cache_runtime.py',
                 '--input',str(root/'input.npz'),'--config',str(root/'config.json'),
                 '--output',str(root/'result.json')]
        process=subprocess.run(command,text=True,capture_output=True,timeout=120)
        assert process.returncode==0,process.stderr
        cli=json.loads((root/'result.json').read_text())
        cli_predictions=np.load(root/'result.predictions.npy',allow_pickle=False)
        assert np.array_equal(predictions,cli_predictions)
        for key in ('final_parameter_sha256','final_velocity_sha256','scores_sha256'):
            assert api[key]==cli['metadata'][key],key
        np.savez(root/'forbidden.npz',**arrays,test_labels=np.zeros(7,dtype=np.int64))
        invalid=command.copy();invalid[invalid.index('--input')+1]=str(root/'forbidden.npz')
        rejected=subprocess.run(invalid,text=True,capture_output=True,timeout=60)
        assert rejected.returncode!=0 and 'exactly the three' in rejected.stderr
        guard=subprocess.run([sys.executable,'-c',
            'import torch; torch.cuda.init(); import cache_runtime'],text=True,capture_output=True,timeout=60)
        assert guard.returncode!=0 and 'before CUDA initialization' in guard.stderr
    return {'passed':True,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
            'gpu':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),
            'workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'numerical_validation':numerical,
            'api_cli_smoke':{'passed':True,'train_examples':19,'query_examples':7,'batch_sizes':[8,8,3],
                             'predictions_equal':True,'final_parameters_equal':True,'final_velocity_equal':True,
                             'scores_equal':True,'repeated_api_replay_byte_equal':True,
                             'query_label_archive_rejected':True,'late_cuda_initialization_rejected':True,
                             'config':tiny,'metadata':api},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path('/workspace/rev88/learner.py'),Path('/workspace/rev88/cache-audit/cache_runtime.py'))}}


@app.local_entrypoint()
def main():
    config=json.loads((BASE/'config.json').read_text())
    result=validate.remote(config)
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['execution']={'app_id':app.app_id,'max_gpu_workers':1,'worker_finished':True,
                         'lifecycle':'Ephemeral Modal run; CLI exit stops this validation app.'}
    (HERE/'cuda-validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS',result['numerical_validation']['checks'],'numerical comparisons plus API/CLI/replay/input guards',flush=True)
