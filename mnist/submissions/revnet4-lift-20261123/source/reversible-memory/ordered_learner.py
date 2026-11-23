"""Fresh exact-recipe training with ordered reconstruction backward.

The selected lossless ReLU-lift architecture and all training hyperparameters
are unchanged. Forward kernels, initialization, schedules, and SGD are the
original ordered FP32 operations. Reconstructed intermediate states can differ
by floating-point rounding, so this is a new accuracy measurement, not a claim
of bitwise equivalence to stored-activation training.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import time

import numpy as np

from ordered_models import OrderedReconstructionTrainer, workspace_storage_bytes


HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'reversible-refinement'
PREVIOUS=HERE.parent/'procedure-transfer'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    array=np.asarray(array)
    return hashlib.sha256(np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'),copy=False)).tobytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def previous_learner():
    spec=importlib.util.spec_from_file_location('ordered_memory_previous_learner',PREVIOUS/'learner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def config_for(architecture):
    if architecture not in ('rev32_b4_lift_relu','rev32_b8_lift_relu'):
        raise ValueError('Accuracy runs are restricted to frozen selected-study architectures')
    protocol=read(SOURCE/'protocol.json')
    for name in ('models.py','extra_ops.py'):
        if sha(SOURCE/name)!=protocol['source_sha256'][name]:
            raise RuntimeError(f'Frozen selected architecture source changed: {name}')
    return protocol['architecture_configs'][architecture].copy()


def memory_breakdown(trainer):
    import torch
    net=trainer.network
    groups={
        'parameters':net.params,
        'parameter_gradients':net.grad,
        'momentum':net.velocity,
        'seed_initialization_copies':net.initial,
        'training_workspaces':trainer.workspaces,
        'inference_workspaces':trainer.query_workspaces,
        'precomputed_schedules':{name:getattr(trainer,name) for name in ('indices','coefficients','order','offset')},
        'training_and_query_data':{name:getattr(trainer,name) for name in ('raw','labels','query')},
        'outputs':{name:getattr(trainer,name) for name in ('output_scores','predictions')},
    }
    counts={name:workspace_storage_bytes(value)['allocated_storage_bytes'] for name,value in groups.items()}
    tracked=workspace_storage_bytes(groups)['allocated_storage_bytes']
    allocated=torch.cuda.memory_allocated()
    return {'tensor_storage_bytes_by_category':counts,'tracked_unique_tensor_storage_bytes':tracked,
            'current_cuda_allocated_bytes':allocated,'allocated_bytes_not_in_named_tensor_groups':allocated-tracked,
            'scope':'Actual unique tensor storages; includes original precomputed schedules and graph workspaces. CUDA allocator peaks measured separately.'}


def train_predict(train_images,train_labels,test_images,draw_seed,architecture='rev32_b4_lift_relu'):
    started=time.perf_counter()
    previous=previous_learner();provenance=previous.verify_reference()
    arrays=previous._validate_inputs(train_images,train_labels,test_images,draw_seed)
    previous._reference_trainer()
    import torch
    import triton
    if not torch.cuda.is_available():raise RuntimeError('Ordered reconstruction requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    config=config_for(architecture)
    n_train,n_test=len(arrays['train_images']),len(arrays['test_images'])
    config.update(n_train=n_train,n_test=n_test)
    ensemble=np.zeros((n_test,10),np.float32)
    members=[];ptx={};training_seconds=0.
    for seed in config['member_seeds']:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        start_allocated=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        trainer=OrderedReconstructionTrainer(arrays['train_images'],arrays['train_labels'],config,seed)
        before_prepare=torch.cuda.memory_allocated()
        trainer.prepare(arrays['test_images'])
        torch.cuda.synchronize()
        after_prepare=torch.cuda.memory_allocated()
        prepare_peak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter();trainer.invoke();torch.cuda.synchronize()
        elapsed=time.perf_counter()-began;training_seconds+=elapsed
        invoke_peak=torch.cuda.max_memory_allocated()
        invoke_reserved=torch.cuda.max_memory_reserved()
        # Measure before state serialization and all inverse diagnostics.
        memory={'start_allocated_bytes':start_allocated,'before_prepare_allocated_bytes':before_prepare,
                'after_prepare_allocated_bytes':after_prepare,'prepare_peak_allocated_bytes':prepare_peak,
                'invoke_peak_allocated_bytes':invoke_peak,'invoke_peak_reserved_bytes':invoke_reserved,
                'end_to_end_peak_allocated_bytes':max(prepare_peak,invoke_peak),
                'end_to_end_peak_incremental_bytes':max(prepare_peak,invoke_peak)-start_allocated,
                'measured_before_diagnostics':True,'breakdown_after_invoke':memory_breakdown(trainer)}
        output=trainer.outputs();scores=output['scores']
        if scores.shape!=(n_test,10) or scores.dtype!=np.float32 or not np.isfinite(scores).all():
            raise RuntimeError(f'Nonfinite/invalid scores: {architecture} member {seed}')
        if not all(np.isfinite(value).all() for value in output['parameters'].values()):
            raise RuntimeError(f'Nonfinite parameters: {architecture} member {seed}')
        ensemble=np.add(ensemble,scores,dtype=np.float32)
        probe=trainer.network.workspace(min(8,n_train),training=False)
        trainer.network.backend.normalize(torch.from_numpy(arrays['train_images'][:8]).cuda(),probe['x'])
        trainer.network.forward(probe)
        recovered=trainer.network.inverse_core(probe['v']['core_out'])
        actual=recovered.cpu().numpy().astype(np.float64)
        truth=probe['v']['core_in'].cpu().numpy().astype(np.float64)
        original=probe['x'].cpu().numpy().astype(np.float64)
        inverse={'scope':'Trained coupling core on first eight unaugmented training images',
                 'max_abs_error':float(np.max(np.abs(actual-truth))),
                 'relative_l2_error':float(np.linalg.norm(actual-truth)/max(np.linalg.norm(truth),1e-30)),
                 'input_max_abs_error':float(np.max(np.abs(actual[:,:1]-original))),
                 'all_finite':bool(np.isfinite(actual).all()),'input_recoverable':True}
        members.append({'seed':seed,'fresh_state':True,'epochs':config['epochs'],
                        'parameter_count':sum(value.size for value in output['parameters'].values()),
                        'minibatches_per_epoch':(n_train+config['batch_size']-1)//config['batch_size'],
                        'final_batch_size':(n_train-1)%config['batch_size']+1,
                        'initial_parameter_sha256':{k:ah(v) for k,v in trainer.initial_arrays.items()},
                        'final_parameter_sha256':{k:ah(v) for k,v in output['parameters'].items()},
                        'final_velocity_sha256':{k:ah(v) for k,v in output['velocities'].items()},
                        'schedule_manifests':trainer.schedule_manifests,'logits_sha256':ah(scores),
                        'training_prediction_seconds':elapsed,'trained_inverse':inverse,'memory':memory})
        for name,kernel in trainer.network.backend.compiled.items():
            assembly=kernel.asm['ptx']
            if re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b',assembly) or '.ftz.' in assembly or re.search(r'\b(?:mma|wmma)\.',assembly):
                raise RuntimeError(f'Ordered arithmetic changed: {name}')
            ptx[name]=hashlib.sha256(assembly.encode()).hexdigest()
        print(f'Ordered reconstruction {architecture} draw{draw_seed} seed{seed}: eight fresh epochs ({elapsed:.1f}s)',flush=True)
        del trainer,output,scores,probe,recovered,actual,truth,original
        gc.collect();torch.cuda.empty_cache()
    predictions=ensemble.argmax(axis=1).astype(np.int64)
    metadata={'architecture':architecture,'backpropagation':'ordered_reconstruction','draw_seed':int(draw_seed),'config':config,
              'input_sha256':{k:ah(v) for k,v in arrays.items()},'training_examples':n_train,'test_examples':n_test,
              'members':members,'member_seeds':config['member_seeds'],'test_labels_supplied':False,
              'checkpoint_loaded':False,'fresh_state_per_member':True,'reference_sha256':provenance['files'],
              'source_sha256':{n:sha(HERE/n) for n in ('ordered_models.py','ordered_learner.py')},
              'selected_stored_source_sha256':{n:sha(SOURCE/n) for n in ('models.py','extra_ops.py')},
              'selected_stored_protocol_sha256':sha(SOURCE/'protocol.json'),
              'predictions_sha256':ah(predictions),'ensemble_logits_sha256':ah(ensemble),
              'training_seconds':training_seconds,'elapsed_including_preparation_seconds':time.perf_counter()-started,
              'memory_scope':'Synchronized per-member allocated-byte peaks before preparation and during invocation, measured before diagnostics; original schedule tensors included and separately itemized',
              'ptx_sha256':ptx,'ptx_has_fp32_fma':False,'ptx_has_ftz':False,'ptx_has_tensorcore':False,
              'hardware':{'gpu':torch.cuda.get_device_name()},
              'software':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),'triton':str(triton.__version__),'cuda':torch.version.cuda}}
    return predictions,metadata
