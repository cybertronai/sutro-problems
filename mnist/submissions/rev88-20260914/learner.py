"""Small reversible MLP, fresh SGD training, and GPU-resident task replay.

Only the 81 supplied normalized pixels enter the network. Padding one zero
gives an injective 82-coordinate input, split into two 41-coordinate halves.
The classifier is not reversible. The coupling core reconstructs its states
during backward and retains only its final state plus parameter references.

Preparation/CPU seed construction/transfers/capture are outside run(). Each
run() resets state, normalizes raw inputs, trains every epoch, and predicts.
The permutations are seed-only and already resident on the GPU; every epoch
copies its permutation device-to-device into the graph's fixed order buffer.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


class Reconstruct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha, *weights):
        left, right = inputs.chunk(2, dim=1)
        for i in range(0, len(weights), 2):
            left = left + alpha * F.linear(F.relu(right), weights[i])
            right = right + alpha * F.linear(F.relu(left), weights[i + 1])
        output = torch.cat((left, right), dim=1)
        ctx.alpha = alpha
        ctx.save_for_backward(output, *weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        output, *weights = ctx.saved_tensors
        alpha = ctx.alpha
        left, right = output.detach().chunk(2, dim=1)
        dleft, dright = gradient.chunk(2, dim=1)
        parameter_gradients = [None] * len(weights)
        for i in range(len(weights) - 2, -1, -2):
            # At most one branch's ordinary autograd graph exists at a time.
            with torch.enable_grad():
                branch_input = left.detach().requires_grad_(True)
                branch = alpha * F.linear(F.relu(branch_input), weights[i + 1])
                derivative, dw = torch.autograd.grad(
                    branch, (branch_input, weights[i + 1]), dright)
            right = right - branch.detach()
            dleft = dleft + derivative
            parameter_gradients[i + 1] = dw
            with torch.enable_grad():
                branch_input = right.detach().requires_grad_(True)
                branch = alpha * F.linear(F.relu(branch_input), weights[i])
                derivative, dw = torch.autograd.grad(
                    branch, (branch_input, weights[i]), dleft)
            left = left - branch.detach()
            dright = dright + derivative
            parameter_gradients[i] = dw
        return (torch.cat((dleft, dright), dim=1), None, *parameter_gradients)


class Model(nn.Module):
    def __init__(self, depth=1, alpha=.5, seed=11):
        super().__init__()
        if depth not in (1, 2):
            raise ValueError('The bounded candidate family has depth 1 or 2')
        self.depth, self.alpha = int(depth), float(alpha)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.weights = nn.ParameterList(
                nn.Parameter(torch.empty(41, 41)) for _ in range(2 * depth))
            for weight in self.weights:
                nn.init.normal_(weight, mean=0., std=.05 / math.sqrt(41))
            self.head = nn.Linear(82, 10, bias=True)

    def core(self, inputs, reconstruct=True):
        if reconstruct and torch.is_grad_enabled():
            return Reconstruct.apply(inputs, self.alpha, *self.weights)
        left, right = inputs.chunk(2, dim=1)
        for i in range(0, len(self.weights), 2):
            left = left + self.alpha * F.linear(F.relu(right), self.weights[i])
            right = right + self.alpha * F.linear(F.relu(left), self.weights[i + 1])
        return torch.cat((left, right), dim=1)

    def forward(self, inputs, reconstruct=True):
        return self.head(self.core(inputs, reconstruct=reconstruct))

    @torch.no_grad()
    def inverse(self, output):
        left, right = output.chunk(2, dim=1)
        for i in range(len(self.weights) - 2, -1, -2):
            right = right - self.alpha * F.linear(F.relu(left), self.weights[i + 1])
            left = left - self.alpha * F.linear(F.relu(right), self.weights[i])
        return torch.cat((left, right), dim=1)


def validate_reconstruction():
    """Independent ordinary-autograd comparison; CPU only, before fitting."""
    records = []
    for dtype in (torch.float64, torch.float32):
        for depth in (1, 2):
            model = Model(depth=depth, seed=199).to(dtype=dtype)
            generator = torch.Generator(device='cpu').manual_seed(20260914)
            original = torch.rand((3, 82), generator=generator, dtype=dtype) * 4 - .5
            original[:, -1] = 0.
            # Differentiate the 81 real input coordinates. The appended zero
            # is constant; a derivative at its ReLU kink is not a data gradient.
            a = original[:,:81].clone().requires_grad_(True)
            b = original[:,:81].clone().requires_grad_(True)
            a_lift, b_lift = F.pad(a,(0,1)), F.pad(b,(0,1))
            labels = torch.tensor([2, 7, 1], dtype=torch.int64)
            stored = model(a_lift, reconstruct=False)
            reconstructed = model(b_lift, reconstruct=True)
            params = tuple(model.parameters())
            ga = torch.autograd.grad(F.cross_entropy(stored, labels), (a, *params))
            gb = torch.autograd.grad(F.cross_entropy(reconstructed, labels), (b, *params))
            tolerance = (2e-12, 2e-10) if dtype == torch.float64 else (2e-6, 3e-3)
            if not torch.equal(stored, reconstructed):
                raise AssertionError('Stored/reconstructed forward outputs differ')
            for reference, actual in zip(ga, gb):
                torch.testing.assert_close(actual, reference, atol=tolerance[0], rtol=tolerance[1])
            core = model.core(b_lift)
            saved = core.grad_fn.saved_tensors
            if len(saved) != 1 + 2 * depth:
                raise AssertionError('Unexpected reconstruction saved-state count')
            if saved[0].untyped_storage().data_ptr() != core.untyped_storage().data_ptr():
                raise AssertionError('The saved activation is not the core endpoint')
            if any(saved[i + 1] is not weight for i, weight in enumerate(model.weights)):
                raise AssertionError('A hidden activation replaced a parameter reference')
            recovered = model.inverse(core)
            torch.testing.assert_close(recovered, b_lift, atol=tolerance[0], rtol=tolerance[1])
            records.append({'dtype': str(dtype), 'depth': depth, 'passed': True,
                            'forward_bitwise_equal': True,
                            'maximum_gradient_abs_error': max(float((x-y).abs().max()) for x,y in zip(ga,gb)),
                            'inverse_max_abs_error': float((recovered-b_lift).abs().max()),
                            'saved_activation_storages': 1,
                            'saved_activation_bytes': core.numel() * core.element_size(),
                            'saved_parameter_references': 2 * depth})
    return {'passed': True, 'device': 'cpu', 'checks': records,
            'scope': 'Identical weights; ordinary autograd versus custom reconstruction for inputs and every parameter. First-order differentiation only.'}


def _config(config):
    defaults = dict(depth=1, alpha=.5, seed=11, epochs=4, learning_rate=.1,
                    batch_size=128, momentum=.9, weight_decay=0.)
    defaults.update(config)
    for name in ('depth', 'seed', 'epochs', 'batch_size'):
        defaults[name] = int(defaults[name])
    for name in ('alpha', 'learning_rate', 'momentum', 'weight_decay'):
        defaults[name] = float(defaults[name])
    if defaults['depth'] not in (1, 2) or min(defaults['epochs'], defaults['batch_size']) <= 0:
        raise ValueError('Invalid candidate dimensions')
    if not all(math.isfinite(defaults[k]) for k in ('alpha','learning_rate','momentum','weight_decay')):
        raise ValueError('Nonfinite hyperparameter')
    return defaults


class PreparedTask:
    def __init__(self, train_images, train_labels, test_images, config):
        started = time.perf_counter()
        self.config = c = _config(config)
        arrays = [np.asarray(a) for a in (train_images, train_labels, test_images)]
        train, labels, query = arrays
        if train.shape[1:] != (1,9,9) or query.shape[1:] != (1,9,9):
            raise ValueError('Expected N,1,9,9 images')
        if train.dtype != np.float32 or query.dtype != np.float32:
            raise ValueError('Expected FP32 images')
        if not len(train) or not len(query) or labels.shape != (len(train),):
            raise ValueError('Invalid sample count or label shape')
        if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels > 9).any():
            raise ValueError('Training labels must be integer digits')
        if not all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in (train,query)):
            raise ValueError('Input pixels must be finite in [0,1]')
        self.validation = validate_reconstruction()
        if not torch.cuda.is_available():
            raise RuntimeError('Training and CUDA graph replay require CUDA')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.input_hashes = dict(zip(('train_images','train_labels','test_images'),map(array_hash,arrays)))
        self.model = Model(c['depth'], c['alpha'], c['seed']).cuda()
        self.parameters = list(self.model.parameters())
        self.initial = [p.detach().clone() for p in self.parameters]
        self.velocity = [torch.zeros_like(p) for p in self.parameters]
        for p in self.parameters:
            p.grad = torch.zeros_like(p)
        self.raw = torch.from_numpy(np.ascontiguousarray(train.reshape(-1,81))).cuda()
        self.raw_query = torch.from_numpy(np.ascontiguousarray(query.reshape(-1,81))).cuda()
        self.labels = torch.from_numpy(labels.astype(np.int64)).cuda()
        self.x = torch.empty((len(train),82), dtype=torch.float32, device='cuda')
        self.q = torch.empty((len(query),82), dtype=torch.float32, device='cuda')
        generator = np.random.Generator(np.random.PCG64(c['seed']))
        orders = np.stack([generator.permutation(len(train)) for _ in range(c['epochs'])]).astype(np.int64)
        self.order_hashes = [array_hash(order) for order in orders]
        self.orders = torch.from_numpy(orders).cuda()
        self.order = self.orders[0].clone()
        self.predictions = torch.empty(len(query), dtype=torch.int64, device='cuda')
        self.scores = torch.empty((len(query),10), dtype=torch.float32, device='cuda')
        self.initial_parameter_hashes = {name:array_hash(p.detach().cpu().numpy()) for name,p in self.model.named_parameters()}
        self.run_count = 0
        self._capture()
        torch.cuda.synchronize()
        self.preparation_seconds = time.perf_counter() - started

    @torch.no_grad()
    def _reset_eager(self):
        for p, initial, velocity in zip(self.parameters,self.initial,self.velocity):
            p.copy_(initial)
            p.grad.zero_()
            velocity.zero_()
        self.x[:,:81].copy_(self.raw)
        self.x[:,:81].mul_(4.).sub_(.5)
        self.x[:,81].zero_()
        self.q[:,:81].copy_(self.raw_query)
        self.q[:,:81].mul_(4.).sub_(.5)
        self.q[:,81].zero_()

    def _batch(self, first, last):
        indices = self.order[first:last]
        inputs = torch.index_select(self.x,0,indices)
        labels = torch.index_select(self.labels,0,indices)
        for parameter in self.parameters:
            parameter.grad.zero_()
        loss = F.cross_entropy(self.model(inputs), labels)
        loss.backward()
        with torch.no_grad():
            for p, velocity in zip(self.parameters,self.velocity):
                velocity.mul_(self.config['momentum']).add_(p.grad)
                if self.config['weight_decay']:
                    velocity.add_(p,alpha=self.config['weight_decay'])
                p.add_(velocity,alpha=-self.config['learning_rate'])

    def _epoch_eager(self):
        n, batch = len(self.x), self.config['batch_size']
        for first in range(0,n,batch):
            self._batch(first,min(first+batch,n))

    @torch.no_grad()
    def _predict_eager(self):
        for first in range(0,len(self.q),1024):
            last = min(first+1024,len(self.q))
            self.scores[first:last].copy_(self.model(self.q[first:last]))
        torch.argmax(self.scores,dim=1,out=self.predictions)

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._reset_eager()
            n, batch = len(self.x), self.config['batch_size']
            self._batch(0,min(batch,n))
            if n % batch:
                self._batch(n-n%batch,n)
            self._predict_eager()
            self._reset_eager()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.reset_graph = torch.cuda.CUDAGraph()
        self.epoch_graph = torch.cuda.CUDAGraph()
        self.predict_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reset_graph,stream=stream):
            self._reset_eager()
        with torch.cuda.graph(self.epoch_graph,stream=stream):
            self._epoch_eager()
        with torch.cuda.graph(self.predict_graph,stream=stream):
            self._predict_eager()
        self.reset_graph.replay()
        torch.cuda.synchronize()

    def reset(self):
        """Reset exact seed state and normalize both raw input arrays."""
        self.reset_graph.replay()

    def run(self):
        """Asynchronous complete GPU-resident task; caller synchronizes timing."""
        self.reset()
        for epoch in range(self.config['epochs']):
            self.order.copy_(self.orders[epoch])
            self.epoch_graph.replay()
        self.predict_graph.replay()
        self.run_count += 1

    def state_vector(self):
        """CPU state for diagnostics outside the measured invocation."""
        return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in self.parameters])

    def outputs(self):
        torch.cuda.synchronize()
        predictions = self.predictions.cpu().numpy().copy()
        scores = self.scores.cpu().numpy().copy()
        parameters = {name:array_hash(p.detach().cpu().numpy()) for name,p in self.model.named_parameters()}
        state = self.state_vector()
        if not np.isfinite(scores).all() or not np.isfinite(state).all():
            raise RuntimeError('Nonfinite scores or model state')
        velocities = {name:array_hash(v.detach().cpu().numpy()) for (name,_),v in zip(self.model.named_parameters(),self.velocity)}
        metadata = {'config':self.config, 'parameter_count':sum(p.numel() for p in self.parameters),
                    'train_count':len(self.x),'test_count':len(self.q),
                    'fresh_initialization_per_run':True,'checkpoint_loaded':False,
                    'normalization':'4*x-0.5 then append one zero coordinate',
                    'initialization':'branch normal(0,0.05/sqrt(41)); standard torch Linear head; seed-only',
                    'optimizer':'SGD momentum; zero initial velocity; gradient mean cross entropy',
                    'shuffle':'independent PCG64 permutation each epoch, seed-only',
                    'augmentation':'none','input_hashes':self.input_hashes,
                    'epoch_permutation_sha256':self.order_hashes,
                    'initial_parameter_sha256':self.initial_parameter_hashes,
                    'final_parameter_sha256':parameters,'final_velocity_sha256':velocities,
                    'state_vector_sha256':array_hash(state),
                    'predictions_sha256':array_hash(predictions),'scores_sha256':array_hash(scores),
                    'preparation_seconds':self.preparation_seconds,'run_count':self.run_count,
                    'validation':self.validation,'reversible_core':True,
                    'retained_core_states':1,'core_saved_bytes_per_example':82*4,
                    'head_reversible':False,'input_lift_injective':True,
                    'timing_scope':'run resets weights/gradients/momentum, normalizes raw train and query pixels, copies resident epoch permutations, trains and predicts. Excludes setup, CPU permutation generation, allocation, transfer, capture and diagnostics.'}
        return predictions, metadata


def prepare_task(train_images,train_labels,test_images,config):
    return PreparedTask(train_images,train_labels,test_images,config)


def train_predict(train_images,train_labels,test_images,config):
    task = prepare_task(train_images,train_labels,test_images,config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    task.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    predictions, metadata = task.outputs()
    metadata['training_prediction_seconds'] = elapsed
    return predictions, metadata


def validate_cuda_replay(config=None):
    """Bounded GPU audit, explicitly invoked outside fitting/timing.

    Compare every parameter gradient/update and momentum value against
    ordinary eager autograd + torch.optim.SGD, then compare complete captured
    task outputs and repeated resets. Both candidate depths are covered.
    This is not called recursively by PreparedTask.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA replay validation requires CUDA')
    generator = np.random.Generator(np.random.PCG64(913413))
    train = generator.random((19,1,9,9),dtype=np.float32)
    labels = generator.integers(0,10,19,dtype=np.int64)
    query = generator.random((7,1,9,9),dtype=np.float32)
    base = _config(config or {})
    records = []
    checks = 0
    maximum_error = 0.

    def close(actual,reference,kind):
        nonlocal checks, maximum_error
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=3e-3,
                                   msg=lambda message:kind+': '+message)
        maximum_error = max(maximum_error,float((actual-reference).abs().max()))
        checks += 1

    for depth in (1,2):
        candidate = {**base,'depth':depth,'epochs':2,'batch_size':8,'seed':781}
        task = prepare_task(train,labels,query,candidate)
        reference = Model(depth,candidate['alpha'],candidate['seed']).cuda()
        for parameter,initial in zip(reference.parameters(),task.initial):
            parameter.data.copy_(initial)
        optimizer = torch.optim.SGD(reference.parameters(),lr=candidate['learning_rate'],
                                    momentum=candidate['momentum'],
                                    weight_decay=candidate['weight_decay'],foreach=False)
        task.reset()
        torch.cuda.synchronize()
        expected_initial = {name:array_hash(parameter.detach().cpu().numpy())
                            for name,parameter in task.model.named_parameters()}
        if expected_initial != task.initial_parameter_hashes:
            raise AssertionError('CUDA reset did not restore seed parameters exactly')
        for epoch in range(2):
            task.order.copy_(task.orders[epoch])
            for first in range(0,len(train),8):
                last = min(first+8,len(train))
                indices = task.orders[epoch,first:last]
                raw = torch.index_select(task.raw,0,indices)
                # Independent eager normalization/padding and ordinary core.
                inputs = F.pad(raw * 4. - .5,(0,1))
                expected_labels = torch.index_select(task.labels,0,indices)
                optimizer.zero_grad(set_to_none=False)
                loss = F.cross_entropy(reference(inputs,reconstruct=False),expected_labels)
                loss.backward()
                optimizer.step()
                task._batch(first,last)
                for index,(actual,expected) in enumerate(zip(task.parameters,reference.parameters())):
                    close(actual.grad,expected.grad,f'depth{depth}/gradient{index}')
                    close(actual,expected,f'depth{depth}/parameter{index}')
                    expected_velocity = optimizer.state[expected].get('momentum_buffer')
                    if expected_velocity is not None:
                        close(task.velocity[index],expected_velocity,f'depth{depth}/velocity{index}')
        with torch.no_grad():
            expected_scores = reference(F.pad(task.raw_query*4.-.5,(0,1)),reconstruct=False)
            expected_predictions = expected_scores.argmax(dim=1)
        task.run()
        torch.cuda.synchronize()
        close(task.scores,expected_scores,f'depth{depth}/captured_scores')
        if not torch.equal(task.predictions,expected_predictions):
            raise AssertionError('Captured/eager predictions disagree')
        for index,(actual,expected) in enumerate(zip(task.parameters,reference.parameters())):
            close(actual,expected,f'depth{depth}/captured_parameter{index}')
            close(actual.grad,expected.grad,f'depth{depth}/captured_gradient{index}')
            if candidate['momentum']:
                close(task.velocity[index],optimizer.state[expected]['momentum_buffer'],
                      f'depth{depth}/captured_velocity{index}')
        _, first_output = task.outputs()
        task.run()
        _, second_output = task.outputs()
        for key in ('final_parameter_sha256','final_velocity_sha256','scores_sha256','predictions_sha256'):
            if first_output[key] != second_output[key]:
                raise AssertionError('Complete replay was not byte-repeatable: '+key)
        features = task.x[:3].detach().clone().requires_grad_(True)
        core = task.model.core(features)
        saved = core.grad_fn.saved_tensors
        if len(saved) != 1+2*depth or saved[0].data_ptr() != core.data_ptr():
            raise AssertionError('CUDA core endpoint retention audit failed')
        if any(saved[i+1] is not weight for i,weight in enumerate(task.model.weights)):
            raise AssertionError('CUDA core saved a non-parameter intermediate')
        records.append({'depth':depth,'config':candidate,'passed':True,
                        'epochs':2,'batch_sizes':[8,8,3],
                        'complete_replay_byte_repeatable':True,
                        'predictions_match_independent_eager':True,
                        'saved_core_activation_storages':1,
                        'saved_core_activation_bytes':3*82*4,
                        'saved_parameter_references':2*depth})
        del saved,core,features,task,reference,optimizer
    return {'passed':True,'device':torch.cuda.get_device_name(),
            'checks':checks,'maximum_abs_error':maximum_error,
            'tolerance':{'atol':2e-5,'rtol':3e-3},'cases':records,
            'scope':'Both depths; every gradient, parameter update and momentum state versus independent ordinary eager autograd/SGD, complete captured scores and exact replay-reset checks. Test uses only generated synthetic data.'}
