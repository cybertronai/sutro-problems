"""Ordered-FP32 reconstruction backward with a depth-independent workspace.

The selected model's parameters, initialization, forward arithmetic, augmentation
schedule, head, and optimizer come from the frozen accuracy experiment. Only the
core backward implementation changes: it reconstructs prior states and reuses a
single branch workspace. Reconstruction is approximate in FP32, so gradients
must be validated against the stored implementation and accuracy measured anew.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'reversible-refinement'
PREVIOUS = HERE.parent / 'procedure-transfer'
ARCHITECTURES = tuple(f'rev32_b{n}_lift_relu' for n in (4,8,16,32))


def _module(name, path):
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve()!=Path(path).resolve():
            raise RuntimeError(f'Conflicting module: {name}')
        return existing
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def source_models():
    return _module('ordered_memory_frozen_models',SOURCE/'models.py')


def extended_topology_class():
    """The identical selected topology, extended to16/32 solely for profiling."""
    base=source_models().Topology
    class ExtendedTopology(base):
        def __init__(self,config):
            if config['architecture'] not in ARCHITECTURES:raise ValueError(config['architecture'])
            self.declared_architecture=config['architecture']
            # The frozen constructor initializes bookkeeping, then dispatches
            # to this explicit build method. It is never modified on disk.
            super().__init__({**config,'architecture':'rev32_b8_lift_relu'})
            self.architecture=self.declared_architecture;self.config=config.copy()

        def _build(self):
            count=int(self.declared_architecture.split('_')[1][1:])
            learned=self.conv('x','lift',31)
            learned=self.relu(learned,'lift_relu')
            value=self.node('concat','core_in',['x',learned],(32,self.side,self.side))
            left=self.node('slice','split0',[value],(16,self.side,self.side),start=0)
            right=self.node('slice','split1',[value],(16,self.side,self.side),start=16)
            for i in range(count):
                f=f'block{i}.f';g=f'block{i}.g'
                left=self.add(left,self.branch(right,f,16),f'block{i}.left',self.alpha)
                right=self.add(right,self.branch(left,g,16),f'block{i}.right',self.alpha)
                self.blocks.append(dict(f=f,g=g,deep=False))
            value=self.node('concat','core_out',[left,right],(32,self.side,self.side))
            value=self.linear(value,'head1',128)
            value=self.relu(value,'head_relu')
            self.linear(value,'head2',10)
    return ExtendedTopology


def stored_control_models():
    """Separate in-memory import of stored DAG code with the extended topology.

    For4/8 this must independently match the untouched frozen source. For16/32
    it supplies a matched stored-activation control using that same DAG engine.
    """
    module=_module('ordered_memory_extended_stored_models',SOURCE/'models.py')
    if not getattr(module,'_ordered_memory_topology_extended',False):
        module.Topology=extended_topology_class()
        module._ordered_memory_topology_extended=True
    return module


def make_stored_network(config,initial):
    setup_backend()
    extra_path=SOURCE/'extra_ops.py'
    if 'extra_ops' in sys.modules and Path(sys.modules['extra_ops'].__file__).resolve()!=extra_path:
        raise RuntimeError('Conflicting stored-control extra_ops module')
    _module('extra_ops',extra_path)
    return stored_control_models().ArchitectureNetwork(config,initial)


def setup_backend():
    """Make the immutable original backend's unqualified imports unambiguous."""
    backend=PREVIOUS/'reference/ordered_backend'
    for name in ('network','ops','schedule','bn_ops','bn_ref','cpu_ref'):
        loaded=sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve()!=backend/f'{name}.py':
            raise RuntimeError(f'Conflicting original backend module: {name}')
    if str(backend) not in sys.path:sys.path.insert(0,str(backend))
    return importlib.import_module('network').Trainer


def initial_parameters(config,seed):
    if config['architecture'] not in ARCHITECTURES:raise ValueError(config['architecture'])
    return stored_control_models().initial_parameters(config,seed)


def architecture_info(config):
    info=stored_control_models().architecture_info(config)
    info.update(backpropagation='Ordered FP32 reconstruction and branch recomputation',
                core_activation_workspace_independent_of_depth=True,
                activation_memory_savings_claimed='Must be checked by allocation measurements',
                total_training_memory_constant_in_depth=False)
    return info


def workspace_storage_bytes(workspace):
    """Count unique tensor storage reachable from a workspace, excluding weights."""
    import torch
    storages={}
    def visit(value):
        if torch.is_tensor(value):
            storage=value.untyped_storage()
            key=(str(value.device),storage.data_ptr())
            storages[key]=storage.nbytes()
        elif isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
    visit(workspace)
    return {'unique_tensor_storages':len(storages),'allocated_storage_bytes':sum(storages.values())}


class OrderedReconstructionNetwork:
    def __init__(self,config,initial):
        import torch
        setup_backend()
        from ops import Backend
        extra=_module('ordered_memory_frozen_extra_ops',SOURCE/'extra_ops.py')
        if config['architecture'] not in ARCHITECTURES:raise ValueError(config['architecture'])
        self.config=config.copy()
        self.topology=extended_topology_class()(config)
        self.parameter_specs=self.topology.parameter_specs
        if {n:a.shape for n,a in initial.items()}!=dict(self.parameter_specs):
            raise ValueError('Parameter schema differs from the selected model')
        if self.topology.side!=9 or config.get('dropout',0.)!=0.:
            raise ValueError('Only the selected 9x9, no-dropout procedure is supported')
        self.blocks=len(self.topology.blocks)
        self.alpha=self.topology.alpha
        self.dropout=0.;self.buffers={}
        self.backend=Backend();self.extra=extra.ExtraOps(self.backend)
        self.names=list(initial)
        self.initial={n:torch.from_numpy(a.copy()).cuda() for n,a in initial.items()}
        self.params={n:a.clone() for n,a in self.initial.items()}
        self.grad={n:torch.empty_like(a) for n,a in self.params.items()}
        self.velocity={n:torch.zeros_like(a) for n,a in self.params.items()}

    def reset(self):
        for name in self.names:
            self.params[name].copy_(self.initial[name]);self.velocity[name].zero_()

    def workspace(self,batch,training=True):
        import torch
        def empty(*tail):return torch.empty((batch,*tail),dtype=torch.float32,device='cuda')
        values={'x':empty(1,9,9),'lift':empty(31,9,9),'lift_relu':empty(31,9,9),
                'core_in':empty(32,9,9),'core_out':empty(32,9,9),
                'head1':empty(128),'head_relu':empty(128),'head2':empty(10)}
        ws={'v':values,'x':values['x'],'scores':values['head2'],
            'y':torch.empty(batch,dtype=torch.int64,device='cuda'),
            'lift_pad':empty(1,11,11),
            'left':empty(16,9,9),'right':empty(16,9,9),
            'branch_pre':empty(16,9,9),'branch_pad':empty(16,11,11),
            'branch_raw':empty(16,9,9)}
        if training:
            d={'x':empty(1,9,9),'head2':empty(10),'core_out':empty(32,9,9)}
            ws.update(d=d,dx=d['x'],head_da=empty(128),head_dz=empty(128),
                      dleft=empty(16,9,9),dright=empty(16,9,9),
                      branch_dz=empty(16,9,9),branch_dpad=empty(16,11,11),
                      branch_da=empty(16,9,9),branch_dx=empty(16,9,9),
                      core_dx=empty(32,9,9),lift_da=empty(31,9,9),
                      lift_dz=empty(31,9,9),lift_dpad=empty(31,11,11),
                      lift_dx=empty(1,9,9))
        return ws

    def _branch_forward(self,value,weight,ws):
        b=self.backend
        b.relu(value,ws['branch_pre'])
        b.pad(ws['branch_pre'],ws['branch_pad'])
        b.forward(ws['branch_pad'],weight,ws['branch_raw'])

    def _branch_backward(self,value,dy,name,ws):
        """Differentiate alpha*branch; its recomputed forward cache is in ws."""
        b=self.backend;e=self.extra
        e.axpby(dy,dy,ws['branch_dz'],self.alpha,0.)
        b.dweight(ws['branch_pad'],ws['branch_dz'],self.grad[name])
        b.pad(ws['branch_dz'],ws['branch_dpad'])
        b.dinput(ws['branch_dpad'],self.params[name],ws['branch_da'])
        b.drelu(value,ws['branch_da'],ws['branch_dx'])

    def forward(self,ws,training=False):
        b=self.backend;e=self.extra;p=self.params;v=ws['v']
        b.pad(v['x'],ws['lift_pad'])
        b.forward(ws['lift_pad'],p['lift.weight'],v['lift'])
        b.relu(v['lift'],v['lift_relu'])
        e.concat(v['x'],v['lift_relu'],v['core_in'])
        e.channel_slice(v['core_in'],ws['left'],0)
        e.channel_slice(v['core_in'],ws['right'],16)
        for index in range(self.blocks):
            self._branch_forward(ws['right'],p[f'block{index}.f.conv0.weight'],ws)
            e.axpby(ws['left'],ws['branch_raw'],ws['left'],1.,self.alpha)
            self._branch_forward(ws['left'],p[f'block{index}.g.conv0.weight'],ws)
            e.axpby(ws['right'],ws['branch_raw'],ws['right'],1.,self.alpha)
        e.concat(ws['left'],ws['right'],v['core_out'])
        b.mm(v['core_out'].reshape(len(v['x']),-1),p['head1.weight'],v['head1'],bt=True)
        b.bias(v['head1'],p['head1.bias'],v['head1'])
        b.relu(v['head1'],v['head_relu'])
        b.mm(v['head_relu'],p['head2.weight'],v['head2'],bt=True)
        b.bias(v['head2'],p['head2.bias'],v['head2'])
        # No per-block state is retained. Branch scratch is invalid after forward
        # and is always recomputed before the corresponding backward operation.

    def backward(self,ws):
        b=self.backend;e=self.extra;p=self.params;g=self.grad;v=ws['v'];d=ws['d']
        n=len(v['x']);inv=float(np.float32(1/n))
        b.softmax_delta(v['head2'],ws['y'],d['head2'],inv)
        b.mm(d['head2'],v['head_relu'],g['head2.weight'],at=True)
        b.rows(d['head2'],g['head2.bias'])
        b.mm(d['head2'],p['head2.weight'],ws['head_da'])
        b.drelu(v['head1'],ws['head_da'],ws['head_dz'])
        b.mm(ws['head_dz'],v['core_out'].reshape(n,-1),g['head1.weight'],at=True)
        b.rows(ws['head_dz'],g['head1.bias'])
        b.mm(ws['head_dz'],p['head1.weight'],d['core_out'].reshape(n,-1))

        e.channel_slice(v['core_out'],ws['left'],0)
        e.channel_slice(v['core_out'],ws['right'],16)
        e.channel_slice(d['core_out'],ws['dleft'],0)
        e.channel_slice(d['core_out'],ws['dright'],16)
        for index in range(self.blocks-1,-1,-1):
            name_g=f'block{index}.g.conv0.weight'
            self._branch_forward(ws['left'],p[name_g],ws)
            self._branch_backward(ws['left'],ws['dright'],name_g,ws)
            e.axpby(ws['dleft'],ws['branch_dx'],ws['dleft'])
            e.axpby(ws['right'],ws['branch_raw'],ws['right'],1.,-self.alpha)

            name_f=f'block{index}.f.conv0.weight'
            self._branch_forward(ws['right'],p[name_f],ws)
            self._branch_backward(ws['right'],ws['dleft'],name_f,ws)
            e.axpby(ws['dright'],ws['branch_dx'],ws['dright'])
            e.axpby(ws['left'],ws['branch_raw'],ws['left'],1.,-self.alpha)

        e.concat(ws['dleft'],ws['dright'],ws['core_dx'])
        e.channel_slice(ws['core_dx'],d['x'],0)
        e.channel_slice(ws['core_dx'],ws['lift_da'],1)
        b.drelu(v['lift'],ws['lift_da'],ws['lift_dz'])
        b.dweight(ws['lift_pad'],ws['lift_dz'],g['lift.weight'])
        b.pad(ws['lift_dz'],ws['lift_dpad'])
        b.dinput(ws['lift_dpad'],p['lift.weight'],ws['lift_dx'])
        e.axpby(d['x'],ws['lift_dx'],d['x'])

    def update(self,lr):
        for name in self.names:
            self.backend.momentum(self.params[name],self.grad[name],self.velocity[name],
                float(np.float32(lr)),float(np.float32(self.config.get('momentum',.9))),
                float(np.float32(self.config.get('weight_decay',.0001))))

    def step(self,ws,lr):
        self.forward(ws,training=True);self.backward(ws);self.update(lr)

    def state(self,include_buffers=True):
        return {name:value.cpu().numpy().copy() for name,value in self.params.items()}

    def inverse_core(self,output):
        """A read-only inverse using the same reusable branch workspace."""
        import torch
        n=len(output)
        def empty(c,side=9):return torch.empty((n,c,side,side),dtype=output.dtype,device=output.device)
        ws={'left':empty(16),'right':empty(16),'branch_pre':empty(16),
            'branch_pad':empty(16,11),'branch_raw':empty(16)}
        e=self.extra;p=self.params
        e.channel_slice(output,ws['left'],0);e.channel_slice(output,ws['right'],16)
        for index in range(self.blocks-1,-1,-1):
            self._branch_forward(ws['left'],p[f'block{index}.g.conv0.weight'],ws)
            e.axpby(ws['right'],ws['branch_raw'],ws['right'],1.,-self.alpha)
            self._branch_forward(ws['right'],p[f'block{index}.f.conv0.weight'],ws)
            e.axpby(ws['left'],ws['branch_raw'],ws['left'],1.,-self.alpha)
        result=torch.empty_like(output);e.concat(ws['left'],ws['right'],result)
        return result

    def reconstruct_input(self,core_output):
        import torch
        lifted=self.inverse_core(core_output)
        result=torch.empty((len(lifted),1,9,9),dtype=lifted.dtype,device=lifted.device)
        self.extra.channel_slice(lifted,result,0)
        return result


class OrderedReconstructionTrainer:
    """Reuse the original scheduling, CUDA graph capture, inference and replay."""
    def __init__(self,train_images,train_labels,config,seed):
        import torch
        setup_backend()
        if train_images.dtype!=np.float32 or train_images.shape[1:]!=(1,9,9):
            raise ValueError('Expected float32 N,1,9,9 images')
        if train_labels.shape!=(len(train_images),):raise ValueError('Invalid label shape')
        self.config=config.copy();self.config['image_size']=9
        self.seed=int(seed);self.n=len(train_images);self.epochs=int(config['epochs'])
        self.initial_arrays=initial_parameters(self.config,self.seed)
        self.network=OrderedReconstructionNetwork(self.config,self.initial_arrays)
        sentinel=np.concatenate((train_images.reshape(-1),np.zeros(1,np.float32)))
        self.raw=torch.from_numpy(sentinel).cuda()
        self.labels=torch.from_numpy(train_labels.astype(np.int64)).cuda()
        self.offset=torch.zeros((),dtype=torch.int32,device='cuda')
        self.graphs={};self.schedule_manifests=[]

    def prepare(self,query_images=None):return setup_backend().prepare(self,query_images)
    def initialize(self):return setup_backend().initialize(self)
    def train_epoch(self,epoch):return setup_backend().train_epoch(self,epoch)
    def invoke(self):return setup_backend().invoke(self)
    def outputs(self):return setup_backend().outputs(self)
