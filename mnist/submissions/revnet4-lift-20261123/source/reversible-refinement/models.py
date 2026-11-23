"""Lossless learned-lift refinement on the frozen MNIST training schedule.

The four learned-lift choices were specified before examining initial-sweep
accuracy results. This is a separate source snapshot; initial-sweep files are
not modified. Each lift concatenates the original normalized input with 31
learned channels, preserving the input regardless of convolution parameters.

The reversible cores use additive coupling. This implementation deliberately
stores activations for ordinary backpropagation; reversibility alone is not a
claim of memory savings. The learned input lift is injective onto its image.
The dense classification head is not invertible.
"""
from __future__ import annotations

import importlib
import math
from collections import OrderedDict

import numpy as np


ARCHITECTURES = (
    'cnn32_d3', 'rev32_b4_lift_relu', 'rev32_b8_lift_relu',
    'rev32_b4_lift_linear', 'rev32_b8_lift_linear',
)


class Topology:
    def __init__(self, config):
        self.config = config.copy()
        self.architecture = config['architecture']
        if self.architecture not in ARCHITECTURES:
            raise ValueError(self.architecture)
        self.side = int(config.get('image_size',9))
        self.alpha = float(config.get('residual_scale', .25))
        self.ops = []
        self.shapes = OrderedDict(x=(1,self.side,self.side))
        self.parameter_specs = OrderedDict()
        self.layer_specs = []
        self.blocks = []
        self.reversible = self.architecture.startswith('rev')
        self.injective_stem = self.reversible
        self._build()

    def node(self, op, name, inputs, shape, **kw):
        assert name not in self.shapes
        self.shapes[name] = tuple(shape)
        self.ops.append(dict(op=op,name=name,inputs=list(inputs),**kw))
        return name

    def conv(self, src, name, channels):
        ci,h,w=self.shapes[src]
        self.parameter_specs[name+'.weight']=(channels,ci,3,3)
        self.layer_specs.append(dict(kind='conv',name=name,ci=ci,co=channels))
        return self.node('conv',name,[src],(channels,h,w),parameter=name+'.weight')

    def relu(self, src, name):
        return self.node('relu',name,[src],self.shapes[src])

    def linear(self, src, name, width):
        ci=math.prod(self.shapes[src])
        self.parameter_specs[name+'.weight']=(width,ci)
        self.parameter_specs[name+'.bias']=(width,)
        self.layer_specs.append(dict(kind='linear',name=name,ci=ci,co=width))
        return self.node('linear',name,[src],(width,),parameter=name)

    def add(self, left, right, name, scale):
        assert self.shapes[left]==self.shapes[right]
        return self.node('add',name,[left,right],self.shapes[left],scale=scale)

    def branch(self, src, prefix, channels, deep=False):
        value=self.relu(src,prefix+'.relu0')
        value=self.conv(value,prefix+'.conv0',channels)
        if deep:
            value=self.relu(value,prefix+'.relu1')
            value=self.conv(value,prefix+'.conv1',channels)
        return value

    def _build(self):
        a=self.architecture
        if a=='cnn32_d3':
            value='x'; depth=3
            for i in range(depth):
                value=self.conv(value,f'conv{i}',32)
                value=self.relu(value,'core_out' if i==depth-1 else f'relu{i}')
        else:
            learned=self.conv('x','lift',31)
            if a.endswith('_relu'):
                learned=self.relu(learned,'lift_relu')
            value=self.node('concat','core_in',['x',learned],(32,self.side,self.side))
            count=next(n for n in (8,4) if f'_b{n}' in a)
            left=self.node('slice','split0',[value],(16,self.side,self.side),start=0)
            right=self.node('slice','split1',[value],(16,self.side,self.side),start=16)
            for i in range(count):
                f=f'block{i}.f'; g=f'block{i}.g'
                fv=self.branch(right,f,16)
                left=self.add(left,fv,f'block{i}.left',self.alpha)
                gv=self.branch(left,g,16)
                right=self.add(right,gv,f'block{i}.right',self.alpha)
                self.blocks.append(dict(f=f,g=g,deep=False))
            value=self.node('concat','core_out',[left,right],(32,self.side,self.side))
        value=self.linear(value,'head1',128)
        value=self.relu(value,'head_relu')
        self.linear(value,'head2',10)


def architecture_info(config):
    t=Topology(config)
    return {
        'architecture':t.architecture,
        'parameter_count':sum(math.prod(s) for s in t.parameter_specs.values()),
        'parameter_shapes':dict(t.parameter_specs),
        'reversible_core':t.reversible,
        'injective_input_stem':t.injective_stem,
        'input_lift':('concat(x, ReLU(Conv1to31(x)))' if t.architecture.endswith('_relu')
                      else 'concat(x, Conv1to31(x))' if t.reversible else None),
        'residual_scale':t.alpha if t.architecture.startswith(('rev','res')) else None,
        'reversible_block_count':len(t.blocks),
        'head_invertible':False,
        'backpropagation':'ordinary manual backpropagation with stored activations',
        'activation_memory_savings_claimed':False,
    }


def initial_parameters(config, seed):
    """CPU constructor draws followed by He resets in layer traversal order."""
    import torch
    from torch import nn
    topology=Topology(config)
    torch.manual_seed(int(seed))
    layers=[]
    for row in topology.layer_specs:
        if row['kind']=='conv':
            layer=nn.Conv2d(row['ci'],row['co'],3,padding=1,bias=False)
        else:
            layer=nn.Linear(row['ci'],row['co'])
        layers.append((row['name'],layer))
    result=OrderedDict()
    for name,layer in layers:
        nn.init.kaiming_uniform_(layer.weight,a=0,mode='fan_in',nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        result[name+'.weight']=layer.weight.detach().numpy().copy()
        if layer.bias is not None:
            result[name+'.bias']=layer.bias.detach().numpy().copy()
    assert {n:v.shape for n,v in result.items()}==dict(topology.parameter_specs)
    return result


class ArchitectureNetwork:
    def __init__(self, config, initial):
        import torch
        from ops import Backend
        from extra_ops import ExtraOps
        self.config=config.copy()
        self.topology=Topology(config)
        self.ops=self.topology.ops
        self.shapes=self.topology.shapes
        self.parameter_specs=self.topology.parameter_specs
        assert {n:v.shape for n,v in initial.items()}==dict(self.parameter_specs)
        assert config.get('dropout',0.)==0.
        self.dropout=0.; self.buffers={}
        self.backend=Backend(); self.extra=ExtraOps(self.backend)
        self.names=list(initial)
        self.initial={n:torch.from_numpy(a.copy()).cuda() for n,a in initial.items()}
        self.params={n:a.clone() for n,a in self.initial.items()}
        self.velocity={n:torch.zeros_like(a) for n,a in self.params.items()}
        self.grad={n:torch.empty_like(a) for n,a in self.params.items()}

    def reset(self):
        for n in self.names:
            self.params[n].copy_(self.initial[n]);self.velocity[n].zero_()

    def workspace(self,batch,training=True):
        import torch
        def empty(shape):return torch.empty(shape,dtype=torch.float32,device='cuda')
        values={n:empty((batch,*s)) for n,s in self.shapes.items()}
        ws={'v':values,'x':values['x'],'scores':values['head2'],
            'y':torch.empty(batch,dtype=torch.int64,device='cuda'),'cache':{}}
        if training:
            ws['d']={n:empty((batch,*s)) for n,s in self.shapes.items()}
        for op in self.ops:
            name=op['name'];src=op['inputs'][0];cache={}
            if op['op']=='conv':
                ci,h,w=self.shapes[src];co=self.shapes[name][0]
                cache['pad']=empty((batch,ci,h+2,w+2))
                if training:cache['dpad']=empty((batch,co,h+2,w+2))
            if training:
                cache['dx']=[empty((batch,*self.shapes[s])) for s in op['inputs']]
            ws['cache'][name]=cache
        return ws

    def forward(self,ws,training=False):
        b=self.backend;e=self.extra;p=self.params;v=ws['v']
        for op in self.ops:
            name=op['name'];kind=op['op'];inputs=op['inputs']
            src=v[inputs[0]];out=v[name];cache=ws['cache'][name]
            if kind=='conv':
                b.pad(src,cache['pad']);b.forward(cache['pad'],p[op['parameter']],out)
            elif kind=='relu':b.relu(src,out)
            elif kind=='linear':
                b.mm(src.reshape(len(src),-1),p[name+'.weight'],out,bt=True)
                b.bias(out,p[name+'.bias'],out)
            elif kind=='add':e.axpby(src,v[inputs[1]],out,1.,op['scale'])
            elif kind=='lift':e.lift(src,out)
            elif kind=='slice':e.channel_slice(src,out,op['start'])
            elif kind=='concat':e.concat(src,v[inputs[1]],out)
            else:raise AssertionError(kind)

    def backward(self,ws):
        b=self.backend;e=self.extra;p=self.params;g=self.grad;v=ws['v'];d=ws['d']
        inv=float(np.float32(1/len(ws['x'])))
        b.softmax_delta(ws['scores'],ws['y'],d['head2'],inv)
        assigned={'head2'}
        for op in reversed(self.ops):
            name=op['name'];kind=op['op'];inputs=op['inputs'];cache=ws['cache'][name]
            src=v[inputs[0]];dy=d[name];dx=cache['dx']
            if kind=='conv':
                b.dweight(cache['pad'],dy,g[op['parameter']])
                b.pad(dy,cache['dpad']);b.dinput(cache['dpad'],p[op['parameter']],dx[0])
            elif kind=='relu':b.drelu(src,dy,dx[0])
            elif kind=='linear':
                b.mm(dy,src.reshape(len(src),-1),g[name+'.weight'],at=True)
                b.rows(dy,g[name+'.bias'])
                b.mm(dy,p[name+'.weight'],dx[0].reshape(len(src),-1))
            elif kind=='add':
                dx[0].copy_(dy)
                e.axpby(dy,dy,dx[1],op['scale'],0.)
            elif kind=='lift':e.channel_slice(dy,dx[0],0)
            elif kind=='slice':e.scatter(dy,dx[0],op['start'])
            elif kind=='concat':
                e.channel_slice(dy,dx[0],0)
                e.channel_slice(dy,dx[1],self.shapes[inputs[0]][0])
            else:raise AssertionError(kind)
            for inp,derivative in zip(inputs,dx):
                if inp in assigned:e.axpby(d[inp],derivative,d[inp])
                else:d[inp].copy_(derivative);assigned.add(inp)

    def update(self,lr):
        for name in self.names:
            self.backend.momentum(self.params[name],self.grad[name],self.velocity[name],
                float(np.float32(lr)),float(np.float32(self.config.get('momentum',.9))),
                float(np.float32(self.config.get('weight_decay',.0001))))

    def step(self,ws,lr):
        self.forward(ws,training=True);self.backward(ws);self.update(lr)

    def state(self,include_buffers=True):
        return {n:p.cpu().numpy().copy() for n,p in self.params.items()}

    def _inverse_branch(self,value,prefix,deep):
        import torch
        b=self.backend
        for j in range(2 if deep else 1):
            z=torch.empty_like(value);b.relu(value,z)
            n,c,h,w=z.shape
            pad=torch.empty((n,c,h+2,w+2),dtype=z.dtype,device=z.device)
            out=torch.empty_like(z)
            b.pad(z,pad);b.forward(pad,self.params[f'{prefix}.conv{j}.weight'],out)
            value=out
        return value

    def inverse_core(self,output):
        """Invert coupling core at current weights; returns lifted/stem features.

        FP32 subtraction is approximate, so reconstruction error is measured,
        never asserted to be bitwise zero. The noninvertible head is excluded.
        """
        import torch
        if not self.topology.reversible:raise ValueError('Architecture has no reversible core')
        n,c,h,w=output.shape
        left=torch.empty((n,c//2,h,w),dtype=output.dtype,device=output.device)
        right=torch.empty_like(left)
        self.extra.channel_slice(output,left,0);self.extra.channel_slice(output,right,c//2)
        for row in reversed(self.topology.blocks):
            gv=self._inverse_branch(left,row['g'],row['deep'])
            prior_right=torch.empty_like(right)
            self.extra.axpby(right,gv,prior_right,1.,-self.topology.alpha)
            fv=self._inverse_branch(prior_right,row['f'],row['deep'])
            prior_left=torch.empty_like(left)
            self.extra.axpby(left,fv,prior_left,1.,-self.topology.alpha)
            left,right=prior_left,prior_right
        result=torch.empty_like(output);self.extra.concat(left,right,result)
        return result

    def reconstruct_input(self,core_output):
        """Recover normalized 9x9 input by projecting the inverse onto channel0."""
        import torch
        if not self.topology.injective_stem:raise ValueError('Input stem is not injective')
        lifted=self.inverse_core(core_output)
        result=torch.empty((len(lifted),1,*lifted.shape[2:]),dtype=lifted.dtype,device=lifted.device)
        self.extra.channel_slice(lifted,result,0)
        return result


def _original_trainer():return importlib.import_module('network').Trainer


class ArchitectureTrainer:
    """Reuse frozen schedule/capture/replay methods with a different network."""
    def __init__(self,train_images,train_labels,config,seed):
        import torch
        assert train_images.dtype==np.float32 and train_images.ndim==4
        assert train_images.shape[1:]==(1,9,9)
        assert train_labels.shape==(len(train_images),)
        self.config=config.copy();self.config['image_size']=train_images.shape[-1]
        self.seed=int(seed);self.n=len(train_images);self.epochs=int(config['epochs'])
        self.initial_arrays=initial_parameters(self.config,self.seed)
        self.network=ArchitectureNetwork(self.config,self.initial_arrays)
        sentinel=np.concatenate((train_images.reshape(-1),np.zeros(1,np.float32)))
        self.raw=torch.from_numpy(sentinel).cuda()
        self.labels=torch.from_numpy(train_labels.astype(np.int64)).cuda()
        self.offset=torch.zeros((),dtype=torch.int32,device='cuda')
        self.graphs={};self.schedule_manifests=[]

    def prepare(self,query_images=None):return _original_trainer().prepare(self,query_images)
    def initialize(self):return _original_trainer().initialize(self)
    def train_epoch(self,epoch):return _original_trainer().train_epoch(self,epoch)
    def invoke(self):return _original_trainer().invoke(self)
    def outputs(self):return _original_trainer().outputs(self)
