"""Generic full ordered CNN step and GPU-resident schedule replay.

Only raw fit images/labels, raw query images and seed-only initialization/maps
enter this learner. Every backpropagation finishes before any weight changes.
"""
import numpy as np
import torch
from ops import Backend

class Network:
    def __init__(self,config,initial):
        self.config=config.copy();self.backend=Backend()
        self.names=list(initial)
        self.initial={name:torch.from_numpy(a.copy()).cuda() for name,a in initial.items()}
        self.params={name:a.clone() for name,a in self.initial.items()}
        self.velocity={name:torch.zeros_like(a) for name,a in self.params.items()}
        self.grad={name:torch.empty_like(a) for name,a in self.params.items()}
    def reset(self):
        for name in self.names:
            self.params[name].copy_(self.initial[name]);self.velocity[name].zero_()
    def workspace(self,batch,training=True):
        c=self.config;side=c.get('image_size',9);width=c['width'];head=c['head_width']
        def empty(shape):return torch.empty(shape,dtype=torch.float32,device='cuda')
        ws={'x':empty((batch,1,side,side)), 'y':torch.empty(batch,dtype=torch.int64,device='cuda'),
            'pad':[], 'z':[], 'a':[], 'dz':[], 'da':[], 'dpad':[]}
        ci=1
        for _ in range(c['depth']):
            ws['pad'].append(empty((batch,ci,side+2,side+2)))
            for name in (('z','a','dz','da') if training else ('z','a')):ws[name].append(empty((batch,width,side,side)))
            if training:ws['dpad'].append(empty((batch,width,side+2,side+2)))
            ci=width
        for name,shape in {'head_z':(batch,head),'head_a':(batch,head),'scores':(batch,10),
                **({'d2':(batch,10),'head_da':(batch,head),'head_dz':(batch,head)} if training else {})}.items():ws[name]=empty(shape)
        return ws
    def forward(self,ws):
        b=self.backend;p=self.params;x=ws['x']
        for i in range(self.config['depth']):
            b.pad(x,ws['pad'][i]);b.forward(ws['pad'][i],p[f'conv{i}.weight'],ws['z'][i])
            b.relu(ws['z'][i],ws['a'][i]);x=ws['a'][i]
        flat=x.reshape(len(x),-1)
        b.mm(flat,p['head1.weight'],ws['head_z'],bt=True)
        b.bias(ws['head_z'],p['head1.bias'],ws['head_z'])
        b.relu(ws['head_z'],ws['head_a'])
        b.mm(ws['head_a'],p['head2.weight'],ws['scores'],bt=True)
        b.bias(ws['scores'],p['head2.bias'],ws['scores'])
    def backward(self,ws):
        b=self.backend;p=self.params;g=self.grad
        inv=float(np.float32(1/len(ws['x'])))
        if self.config.get('loss')=='approx_softmax':b.softmax_delta(ws['scores'],ws['y'],ws['d2'],inv)
        else:b.delta(ws['scores'],ws['y'],ws['d2'],inv)
        b.mm(ws['d2'],ws['head_a'],g['head2.weight'],at=True);b.rows(ws['d2'],g['head2.bias'])
        b.mm(ws['d2'],p['head2.weight'],ws['head_da'])
        b.drelu(ws['head_z'],ws['head_da'],ws['head_dz'])
        flat=ws['a'][-1].reshape(len(ws['x']),-1)
        b.mm(ws['head_dz'],flat,g['head1.weight'],at=True);b.rows(ws['head_dz'],g['head1.bias'])
        b.mm(ws['head_dz'],p['head1.weight'],ws['da'][-1].reshape(len(ws['x']),-1))
        for i in range(self.config['depth']-1,-1,-1):
            b.drelu(ws['z'][i],ws['da'][i],ws['dz'][i])
            b.dweight(ws['pad'][i],ws['dz'][i],g[f'conv{i}.weight'])
            if i>0:
                b.pad(ws['dz'][i],ws['dpad'][i])
                b.dinput(ws['dpad'][i],p[f'conv{i}.weight'],ws['da'][i-1])
    def update(self,lr):
        for name in self.names:self.backend.momentum(self.params[name],self.grad[name],self.velocity[name],
            float(np.float32(lr)),float(np.float32(self.config.get('momentum',.9))),
            float(np.float32(self.config.get('weight_decay',.0001))))
    def step(self,ws,lr):
        self.forward(ws);self.backward(ws);self.update(lr)
    def state(self):return {name:p.cpu().numpy().copy() for name,p in self.params.items()}
    def infer(self,raw,batch_size=128):
        outputs=[]
        for first in range(0,len(raw),batch_size):
            values=raw[first:first+batch_size];ws=self.workspace(len(values),training=False)
            self.backend.normalize(values,ws['x']);self.forward(ws)
            outputs.append(ws['scores'].cpu().numpy())
        return np.concatenate(outputs)

class Trainer:
    """Prepare once, then replay an entire newly initialized learning task.

    prepare() allocates buffers, materializes all seed-only schedules, compiles
    kernels and captures graphs. Those costs are outside GPU-resident timing.
    invoke() resets parameters/velocity/position, trains every epoch, normalizes
    all supplied queries, computes every score and emits every prediction.
    Validation can call initialize(), train_epoch(e), and infer separately.
    """
    def __init__(self,train_images,train_labels,config,seed):
        from schedule import initial_parameters
        assert train_images.dtype==np.float32 and train_images.ndim==4
        assert train_images.shape[1]==1 and train_images.shape[2]==train_images.shape[3]
        assert train_labels.shape==(len(train_images),)
        self.config=config.copy();self.config['image_size']=train_images.shape[-1]
        self.seed=int(seed);self.n=len(train_images);self.epochs=int(config['epochs'])
        self.initial_arrays=initial_parameters(self.config,self.seed)
        self.network=Network(self.config,self.initial_arrays)
        sentinel=np.concatenate((train_images.reshape(-1),np.zeros(1,np.float32)))
        self.raw=torch.from_numpy(sentinel).cuda()
        self.labels=torch.from_numpy(train_labels.astype(np.int64)).cuda()
        self.offset=torch.zeros((),dtype=torch.int32,device='cuda')
        self.graphs={};self.schedule_manifests=[]
    def prepare(self,query_images=None):
        import schedule
        side=self.config['image_size'];size=self.n*self.epochs
        self.indices=torch.empty((size,side*side,4),dtype=torch.int32,device='cuda')
        self.coefficients=torch.empty((size,side*side,4),dtype=torch.float32,device='cuda')
        self.order=torch.empty(size,dtype=torch.int32,device='cuda')
        for e,(order,indices,coefficients,manifest) in enumerate(schedule.iter_epochs(
                self.seed,self.n,self.epochs,side,self.config.get('augmentation','mild_affine'))):
            first=e*self.n;last=first+self.n
            self.indices[first:last].copy_(torch.from_numpy(indices))
            self.coefficients[first:last].copy_(torch.from_numpy(coefficients))
            self.order[first:last].copy_(torch.from_numpy(order))
            self.schedule_manifests.append(manifest)
        batch=int(self.config['batch_size'])
        sizes=sorted(set([min(batch,self.n),self.n%batch])-set([0]))
        self.workspaces={n:self.network.workspace(n) for n in sizes}
        rates=sorted(set(float(schedule.learning_rate(self.config,e)) for e in range(1,self.epochs+1)))
        self.rates={e:float(schedule.learning_rate(self.config,e)) for e in range(1,self.epochs+1)}
        def operation(ws,lr):
            b=self.network.backend
            b.augment(self.raw,self.labels,self.indices,self.coefficients,self.order,self.offset,ws['x'],ws['y'])
            self.network.step(ws,lr);b.advance(self.offset,len(ws['x']))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for n,ws in self.workspaces.items():
            for lr in rates:
                self.offset.zero_();self.network.reset()
                with torch.cuda.stream(stream):operation(ws,lr)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                self.offset.zero_();self.network.reset();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):operation(ws,lr)
                self.graphs[n,lr]=graph
        if query_images is not None:
            assert query_images.dtype==np.float32 and query_images.shape[1:]==(1,side,side)
            self.query=torch.from_numpy(query_images.copy()).cuda()
            query_sizes=sorted(set([min(128,len(query_images)),len(query_images)%128])-set([0]))
            self.query_workspaces={n:self.network.workspace(n,training=False) for n in query_sizes}
            self.output_scores=torch.empty((len(query_images),10),dtype=torch.float32,device='cuda')
            self.predictions=torch.empty(len(query_images),dtype=torch.int64,device='cuda')
            def inference():
                for first in range(0,len(self.query),128):
                    last=min(first+128,len(self.query));ws=self.query_workspaces[last-first]
                    self.network.backend.normalize(self.query[first:last],ws['x'])
                    self.network.forward(ws)
                    self.output_scores[first:last].copy_(ws['scores'])
                    self.network.backend.argmax(ws['scores'],self.predictions[first:last])
            with torch.cuda.stream(stream):inference()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.inference_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.inference_graph,stream=stream):inference()
        self.initialization_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.initialization_graph,stream=stream):
            self.network.reset();self.offset.zero_()
        torch.cuda.synchronize();self.initialize()
    def initialize(self):self.initialization_graph.replay()
    def train_epoch(self,epoch):
        batch=self.config['batch_size'];lr=self.rates[epoch]
        for first in range(0,self.n,batch):self.graphs[min(batch,self.n-first),lr].replay()
    def invoke(self):
        self.initialize()
        for epoch in range(1,self.epochs+1):self.train_epoch(epoch)
        self.inference_graph.replay()
    def outputs(self):
        return {'parameters':self.network.state(),
            'scores':self.output_scores.cpu().numpy().copy(),
            'predictions':self.predictions.cpu().numpy().copy(),
            'velocities':{name:v.cpu().numpy().copy() for name,v in self.network.velocity.items()}}
