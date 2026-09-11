"""Independent vectorized NumPy oracle for the explicit BatchNorm equations."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('bn_conv_oracle',Path(__file__).parent/'ordered_backend/cpu_ref.py')
conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)

def channel_sum(x):
    total=np.zeros(x.shape[1],np.float32)
    for n in range(x.shape[0]):
        for y in range(x.shape[2]):
            for xx in range(x.shape[3]):total=total+x[n,:,y,xx]
    return total

def view(x):return x[None,:,None,None]

def batchnorm(x,gamma,beta,mean_running,var_running,training):
    m=x.shape[0]*x.shape[2]*x.shape[3]
    if training:
        mean=channel_sum(x)*np.float32(1/m)
        shifted=x-view(mean)
        variance=channel_sum(shifted*shifted)*np.float32(1/m)
        new_mean=mean_running*np.float32(.9)+mean*np.float32(.1)
        new_var=var_running*np.float32(.9)+(variance*np.float32(m/(m-1)))*np.float32(.1)
    else:
        mean=mean_running;variance=var_running;new_mean=mean_running;new_var=var_running
    value=variance+np.float32(.00001)
    root=np.ones_like(value)
    for _ in range(16):root=(root+value/root)*np.float32(.5)
    inverse=np.float32(1)/root
    xhat=(x-view(mean))*view(inverse)
    out=xhat*view(gamma)+view(beta)
    return out,xhat,inverse,new_mean,new_var

def backward(dy,xhat,gamma,inverse):
    inv=np.float32(1/(dy.shape[0]*dy.shape[2]*dy.shape[3]))
    dg=channel_sum(dy*xhat);db=channel_sum(dy)
    dx=(((dy-view(db*inv))-xhat*view(dg*inv))*view(gamma))*view(inverse)
    return dx,dg,db

def network_forward(x,p,depth,running,training):
    cache={'x':x,'a':[],'z':[],'xhat':[],'inverse':[]};new_running={};a=x
    for i in range(depth):
        raw=conv.forward(a,p[f'conv{i}.weight'])
        z,xhat,inverse,rm,rv=batchnorm(raw,p[f'bn{i}.weight'],p[f'bn{i}.bias'],
            running[f'running_mean{i}'],running[f'running_variance{i}'],training)
        a=np.where(z>0,z,np.float32(0))
        for name,value in [('z',z),('a',a),('xhat',xhat),('inverse',inverse)]:cache[name].append(value)
        new_running.update({f'running_mean{i}':rm,f'running_variance{i}':rv})
    flat=a.reshape(len(a),-1);z=conv.mm(flat,p['head1.weight'].T)+p['head1.bias'];h=np.where(z>0,z,np.float32(0))
    scores=conv.mm(h,p['head2.weight'].T)+p['head2.bias']
    cache.update(flat=flat,head_z=z,head_a=h,scores=scores)
    return cache,new_running

def network_step(x,labels,p,v,config,lr,running):
    depth=config['depth'];cache,running=network_forward(x,p,depth,running,True);g={}
    d2=conv.softmax_delta(cache['scores'],labels)
    g['head2.weight']=conv.mm(d2.T,cache['head_a']);g['head2.bias']=conv.rows(d2)
    dh=conv.mm(d2,p['head2.weight']);dz=np.where(cache['head_z']>0,dh,np.float32(0))
    g['head1.weight']=conv.mm(dz.T,cache['flat']);g['head1.bias']=conv.rows(dz)
    da=conv.mm(dz,p['head1.weight']).reshape(cache['a'][-1].shape)
    for i in reversed(range(depth)):
        dy=np.where(cache['z'][i]>0,da,np.float32(0))
        dx,dg,db=backward(dy,cache['xhat'][i],p[f'bn{i}.weight'],cache['inverse'][i])
        g[f'bn{i}.weight']=dg;g[f'bn{i}.bias']=db
        prior=cache['x'] if i==0 else cache['a'][i-1]
        g[f'conv{i}.weight']=conv.dweight(prior,dx)
        if i:da=conv.dinput(dx,p[f'conv{i}.weight'])
    params={};velocity={}
    for name in p:params[name],velocity[name]=conv.momentum(p[name],g[name],v[name],lr,config.get('momentum',.9),config.get('weight_decay',.0001))
    return params,velocity,running
