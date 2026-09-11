"""NumPy FP32 oracle, independently structured as array-wise ordered loops."""
import numpy as np

def pad(x):return np.pad(x,((0,0),(0,0),(1,1),(1,1)))

def forward(x,w):
    n,ci,h,ww=x.shape;co=w.shape[0];p=pad(x)
    out=np.zeros((n,co,h,ww),np.float32)
    for c in range(ci):
        for ky in range(3):
            for kx in range(3):
                term=p[:,c,None,ky:ky+h,kx:kx+ww]*w[None,:,c,ky,kx,None,None]
                out=out+term
    return out

def dinput(dy,w):
    n,co,h,ww=dy.shape;ci=w.shape[1];p=pad(dy)
    out=np.zeros((n,ci,h,ww),np.float32)
    for c in range(co):
        for ky in range(3):
            for kx in range(3):
                term=p[:,c,None,ky:ky+h,kx:kx+ww]*w[c,None,:,2-ky,2-kx,None,None]
                out=out+term
    return out

def dweight(x,dy):
    n,ci,h,ww=x.shape;co=dy.shape[1];p=pad(x)
    out=np.zeros((co,ci,3,3),np.float32)
    for nn in range(n):
        for y in range(h):
            for xx in range(ww):
                term=dy[nn,:,y,xx,None,None,None]*p[nn,None,:,y:y+3,xx:xx+3]
                out=out+term
    return out

def mm(a,b):
    out=np.zeros((a.shape[0],b.shape[1]),np.float32)
    for k in range(a.shape[1]):out=out+a[:,k,None]*b[None,k,:]
    return out

def rows(a):
    out=np.zeros(a.shape[1],np.float32)
    for row in a:out=out+row
    return out

def momentum(w,g,v,lr,mu=.9,decay=.0001):
    d=g+np.float32(decay)*w
    nv=np.float32(mu)*v+d
    nw=w-np.float32(lr)*nv
    return nw,nv

def softmax_delta(scores,labels):
    largest=scores[:,0].copy()
    for c in range(1,10):largest=np.where(scores[:,c]>largest,scores[:,c],largest)
    shifted=scores-largest[:,None]
    clamped=np.minimum(np.maximum(shifted,np.float32(-16)),np.float32(0))
    term=np.float32(1)+clamped*np.float32(1/1024)
    for _ in range(10):term=term*term
    total=np.zeros(len(scores),np.float32)
    for c in range(10):total=total+term[:,c]
    probability=term/total[:,None]
    return (probability-(labels[:,None]==np.arange(10)).astype(np.float32))*np.float32(1/len(scores))

def network_forward(x,p,depth):
    cache={'x':x,'z':[],'a':[]}
    a=x
    for i in range(depth):
        z=forward(a,p[f'conv{i}.weight']);a=np.where(z>0,z,np.float32(0))
        cache['z'].append(z);cache['a'].append(a)
    flat=a.reshape(len(a),-1)
    z=mm(flat,p['head1.weight'].T)+p['head1.bias'];h=np.where(z>0,z,np.float32(0))
    scores=mm(h,p['head2.weight'].T)+p['head2.bias']
    cache.update({'flat':flat,'head_z':z,'head_a':h,'scores':scores})
    return cache

def network_step(x,labels,p,v,config,lr):
    depth=config['depth'];cache=network_forward(x,p,depth);g={}
    if config.get('loss')=='approx_softmax':d2=softmax_delta(cache['scores'],labels)
    else:d2=(cache['scores']-(labels[:,None]==np.arange(10)).astype(np.float32))*np.float32(1/len(x))
    g['head2.weight']=mm(d2.T,cache['head_a']);g['head2.bias']=rows(d2)
    dh=mm(d2,p['head2.weight']);dz=np.where(cache['head_z']>0,dh,np.float32(0))
    g['head1.weight']=mm(dz.T,cache['flat']);g['head1.bias']=rows(dz)
    da=mm(dz,p['head1.weight']).reshape(cache['a'][-1].shape)
    for i in range(depth-1,-1,-1):
        dz=np.where(cache['z'][i]>0,da,np.float32(0))
        prior=cache['x'] if i==0 else cache['a'][i-1]
        g[f'conv{i}.weight']=dweight(prior,dz)
        if i>0:da=dinput(dz,p[f'conv{i}.weight'])
    newp={};newv={}
    for name in p:newp[name],newv[name]=momentum(p[name],g[name],v[name],lr,config.get('momentum',.9),config.get('weight_decay',.0001))
    return newp,newv,g,cache['scores']
