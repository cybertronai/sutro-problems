"""Independent ordered-FP32 batch normalization with a fixed Newton square root."""
import numpy as np

F=np.float32


def sums(x):
    n,c,h,w=x.shape
    result=np.zeros(c,np.float32)
    for sample in range(n):
        for y in range(h):
            for z in range(w):
                result=result+x[sample,:,y,z]
    return result


def invstd(variance):
    value=variance+F(1e-5)
    root=np.ones_like(value)
    for _ in range(16):
        quotient=value/root
        root=(root+quotient)*F(.5)
    return np.ones_like(root)/root


def train(x,gamma,beta,running_mean,running_var):
    m=x.shape[0]*x.shape[2]*x.shape[3]
    assert m>1
    mean=sums(x)*F(1/m)
    centered=x-mean[None,:,None,None]
    variance=sums(centered*centered)*F(1/m)
    inverse=invstd(variance)
    normalized=centered*inverse[None,:,None,None]
    out=(normalized*gamma[None,:,None,None])+beta[None,:,None,None]
    new_mean=(running_mean*F(.9))+(mean*F(.1))
    unbiased=variance*F(m/(m-1))
    new_var=(running_var*F(.9))+(unbiased*F(.1))
    return out,normalized,mean,inverse,new_mean,new_var


def backward(dy,normalized,gamma,inverse):
    m=dy.shape[0]*dy.shape[2]*dy.shape[3]
    dbeta=sums(dy)
    dgamma=sums(dy*normalized)
    average=dbeta*F(1/m)
    weighted=dgamma*F(1/m)
    centered=dy-average[None,:,None,None]
    centered=centered-(normalized*weighted[None,:,None,None])
    dx=(centered*gamma[None,:,None,None])*inverse[None,:,None,None]
    return dx,dgamma,dbeta


def infer(x,gamma,beta,running_mean,running_var):
    inverse=invstd(running_var)
    normalized=(x-running_mean[None,:,None,None])*inverse[None,:,None,None]
    out=(normalized*gamma[None,:,None,None])+beta[None,:,None,None]
    return out,normalized,inverse
