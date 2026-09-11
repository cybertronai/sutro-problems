"""Explicit v4 BatchNorm; fixed Newton square root is ordinary div/add/mul.

All n,y,x sums start at +0. Constants: k20 epsilon, k21 invM,
k22 M/(M-1), k23 .1; k2 .9, k4 one, k6 .5. The caller initializes them.
"""
from ir_core import ref as R,ins as I,loop as L
from ir_conv import nested

def inverse_std(variance,invstd,channel,prefix):
    value,r,quotient=[R('s',j) for j in (7,3,4)]
    return [I('add',value,R(variance,**{channel:1}),R('k',20)),I('copy',r,R('k',4)),
        L(prefix+'_newton',16,[I('div',quotient,value,r),I('add',r,r,quotient),I('mul',r,r,R('k',6))]),
        I('div',R(invstd,**{channel:1}),R('k',4),r)]

def forward(raw,gamma,beta,mean,variance,invstd,xhat,output,running_mean,running_variance,n,c,h,w,prefix,training):
    ch,nn,yy,xx=[prefix+'_'+s for s in ('c','n','y','x')]
    indices={nn:c*h*w,ch:h*w,yy:w,xx:1};dims=[(nn,n),(yy,h),(xx,w)]
    acc=R('s',1);term=R('s',0)
    mu=R(mean,**{ch:1});var=R(variance,**{ch:1});rm=R(running_mean,**{ch:1});rv=R(running_variance,**{ch:1})
    body=[]
    if training:
        body += [I('set',acc,0)]+nested(dims,[I('add',acc,acc,R(raw,**indices))])+[I('mul',mu,acc,R('k',21)),I('set',acc,0)]
        body += nested(dims,[I('sub',term,R(raw,**indices),mu),I('mul',term,term,term),I('add',acc,acc,term)])
        body += [I('mul',var,acc,R('k',21))]
        body += [I('mul',term,rm,R('k',2)),I('mul',acc,mu,R('k',23)),I('add',rm,term,acc),
            I('mul',term,rv,R('k',2)),I('mul',acc,var,R('k',22)),I('mul',acc,acc,R('k',23)),I('add',rv,term,acc)]
    else:body += [I('copy',mu,rm),I('copy',var,rv)]
    body += inverse_std(variance,invstd,ch,prefix)
    body += nested(dims,[I('sub',term,R(raw,**indices),mu),I('mul',R(xhat,**indices),term,R(invstd,**{ch:1})),
        I('mul',term,R(xhat,**indices),R(gamma,**{ch:1})),I('add',R(output,**indices),term,R(beta,**{ch:1}))])
    return [L(ch,c,body)]

def backward(dy,xhat,gamma,invstd,dgamma,dbeta,dx,n,c,h,w,prefix):
    ch,nn,yy,xx=[prefix+'_'+s for s in ('c','n','y','x')]
    indices={nn:c*h*w,ch:h*w,yy:w,xx:1};dims=[(nn,n),(yy,h),(xx,w)]
    dg=R(dgamma,**{ch:1});db=R(dbeta,**{ch:1});term=R('s',0);acc=R('s',1)
    mean_dy,mean_dyx=R('s',3),R('s',4)
    body=[I('set',dg,0),I('set',db,0)]
    body += nested(dims,[I('mul',term,R(dy,**indices),R(xhat,**indices)),I('add',dg,dg,term),I('add',db,db,R(dy,**indices))])
    body += [I('mul',mean_dy,db,R('k',21)),I('mul',mean_dyx,dg,R('k',21))]
    body += nested(dims,[I('sub',acc,R(dy,**indices),mean_dy),I('mul',term,R(xhat,**indices),mean_dyx),
        I('sub',acc,acc,term),I('mul',acc,acc,R(gamma,**{ch:1})),I('mul',R(dx,**indices),acc,R(invstd,**{ch:1}))])
    return [L(ch,c,body)]
