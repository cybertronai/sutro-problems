"""Scalar affine-v4 dense layers and the explicit approximate-softmax gradient."""
from ir_core import ref as R, ins as I, loop as L
from ir_conv import nested


def matmul(left,right,destination,m,k,n,prefix,at=False,bt=False,scratch='s'):
    pm,pn,pk=[prefix+'_'+s for s in ('m','n','k')]
    a=R(left,**({pk:m,pm:1} if at else {pm:k,pk:1}))
    b=R(right,**({pn:k,pk:1} if bt else {pk:n,pn:1}))
    product,acc=R(scratch,0),R(scratch,1)
    return nested([(pm,m),(pn,n)],[I('set',acc,0),L(pk,k,[I('mul',product,a,b),I('add',acc,acc,product)]),
        I('copy',R(destination,**{pm:n,pn:1}),acc)])


def bias_add(source,bias,destination,rows,columns,prefix):
    pn,pc=prefix+'_n',prefix+'_c'
    return nested([(pn,rows),(pc,columns)],[I('add',R(destination,**{pn:columns,pc:1}),
        R(source,**{pn:columns,pc:1}),R(bias,**{pc:1}))])


def rowsum(source,destination,rows,columns,prefix,scratch='s'):
    pn,pc=prefix+'_n',prefix+'_c'
    acc=R(scratch,1)
    return [L(pc,columns,[I('set',acc,0),L(pn,rows,[I('add',acc,acc,R(source,**{pn:columns,pc:1}))]),
        I('copy',R(destination,**{pc:1}),acc)])]


def delta(scores,labels,destination,batch,prefix,loss='approx_softmax',scratch='s',constants='k'):
    """k slots:0zero,4one,5four,6half,7invB,8negative16,9inv1024,10..19rawclasses.

    s slots:0product,1accumulator,2condition,3maximum,4total,5onehot.
    The ten repeated squarings are literal mul operations, not exp calls.
    """
    pn,pc,pk=[prefix+'_'+s for s in ('n','c','square')]
    acc,condition,largest,total,target=[R(scratch,i) for i in (1,2,3,4,5)]
    zero,one,inv_batch,negative16,inv1024=[R(constants,i) for i in (0,4,7,8,9)]
    body=[]
    if loss=='approx_softmax':
        body.append(I('copy',largest,R(scores,**{pn:10})))
        for c in range(1,10):
            value=R(scores,c,**{pn:10})
            body += [I('cmp',condition,largest,value),I('select',largest,condition,value,largest)]
        body.append(I('set',total,0))
        value=R(scores,**{pn:10,pc:1});out=R(destination,**{pn:10,pc:1})
        body += [L(pc,10,[I('sub',acc,value,largest),
            I('cmp',condition,acc,negative16),I('select',acc,condition,negative16,acc),
            I('cmp',condition,zero,acc),I('select',acc,condition,zero,acc),
            I('mul',acc,acc,inv1024),I('add',acc,one,acc),
            L(pk,10,[I('mul',acc,acc,acc)]),I('add',total,total,acc),I('copy',out,acc)])]
    elif loss!='mse':raise ValueError(loss)
    value=R(destination if loss=='approx_softmax' else scores,**{pn:10,pc:1})
    body += [L(pc,10,([I('div',acc,value,total)] if loss=='approx_softmax' else [I('copy',acc,value)]) + [
        I('cmp',condition,R(labels,**{pn:1}),R(constants,10,**{pc:1}),predicate='eq'),
        I('select',target,condition,one,zero),I('sub',acc,acc,target),
        I('mul',R(destination,**{pn:10,pc:1}),acc,inv_batch)])]
    return [L(pn,batch,body)]
