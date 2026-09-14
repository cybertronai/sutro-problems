"""Actual affine-v4 lowering for NCHW 3x3 convolution and its gradients.

Every returned leaf is an existing v4 operation. Tensor names identify fixed
scratch regions supplied by a caller. Pads, accumulators, gradients and updates
are explicit; these functions return instructions, never cost certificates.
"""
from ir_core import ref as R, ins as I, loop as L


def nested(dimensions, body):
    for variable, count in reversed(dimensions):
        body = [L(variable, count, body)]
    return body


def pad_nchw(source, destination, n, channels, height, width, prefix):
    """Explicitly zero the padded tensor, then copy its full interior."""
    pn, pc, py, px, pi = [prefix+'_'+s for s in ('n', 'c', 'y', 'x', 'i')]
    ph, pw = height+2, width+2
    return [L(pi, n*channels*ph*pw, [I('set', R(destination, **{pi: 1}), 0)])] + nested(
        [(pn,n),(pc,channels),(py,height),(px,width)],
        [I('copy', R(destination, pw+1, **{pn:channels*ph*pw,pc:ph*pw,py:pw,px:1}),
                   R(source, **{pn:channels*height*width,pc:height*width,py:width,px:1}))])


def conv_forward(padded, weights, output, n, cin, cout, height, width, prefix,
                 scratch='s', bias=None):
    pn, po, py, px, pi, pk, pl = [prefix+'_'+s for s in ('n','o','y','x','i','ky','kx')]
    product, accumulator = R(scratch,0), R(scratch,1)
    inner = nested([(pi,cin),(pk,3),(pl,3)], [
        I('mul', product,
          R(padded, **{pn:cin*(height+2)*(width+2),pi:(height+2)*(width+2),py:width+2,px:1,pk:width+2,pl:1}),
          R(weights, **{po:cin*9,pi:9,pk:3,pl:1})),
        I('add', accumulator, accumulator, product)])
    out = R(output, **{pn:cout*height*width,po:height*width,py:width,px:1})
    finish = ([I('add', out, accumulator, R(bias, **{po:1}))] if bias
              else [I('copy', out, accumulator)])
    return nested([(pn,n),(po,cout),(py,height),(px,width)],
                  [I('set', accumulator, 0)] + inner + finish)


def conv_weight_gradient(padded, delta, gradient, n, cin, cout, height, width, prefix,
                         scratch='s'):
    po, pi, pk, pl, pn, py, px = [prefix+'_'+s for s in ('o','i','ky','kx','n','y','x')]
    product, accumulator = R(scratch,0), R(scratch,1)
    inner = nested([(pn,n),(py,height),(px,width)], [
        I('mul', product,
          R(padded, **{pn:cin*(height+2)*(width+2),pi:(height+2)*(width+2),py:width+2,px:1,pk:width+2,pl:1}),
          R(delta, **{pn:cout*height*width,po:height*width,py:width,px:1})),
        I('add', accumulator, accumulator, product)])
    return nested([(po,cout),(pi,cin),(pk,3),(pl,3)], [I('set', accumulator,0)] + inner + [
        I('copy', R(gradient, **{po:cin*9,pi:9,pk:3,pl:1}), accumulator)])


def conv_input_gradient(padded_delta, weights, gradient, n, cin, cout, height, width,
                        prefix, scratch='s'):
    """Reduce ascending co,ky,kx against W[co,ci,2-ky,2-kx]."""
    pn, pi, py, px, po, pk, pl = [prefix+'_'+s for s in ('n','i','y','x','o','ky','kx')]
    product, accumulator = R(scratch,0), R(scratch,1)
    inner = nested([(po,cout),(pk,3),(pl,3)], [
        I('mul', product,
          R(padded_delta, **{pn:cout*(height+2)*(width+2),po:(height+2)*(width+2),py:width+2,px:1,pk:width+2,pl:1}),
          R(weights, 8, **{po:cin*9,pi:9,pk:-3,pl:-1})),
        I('add', accumulator, accumulator, product)])
    return nested([(pn,n),(pi,cin),(py,height),(px,width)], [I('set', accumulator,0)] + inner + [
        I('copy',R(gradient, **{pn:cin*height*width,pi:height*width,py:width,px:1}),accumulator)])


def conv_bias_gradient(delta, gradient, n, cout, height, width, prefix, scratch='s'):
    po, pn, py, px = [prefix+'_'+s for s in ('o','n','y','x')]
    accumulator = R(scratch,1)
    return nested([(po,cout)], [I('set',accumulator,0)] + nested(
        [(pn,n),(py,height),(px,width)], [I('add',accumulator,accumulator,
            R(delta, **{pn:cout*height*width,po:height*width,py:width,px:1}))]) + [
        I('copy',R(gradient, **{po:1}),accumulator)])


def relu_forward(source, destination, words, prefix, scratch='s', constants='k'):
    pi=prefix+'_i'
    condition=R(scratch,2)
    return [L(pi, words, [I('cmp',condition,R(constants,0),R(source, **{pi:1})),
        I('select',R(destination, **{pi:1}),condition,R(source, **{pi:1}),R(constants,0))])]


def relu_backward(activation, incoming, destination, words, prefix, scratch='s', constants='k'):
    pi=prefix+'_i'
    condition=R(scratch,2)
    return [L(pi, words, [I('cmp',condition,R(constants,0),R(activation, **{pi:1})),
        I('select',R(destination, **{pi:1}),condition,R(incoming, **{pi:1}),R(constants,0))])]


def momentum_update(weights, gradient, velocity, words, prefix, scratch='s', constants='k'):
    """k = [0, decay, momentum, lr]; all gradients already batch-normalized.

    d=FP32(g+FP32(decay*w)); v=FP32(FP32(momentum*v)+d);
    w=FP32(w-FP32(lr*v)). Every product is separate from its following add.
    """
    pi=prefix+'_i'
    product, adjusted=R(scratch,0),R(scratch,1)
    w,g,v=[R(name, **{pi:1}) for name in (weights,gradient,velocity)]
    return [L(pi,words,[I('mul',product,R(constants,1),w),I('add',adjusted,g,product),
        I('mul',product,R(constants,2),v),I('add',v,product,adjusted),
        I('mul',product,R(constants,3),v),I('sub',w,w,product)])]
