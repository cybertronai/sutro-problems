"""Ordered-FP32 panel lowering for all seven MNIST-small matrix products.

Based on 4x4 lifetime reuse and 16x16 asymmetric panels/persistent captures.
Cost decisions use v4 reads AND writes, not the matmul leaderboard's v0 tiers.
No CUDA, training hyperparameter search, or test-label access is performed here.
"""
import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'mnist/experiments/accuracy-il-20260911'
sys.path.insert(0,str(BASE))
import il
from il import ref as R, ins as I, loop as L
from mlp_il import build_mlp

STAGES = {'hidden':0,'output':1,'backward':2,'grad_w1':3,'grad_w2':5}
PRODUCTS = tuple(STAGES)+('infer_hidden','infer_output')
SHAPES = {'hidden':(25,9,32),'output':(25,32,10),'backward':(25,10,32),
          'grad_w1':(9,25,32),'grad_w2':(32,25,10),'infer_hidden':(8,9,32),'infer_output':(8,32,10)}


def substitute(value,bindings,acc=None,regions=None):
    if isinstance(value,list):
        return [substitute(v,bindings,acc,regions) for v in value]
    if not isinstance(value,dict):
        return value
    if 'region' not in value:
        return {k:substitute(v,bindings,acc,regions) for k,v in value.items()}
    if acc is not None and value==R('s',1):
        return acc
    out = copy.deepcopy(value)
    for var,(offset,coeffs) in bindings.items():
        scale = out['coefficients'].pop(var,0)
        out['offset'] += scale*offset
        for key,coefficient in coeffs.items():
            if scale*coefficient:
                out['coefficients'][key] = out['coefficients'].get(key,0)+scale*coefficient
    if regions and out['region'] in regions:
        name,drop = regions[out['region']]
        out['region'] = name
        for var in drop:
            out['coefficients'].pop(var,None)
    return out


def parts(size,width):
    """Disjoint affine panels. 6+10 alternation follows the 16x16 family."""
    widths = [6,10] if width=='6+10' else [int(width)]
    period = sum(widths)
    full,tail = divmod(size,period)
    offset = 0
    for n in widths:
        if full:
            yield offset,n,full,period
        offset += n
    offset = full*period
    for n in widths:
        count = min(n,tail)
        if count:
            yield offset,count,1,period
            offset += count
            tail -= count


def footprint(cfg,k):
    if cfg is None:
        return 0,0,0
    mr,nc,stage = cfg['m'],max([6,10] if cfg['n']=='6+10' else [cfg['n']]),cfg['stage']
    operand = (mr if stage in ('left','both') else 0)+(nc if stage in ('right','both') else 0)
    panel = mr*k if stage=='left_panel' else k*nc if stage=='right_panel' else 0
    return mr*nc,operand,panel


def lower(node,cfg):
    if cfg is None:
        return [copy.deepcopy(node)]
    outer,inner = node,node['body'][0]
    init,contraction,*finish = inner['body']
    assert init==I('set',R('s',1),0)
    mul,add = contraction['body']
    assert mul['op']=='mul' and add==I('add',R('s',1),R('s',1),R('s'))
    left,right = mul['src']
    assert inner['loop'] not in left['coefficients']
    assert outer['loop'] not in right['coefficients']
    assert outer['start']==inner['start']==contraction['start']==0
    stage = cfg['stage']
    kcount = contraction['count']
    maxn = max([6,10] if cfg['n']=='6+10' else [cfg['n']])
    groups = []
    for mo,mr,mc,ms in parts(outer['count'],cfg['m']):
        for no,nc,nn,ns in parts(inner['count'],cfg['n']):
            def bound(i,j,red=None):
                b = {outer['loop']:(mo+i,{'pm':ms}),inner['loop']:(no+j,{'pn':ns})}
                if red:
                    b[contraction['loop']] = (0,{red:1})
                return b
            init_body,step,end = [],[],[]
            for i in range(mr):
                if stage in ('left','both'):
                    step.append(I('copy',R('operand',i),substitute(left,bound(i,0))))
            right_offset = cfg['m'] if stage=='both' else 0
            for j in range(nc):
                if stage in ('right','both'):
                    step.append(I('copy',R('operand',right_offset+j),substitute(right,bound(0,j))))
            for i in range(mr):
                for j in range(nc):
                    acc = R('acc',i*maxn+j)
                    init_body.append(I('set',acc,0))
                    a,b = substitute(left,bound(i,j)),substitute(right,bound(i,j))
                    if stage in ('left','both'):
                        a = R('operand',i)
                    elif stage=='left_panel':
                        a = R('panel',i*kcount,**{contraction['loop']:1})
                    if stage in ('right','both'):
                        b = R('operand',right_offset+j)
                    elif stage=='right_panel':
                        b = R('panel',j,**{contraction['loop']:maxn})
                    step.extend([I('mul',R('s'),a,b),I('add',acc,acc,R('s'))])
                    end.extend(substitute(finish,bound(i,j),acc))
            body = init_body+[L(contraction['loop'],kcount,step)]+end
            if stage=='left_panel':
                fill = [I('copy',R('panel',i*kcount,pk=1),substitute(left,bound(i,0,'pk'))) for i in range(mr)]
                # Left operand has no pn coefficient: retain it across column panels.
                groups.append(L('pm',mc,[L('pk',kcount,fill),L('pn',nn,body)]))
            elif stage=='right_panel':
                fill = [I('copy',R('panel',j,pk=maxn),substitute(right,bound(0,j,'pk'))) for j in range(nc)]
                groups.append(L('pn',nn,[L('pk',kcount,fill),L('pm',mc,body)]))
            elif cfg.get('order','row')=='column':
                groups.append(L('pn',nn,[L('pm',mc,body)]))
            else:
                groups.append(L('pm',mc,[L('pn',nn,body)]))
    return groups


def default_config():
    return {'products':{name:None for name in PRODUCTS},'inference_batch':1,'cache_x':False,'layout':'original'}


def build(config=None,epochs=300,n_train=1000,n_test=1000):
    doc = build_mlp(32,epochs,.2,101,n_train,n_test,25)
    if config is None:
        return doc
    c = copy.deepcopy(config)
    cfgs = c['products']
    ib = c['inference_batch']
    assert 1<=ib<=25
    footprint_max = [0,0,0]
    for name,cfg in cfgs.items():
        for i,n in enumerate(footprint(cfg,SHAPES[name][1])):
            footprint_max[i] = max(footprint_max[i],n)
    extra = [(name,count) for name,count in zip(('acc','operand','panel'),footprint_max) if count]
    if c['cache_x']:
        extra.append(('xc',225))
    doc['regions'][1:1] = [{'name':name,'words':count} for name,count in extra]
    doc['body'][1:1] = [L('init',count,[I('set',R(name,init=1),0)]) for name,count in extra]
    layout = c['layout']
    front = {'original':[], 'd2_first':['d2'], 'w2_first':['w2','b2'],
             'h_first':['h'],'d1_first':['d1']}[layout]
    if front:
        selected = [r for name in front for r in doc['regions'] if r['name']==name]
        others = [r for r in doc['regions'] if r['name'] not in front]
        # Keep scalar/panel scratch and constants at the front; reorder bulk only.
        split = next(i for i,r in enumerate(others) if r['name']=='w1')
        doc['regions'] = others[:split]+selected+others[split:]
    epoch = next(n for n in doc['body'] if n.get('loop')=='epoch')
    stages = epoch['body'][0]['body']
    for name,index in STAGES.items():
        node = stages[index]
        if c['cache_x'] and name in ('hidden','grad_w1'):
            node = substitute(node,{},regions={'x':('xc',['batch'])})
        stages[index] = L('product_'+name,1,lower(node,cfgs[name]))
    if c['cache_x']:
        stages.insert(0,L('cache_x',225,[I('copy',R('xc',cache_x=1),R('x',batch=225,cache_x=1))]))
    qi = next(i for i,n in enumerate(doc['body']) if n.get('loop')=='query')
    if ib==1 and cfgs['infer_hidden'] is None and cfgs['infer_output'] is None and not c['cache_x']:
        # Exact scalar inference is a valid control, not an extra score buffer.
        pass
    else:
        groups = []
        for offset,rows,count,stride in parts(n_test,ib):
            q = R('q',offset*9,qblock=stride*9,r=9,f=1)
            if c['cache_x']:
                q = R('xc',r=9,f=1)
            hidden = L('r',rows,[L('h',32,[I('set',R('s',1),0),L('f',9,[
                I('mul',R('s'),q,R('w1',f=32,h=1)),I('add',R('s',1),R('s',1),R('s'))]),
                I('add',R('s',1),R('s',1),R('b1',h=1)),I('cmp',R('s',2),R('k'),R('s',1)),
                I('select',R('h',r=32,h=1),R('s',2),R('s',1),R('k'))])])
            output = L('r',rows,[L('c',10,[I('set',R('s',1),0),L('h',32,[
                I('mul',R('s'),R('h',r=32,h=1),R('w2',h=10,c=1)),
                I('add',R('s',1),R('s',1),R('s'))]),I('add',R('d2',r=10,c=1),R('s',1),R('b2',c=1))])])
            body = []
            if c['cache_x']:
                body.append(L('cache_q',rows*9,[I('copy',R('xc',cache_q=1),R('q',offset*9,qblock=stride*9,cache_q=1))]))
            body += [L('product_infer_hidden',1,lower(hidden,cfgs['infer_hidden'])),
                     L('product_infer_output',1,lower(output,cfgs['infer_output']))]
            argmax = [I('copy',R('s',3),R('d2',r=10)),I('copy',R('s',4),R('k',5))]
            for col in range(1,10):
                v = R('d2',col,r=10)
                argmax += [I('cmp',R('s',2),R('s',3),v),I('select',R('s',3),R('s',2),v,R('s',3)),
                           I('select',R('s',4),R('s',2),R('k',5+col),R('s',4))]
            argmax.append(I('send',R('s',4)))
            body.append(L('r',rows,argmax))
            groups.append(L('qblock',count,body))
        doc['body'][qi] = L('inference',1,groups)
    doc['metadata'].update({'algorithm':'ordered FP32 all-product panel MLP small', 'panel_config':c,
        'references':'4x4 lifetime/copy elimination; 16x16 asymmetric panels and persistent captures; repriced with v4 writes',
        'arithmetic':'Every output starts +0, ascending full K, separate mul/add, no partial sums or padding',
        'lifetime':'Shared panel/operand/acc buffers between products; xc between X@W1 and X.T@D1, then inference Q; d2 reused for inference scores'})
    return doc
