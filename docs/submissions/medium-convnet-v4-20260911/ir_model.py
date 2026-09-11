"""Complete scalar-v4 CNN training and inference, with seed-only literal maps.

Loops compress a straight-line program; every executed leaf remains charged.
This builder does not use training data, predictions, learned weights or labels.
"""
import hashlib
from pathlib import Path
import numpy as np
from ir_core import make_program, ref as R, ins as I, loop as L
from ir_tables import packed_table, lookup
import ir_conv as conv
import ir_dense as dense
import ir_bn as bn

HERE=Path(__file__).resolve().parent

def bits(value):return int(np.float32(value).view(np.uint32))

def uses_batch_norm(config):
    values=[config[k] for k in ('batch_norm','batchnorm') if k in config]
    if any(type(value) is not bool for value in values) or (len(values)==2 and values[0]!=values[1]):raise ValueError('Invalid/conflicting BatchNorm flags')
    return values[0] if values else False

def parameter_shapes(config):
    width,depth,head,side=[config[k] for k in ('width','depth','head_width','image_size')]
    shapes={}
    for i in range(depth):
        shapes[f'conv{i}.weight']=(width,1 if i==0 else width,3,3)
        if uses_batch_norm(config):
            shapes.update({f'bn{i}.weight':(width,),f'bn{i}.bias':(width,)})
    shapes.update({'head1.weight':(head,width*side*side),'head1.bias':(head,),
                   'head2.weight':(10,head),'head2.bias':(10,)})
    return shapes

def build(config,initial_arrays,schedule_manifests=None):
    """initial_arrays is one seed-generated name→FP32-array dictionary per member.

    The caller must verify the generation recipe separately. Literal words are
    retained in the program and used by expanded execution, never learned data.
    """
    cfg=dict(config)
    for key,expected in {'optimizer':'sgd','activation':'relu','normalization':'4*x-0.5','pooling':'none',
                         'conv_bias':False,'head_bias':True,'bn_epsilon':.00001,'bn_momentum':.1}.items():
        if key in cfg and cfg[key]!=expected:raise ValueError('Unsupported '+key+'; refusing to silently score another algorithm')
    for key in ('width','depth','head_width','image_size','epochs','batch_size','n_train','n_test'):
        if type(cfg[key]) is not int or cfg[key]<1:raise ValueError('Invalid '+key)
    if not cfg['seeds'] or len(cfg['seeds'])!=len(initial_arrays):raise ValueError('Seed/member count mismatch')
    loss={'approx_softmax_gradient':'approx_softmax','half_sum_mse_batch_mean':'mse'}.get(
        cfg.get('loss','approx_softmax'),cfg.get('loss','approx_softmax'))
    if loss not in ('mse','approx_softmax'):raise ValueError('Unsupported loss')
    n,q,b,side,c,d,h,e=[cfg[k] for k in ('n_train','n_test','batch_size','image_size','width','depth','head_width','epochs')]
    m=len(cfg['seeds']);p=side*side;flat=c*p
    batch_norm=uses_batch_norm(cfg)
    dropout=cfg.get('dropout',0)
    if dropout not in (0,.2):raise ValueError('Only fixed head dropout 0 or .2 is supported')
    if type(batch_norm) is not bool:raise ValueError('batch_norm must be Boolean')
    if batch_norm and cfg.get('momentum',.9)!=.9:raise ValueError('This BN lowering requires the declared SGD momentum .9')
    if batch_norm and min(b,n%b or b)*p<=1:raise ValueError('BN requires more than one sample/pixel per channel')
    shapes=parameter_shapes(cfg);sizes={k:int(np.prod(v)) for k,v in shapes.items()}
    parameter_words=sum(sizes.values())
    literal=[]
    for arrays in initial_arrays:
        if set(arrays)!=set(shapes):raise ValueError('Incorrect initial parameter names')
        for name,shape in shapes.items():
            value=np.asarray(arrays[name])
            if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Invalid initial parameter '+name)
            literal.append(value.ravel().view(np.uint32))
    # All numerical tensors share this single fixed physical allocation.
    regions=[('s',8),('k',24 if batch_norm else 20),('raw',n*p+1),('labels',n),('query',q*p),('sum',q*10),('prediction',q)]
    for name,words in sizes.items():
        regions.extend([(name,words),('g:'+name,words),('v:'+name,words)])
    regions.extend([('x',b*p),('y',b),('head_z',b*h),('head_a',b*h),('scores',b*10),
                    ('d2',b*10),('head_da',b*h),('head_dz',b*h)])
    if dropout:regions.append(('head_mask',b*h))
    for i in range(d):
        ci=1 if i==0 else c
        regions.append((f'pad{i}',b*ci*(side+2)**2))
        for kind in ('z','a','dz','da'):regions.append((f'{kind}{i}',b*c*p))
        if i:regions.append((f'dpad{i}',b*c*(side+2)**2))
        if batch_norm:
            for kind in ('convout','xhat','dconv'):regions.append((f'{kind}{i}',b*c*p))
            for kind in ('mean','variance','invstd','running_mean','running_variance'):regions.append((f'{kind}{i}',c))
    body=[L('zero',words,[I('set',R(name,zero=1),0)]) for name,words in regions]
    constants=[0,cfg.get('weight_decay',.0001),cfg.get('momentum',.9),0,1,4,.5,0,-16,1/1024]
    body += [I('set',R('k',j),bits(value)) for j,value in enumerate(constants)]
    body += [I('set',R('k',10+j),j) for j in range(10)]
    if batch_norm:body += [I('set',R('k',20),bits(.00001)),I('set',R('k',23),bits(.1))]
    # Canonical tape: all train pixels, all labels, all query pixels, then outputs.
    for name,words in [('raw',n*p),('labels',n),('query',q*p)]:
        body.append(L('receive',words,[I('recv',R(name,receive=1))]))
    source=HERE/'ordered_backend/schedule.py'
    spec={'kind':'seeded-cnn-v1','seeds':list(cfg['seeds']),'epochs':e,'n_train':n,'side':side,
          'augmentation':cfg.get('augmentation','mild_affine'),'generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    if schedule_manifests is not None:spec['epoch_manifests']=schedule_manifests
    tables={field:{**spec,'field':field} for field in ('order','indices','coefficients')}
    if dropout:tables['head_masks']={**spec,'field':'head_masks','head_width':h,'dropout':dropout}
    tables['initial']=packed_table(np.concatenate(literal))
    from importlib.util import spec_from_file_location,module_from_spec
    module_spec=spec_from_file_location('model_fixed_schedule',source)
    schedule=module_from_spec(module_spec);module_spec.loader.exec_module(schedule)
    tables['rates']=packed_table(np.asarray([schedule.learning_rate(cfg,j+1) for j in range(e)],np.float32).view(np.uint32))

    def forward(batch,prefix,training):
        result=[];previous='x'
        for i in range(d):
            ci=1 if i==0 else c
            result+=conv.pad_nchw(previous,f'pad{i}',batch,ci,side,side,f'{prefix}_pad{i}')
            result+=conv.conv_forward(f'pad{i}',f'conv{i}.weight',f'convout{i}' if batch_norm else f'z{i}',batch,ci,c,side,side,f'{prefix}_conv{i}')
            if batch_norm:
                result+=bn.forward(f'convout{i}',f'bn{i}.weight',f'bn{i}.bias',f'mean{i}',f'variance{i}',f'invstd{i}',
                    f'xhat{i}',f'z{i}',f'running_mean{i}',f'running_variance{i}',batch,c,side,side,f'{prefix}_bn{i}',training)
            result+=conv.relu_forward(f'z{i}',f'a{i}',batch*c*p,f'{prefix}_relu{i}')
            previous=f'a{i}'
        result+=dense.matmul(previous,'head1.weight','head_z',batch,flat,h,prefix+'_head1',bt=True)
        result+=dense.bias_add('head_z','head1.bias','head_z',batch,h,prefix+'_bias1')
        result+=conv.relu_forward('head_z','head_a',batch*h,prefix+'_headrelu')
        if training and dropout:
            result += [L(prefix+'_drop',batch*h,[I('mul',R('head_a',**{prefix+'_drop':1}),
                R('head_a',**{prefix+'_drop':1}),R('head_mask',**{prefix+'_drop':1}))])]
        result+=dense.matmul('head_a','head2.weight','scores',batch,h,10,prefix+'_head2',bt=True)
        result+=dense.bias_add('scores','head2.bias','scores',batch,10,prefix+'_bias2')
        return result

    def train_step(batch,first,prefix):
        # first is an affine row expression; all table lookups depend only on loops.
        r,px=prefix+'_r',prefix+'_pixel'
        row={**first,r:1};index={v:a*p*4 for v,a in row.items()}
        index.update({'member':e*n*p*4,'epoch':n*p*4,px:4})
        initial_offset=index.pop('_offset',0)
        interpolate=[]
        for neighbor in range(4):
            src=R('raw');src['lookup']=lookup('indices',initial_offset+neighbor,**index)
            interpolate.append(I('set',R('s',6),lookup('coefficients',initial_offset+neighbor,**index)))
            interpolate.append(I('mul',R('s',0 if neighbor else 1),src,R('s',6)))
            if neighbor:interpolate.append(I('add',R('s',1),R('s',1),R('s',0)))
        interpolate += [I('mul',R('s',1),R('s',1),R('k',5)),I('sub',R('x',**{r:p,px:1}),R('s',1),R('k',6))]
        order={**row,'member':e*n,'epoch':n};offset=order.pop('_offset',0)
        label=R('labels');label['lookup']=lookup('order',offset,**order)
        result=[I('set',R('k',7),bits(1/batch)),L(r,batch,[L(px,p,interpolate),I('copy',R('y',**{r:1}),label)])]
        if dropout:
            unit=prefix+'_unit'
            mask_coeff={v:a*h for v,a in row.items() if v!='_offset'}
            mask_coeff.update({'member':e*n*h,'epoch':n*h,unit:1})
            result.append(L(r,batch,[L(unit,h,[I('set',R('head_mask',**{r:h,unit:1}),
                lookup('head_masks',row.get('_offset',0)*h,**mask_coeff))])]))
        if batch_norm:result += [I('set',R('k',21),bits(1/(batch*p))),I('set',R('k',22),bits((batch*p)/(batch*p-1)))]
        result+=forward(batch,prefix+'_forward',True)
        result+=dense.delta('scores','y','d2',batch,prefix+'_loss',loss)
        result+=dense.matmul('d2','head_a','g:head2.weight',10,batch,h,prefix+'_gw2',at=True)
        result+=dense.rowsum('d2','g:head2.bias',batch,10,prefix+'_gb2')
        result+=dense.matmul('d2','head2.weight','head_da',batch,10,h,prefix+'_dh')
        if dropout:
            result += [L(prefix+'_ddrop',batch*h,[I('mul',R('head_da',**{prefix+'_ddrop':1}),
                R('head_da',**{prefix+'_ddrop':1}),R('head_mask',**{prefix+'_ddrop':1}))])]
        result+=conv.relu_backward('head_z','head_da','head_dz',batch*h,prefix+'_headback')
        result+=dense.matmul('head_dz',f'a{d-1}','g:head1.weight',h,batch,flat,prefix+'_gw1',at=True)
        result+=dense.rowsum('head_dz','g:head1.bias',batch,h,prefix+'_gb1')
        result+=dense.matmul('head_dz','head1.weight',f'da{d-1}',batch,h,flat,prefix+'_df')
        for i in reversed(range(d)):
            ci=1 if i==0 else c
            result+=conv.relu_backward(f'z{i}',f'da{i}',f'dz{i}',batch*c*p,prefix+f'_back{i}')
            gradient=f'dz{i}'
            if batch_norm:
                result+=bn.backward(f'dz{i}',f'xhat{i}',f'bn{i}.weight',f'invstd{i}',f'g:bn{i}.weight',f'g:bn{i}.bias',
                    f'dconv{i}',batch,c,side,side,prefix+f'_bnback{i}')
                gradient=f'dconv{i}'
            result+=conv.conv_weight_gradient(f'pad{i}',gradient,f'g:conv{i}.weight',batch,ci,c,side,side,prefix+f'_gw{i}')
            if i:
                result+=conv.pad_nchw(gradient,f'dpad{i}',batch,c,side,side,prefix+f'_dpad{i}')
                result+=conv.conv_input_gradient(f'dpad{i}',f'conv{i}.weight',f'da{i-1}',batch,ci,c,side,side,prefix+f'_dx{i}')
        for j,(name,words) in enumerate(sizes.items()):
            result+=conv.momentum_update(name,'g:'+name,'v:'+name,words,prefix+f'_update{j}')
        return result

    def infer(batch,first,prefix):
        row,px,cl=prefix+'_r',prefix+'_pixel',prefix+'_class'
        query_coeff={k:v*p for k,v in first.items() if k!='_offset'}
        query_coeff.update({row:p,px:1})
        result=[L(row,batch,[L(px,p,[I('mul',R('s',1),R('query',first.get('_offset',0)*p,**query_coeff),R('k',5)),
            I('sub',R('x',**{row:p,px:1}),R('s',1),R('k',6))])])]
        result+=forward(batch,prefix+'_forward',False)
        outcoeff={k:v*10 for k,v in first.items() if k!='_offset'};outcoeff.update({row:10,cl:1})
        out=R('sum',first.get('_offset',0)*10,**outcoeff)
        result += [L(row,batch,[L(cl,10,[I('add',out,out,R('scores',**{row:10,cl:1}))])])]
        return result

    member=[];offset=0
    for name,words in sizes.items():
        member += [L('parameter',words,[I('set',R(name,parameter=1),lookup('initial',offset,member=parameter_words,parameter=1)),
            I('set',R('v:'+name,parameter=1),0)])]
        offset+=words
    if batch_norm:
        for i in range(d):
            member.append(L('bn_channel',c,[I('set',R(f'running_mean{i}',bn_channel=1),0),
                I('set',R(f'running_variance{i}',bn_channel=1),bits(1))]))
    epoch=[I('set',R('k',3),lookup('rates',epoch=1))]
    if n//b:epoch.append(L('batch',n//b,train_step(b,{'batch':b},'train_full')))
    if n%b:epoch+=train_step(n%b,{'_offset':n//b*b},'train_tail')
    member.append(L('epoch',e,epoch))
    if q//b:member.append(L('query_batch',q//b,infer(b,{'query_batch':b},'infer_full')))
    if q%b:member+=infer(q%b,{'_offset':q//b*b},'infer_tail')
    body.append(L('member',m,member))
    # Ascending strict comparisons preserve the smallest class on exact ties.
    argmax=[I('copy',R('s',3),R('sum',query_row=10)),I('set',R('prediction',query_row=1),0)]
    for digit in range(1,10):
        value=R('sum',digit,query_row=10)
        argmax += [I('cmp',R('s',2),R('s',3),value),I('select',R('s',3),R('s',2),value,R('s',3)),
            I('select',R('prediction',query_row=1),R('s',2),R('k',10+digit),R('prediction',query_row=1))]
    argmax.append(I('send',R('prediction',query_row=1)))
    body.append(L('query_row',q,argmax))
    document=make_program(regions,body,{'algorithm':'ordered ReLU CNN with complete fresh training and ensemble inference',
        'config':cfg,'initial_parameter_words_per_member':parameter_words,
        'initialization_recipe':'ordered_backend/schedule.py initial_parameters; externally verified seed-only literal arrays',
        'tape_words_received':n*p+n+q*p,'tape_words_sent':q,
        'buffer_policy':'One fixed maximum-batch workspace reused by all epochs, queries, and sequential members; raw queries retained once.',
        'ensemble':'FP32 sum from +0 in listed member order; strict ascending-class argmax',
        'operations':'Separate FP32 RNE mul/add/sub/div; no fused multiply-add; static gather literals contain no runtime data.'})
    document['tables']=tables
    return document
