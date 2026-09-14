"""Fully expanded small complete learners versus an independent NumPy oracle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys
import numpy as np
from ir_model import build,parameter_shapes
from ir_core import Program,score,expand
from ir_machine import SemanticMachine

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'ordered_backend'))
def module(name):
    spec=importlib.util.spec_from_file_location('oracle_'+name,HERE/'ordered_backend'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
ref=module('cpu_ref');schedule=module('schedule')

def independent_maps(item,side):
    """Scalar coordinate/neighbor construction, independent of vectorized maps."""
    n=len(item['order']);indices=np.empty((n,side*side,4),np.int32);coef=np.empty(indices.shape,np.float32)
    centre=np.float32((side-1)/2)
    for row in range(n):
        theta=item['theta'][row]
        for y in range(side):
            for x in range(side):
                xx=np.float32(x)-centre;yy=np.float32(y)-centre
                sx=np.float32(np.float32(theta[0,0]*xx)+np.float32(theta[0,1]*yy))+theta[0,2]
                sy=np.float32(np.float32(theta[1,0]*xx)+np.float32(theta[1,1]*yy))+theta[1,2]
                ix,iy=int(np.floor(sx)),int(np.floor(sy));fx=np.float32(sx-np.float32(ix));fy=np.float32(sy-np.float32(iy))
                ox,oy=np.float32(1)-fx,np.float32(1)-fy
                for k,(dx,dy,c) in enumerate([(0,0,ox*oy),(1,0,fx*oy),(0,1,ox*fy),(1,1,fx*fy)]):
                    px,py=ix+dx,iy+dy
                    indices[row,y*side+x,k]=(int(item['order'][row])*side*side+py*side+px) if 0<=px<side and 0<=py<side else n*side*side
                    coef[row,y*side+x,k]=c
    return indices,coef

def case(side,depth,epochs,members,loss,batch_norm=False,dropout=0):
    cfg={'width':2,'head_width':3,'depth':depth,'image_size':side,'epochs':epochs,'batch_size':2,
         'n_train':3,'n_test':3,'seeds':[101+i for i in range(members)],'augmentation':'mild_affine',
         'learning_rate':.03,'momentum':.9,'weight_decay':.0001,'loss':loss}
    if batch_norm:cfg['batch_norm']=True
    if dropout:cfg['dropout']=dropout
    rng=np.random.default_rng(700+side)
    initial=[{name:(rng.normal(size=shape)*.1).astype(np.float32) for name,shape in parameter_shapes(cfg).items()} for _ in range(members)]
    train=rng.random((3,1,side,side),dtype=np.float32);labels=np.asarray([2,0,7],np.uint32)
    query=rng.random((3,1,side,side),dtype=np.float32)
    document=build(cfg,initial);program=Program(document)
    tape=np.concatenate((train.ravel().view(np.uint32),labels,query.ravel().view(np.uint32)))
    machine=SemanticMachine(program.coordinates,tape);machine.run(expand(document))
    total=np.zeros((3,10),np.float32)
    for member,seed in enumerate(cfg['seeds']):
        params={k:v.copy() for k,v in initial[member].items()};velocity={k:np.zeros_like(v) for k,v in params.items()}
        if batch_norm:
            import ir_bn_reference as bnref
            running={f'running_{kind}{i}':np.full(cfg['width'],value,np.float32) for i in range(depth) for kind,value in [('mean',0),('variance',1)]}
        if dropout:
            mask_rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,20261202]).spawn(3)[2]))
            mask_sequence=list(schedule.head_masks(seed,3,epochs,cfg['head_width'],dropout))
        for item in schedule.schedules(seed,3,epochs,side,'mild_affine'):
            if dropout:
                mask=(mask_rng.random((3,cfg['head_width']),dtype=np.float32)>=np.float32(.2)).astype(np.float32)*np.float32(1.25)
                np.testing.assert_array_equal(mask.view(np.uint32),mask_sequence[item['epoch']-1].view(np.uint32))
            indices,coefficients=independent_maps(item,side)
            actual_idx,actual_coef=schedule.maps(item,side)
            np.testing.assert_array_equal(indices,actual_idx)
            np.testing.assert_array_equal(coefficients.view(np.uint32),actual_coef.view(np.uint32))
            source=np.concatenate((train.reshape(-1),np.zeros(1,np.float32)))
            augmented=source[indices[:,:,0]]*coefficients[:,:,0]
            for neighbor in range(1,4):augmented=augmented+source[indices[:,:,neighbor]]*coefficients[:,:,neighbor]
            augmented=(augmented*np.float32(4)-np.float32(.5)).reshape(train.shape)
            ordered_labels=labels[item['order']]
            for first in range(0,3,2):
                if dropout:params,velocity,_,_=ref.network_step(augmented[first:first+2],ordered_labels[first:first+2],params,velocity,cfg,schedule.learning_rate(cfg,item['epoch']),running,mask[first:first+2])
                elif batch_norm:params,velocity,running=bnref.network_step(augmented[first:first+2],ordered_labels[first:first+2],params,velocity,cfg,schedule.learning_rate(cfg,item['epoch']),running)
                else:params,velocity,_,_=ref.network_step(augmented[first:first+2],ordered_labels[first:first+2],params,velocity,cfg,schedule.learning_rate(cfg,item['epoch']))
        for first in range(0,3,2):
            values=query[first:first+2]*np.float32(4)-np.float32(.5)
            scores=(ref.network_forward(values,params,depth,running,False) if dropout else (bnref.network_forward(values,params,depth,running,False)[0] if batch_norm else ref.network_forward(values,params,depth)))['scores']
            total[first:first+2]=total[first:first+2]+scores
    expected={'sum':total,'prediction':np.argmax(total,axis=1).astype(np.uint32),**params,**{'v:'+k:v for k,v in velocity.items()}}
    if batch_norm:expected.update(running)
    for name,value in expected.items():
        base,length=program.regions[name];actual=np.asarray(machine.memory[base:base+length],np.uint32)
        target=value.ravel() if value.dtype==np.uint32 else value.ravel().view(np.uint32)
        np.testing.assert_array_equal(actual,target,err_msg=name)
    result,reads,writes=score(document,include_counts=True)
    np.testing.assert_array_equal(reads,machine.read_counts);np.testing.assert_array_equal(writes,machine.write_counts)
    assert result['time_ticks_0_2_ps']==machine.time_ticks and result['energy_fj']==machine.energy_fj
    assert result['instructions']==dict(machine.instructions)
    return {'config':cfg,'program_sha256':result['program_sha256'],'expanded_instructions':result['total_instructions'],
        'all_parameter_velocity_score_prediction_bits_match':True,'all_address_histograms_and_geometry_costs_match':True,
        'all_schedule_neighbor_indices_and_coefficient_bits_match':True,'tape_words_received':result['input_tape_words'],
        'tape_words_sent':result['output_tape_words'],'area_words':result['area_um2_occupied_cells']}

if __name__=='__main__':
    start=time.perf_counter()
    cases=[case(3,2,2,2,'approx_softmax'),case(4,1,1,1,'mse'),case(9,1,1,1,'approx_softmax'),
           case(3,2,2,2,'approx_softmax',True),case(9,1,1,1,'approx_softmax',True),
           case(3,2,2,2,'approx_softmax',True,.2)]
    paths=sorted(HERE.glob('ir_*.py'))+[HERE/'ordered_backend/cpu_ref.py',HERE/'ordered_backend/schedule.py']
    output={'all_passed':True,'cases':cases,'wall_seconds':time.perf_counter()-start,
        'source_sha256':{str(p.relative_to(HERE.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope':'Complete synthetic small programs, including 81-feature inputs, tails, multiple epochs/members, all backward operations, seed-only gathers, exact approximate-softmax or MSE gradient, reset momentum, final summed inference. Synthetic initial words are independent test fixtures, not claimed production seeded initialization. No MNIST accuracy claim.'}
    (HERE/'ir-model-validation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
