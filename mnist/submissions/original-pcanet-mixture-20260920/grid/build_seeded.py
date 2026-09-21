"""Correct remaining seed and empty-cluster differences before qualification.

The first k outputs of CPU randperm(n, seed) need only k forward Fisher-Yates
swaps. MT19937's first k words are public seed-derived constants. Their modulus
by the runtime class count uses binary long division: every FP32 integer stays
below 120,000 and is exact. A k-entry swap map and a masked class-relative scan
select the same original training rows without indexed memory instructions.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_candidate as BC
from affine import ref as R,ins as I,loop as L,make_program
B=BC.B
rs=lambda i:R('rng_s',i)


def rank_selection(count, components):
    den,rem,tmp,target,vi,vj,cond=[rs(i) for i in range(7)]
    body=[]
    for i in range(components):
        ii=R('jval',i)
        body += [I('sub',den,count,ii),I('set',rem,B.raw(0)),
            L('seed_bit',32,[I('add',rem,rem,rem),
                I('add',rem,rem,R('rng_bits',i*32,c=components*32,seed_bit=1)),
                I('sub',tmp,rem,den),I('cmp',cond,rem,den),
                I('select',rem,cond,rem,tmp)]),
            I('add',target,rem,ii),I('copy',vi,ii),I('copy',vj,target)]
        for h in range(i):
            body += [I('cmp',cond,R('rng_keys',h),ii,predicate='eq'),
                I('select',vi,cond,R('rng_vals',h),vi),
                I('cmp',cond,R('rng_keys',h),target,predicate='eq'),
                I('select',vj,cond,R('rng_vals',h),vj)]
        body += [I('copy',R('rng_pick',i),vj),I('copy',R('rng_keys',i),target),
                 I('copy',R('rng_vals',i),vi)]
    return body


def setup_regions(components):
    regions=[('rng_s',10),('rng_keys',components),('rng_vals',components),
             ('rng_pick',components),('rng_bits',10*components*32),('exp_s',2),('rng_eps',1)]
    init=[L('seed_init',size,[I('set',R(name,seed_init=1),0)]) for name,size in regions]
    init.append(I('set',R('rng_eps',0),B.raw(1e-9)))
    for c in range(10):
        words=np.random.RandomState(900+c).randint(0,2**32,components,dtype=np.uint32)
        for i,word in enumerate(words):
            for bit in range(32):
                init.append(I('set',R('rng_bits',(c*components+i)*32+bit),
                              B.raw((int(word)>>(31-bit))&1)))
    return regions,init


def seed_centroids(N,K,components):
    body=rank_selection(B.s_(B.S_CNT),components)
    ordinal,mask,joint=rs(7),rs(8),rs(9)
    body += [I('set',ordinal,B.raw(0)),
             L('seed_zero',components*K,[I('set',R('muA',seed_zero=1),0)]),
        L('seed_row',N,[
            I('cmp',mask,R('labels',seed_row=1),R('clsval',c=1),predicate='eq'),
            L('seed_component',components,[
                I('cmp',rs(6),ordinal,R('rng_pick',seed_component=1),predicate='eq'),
                I('select',joint,mask,rs(6),B.k_(B.KZ)),
                L('seed_dim',K,[I('select',R('muA',seed_component=K,seed_dim=1),joint,
                    R('zA',seed_row=K,seed_dim=1),R('muA',seed_component=K,seed_dim=1))])]),
            I('select',rs(2),mask,B.k_(B.KONE),B.k_(B.KZ)),
            I('add',ordinal,ordinal,rs(2))])]
    return body


ORIGINAL_EXP=B.exp_neg
def stable_exp(dst,src):
    # exp(-128) rounds to zero in FP32. Saturation therefore prevents the
    # old polynomial's large-negative-argument overflow without discarding a
    # representable target value at this boundary.
    arg,limit=R('exp_s',0),R('exp_s',1)
    return [I('set',limit,B.raw(128)),I('cmp',B.s_(B.S_C),src,limit),
            I('select',arg,B.s_(B.S_C),src,limit),*ORIGINAL_EXP(dst,arg)]


def build(**kw):
    B.EXPSH=10;B.exp_neg=stable_exp
    doc=BC.build(**kw)
    cfg=doc['metadata']['config'];N,K,k=cfg['N'],cfg['K'],cfg['components']
    regions,init=setup_regions(k)
    doc['regions'] += [{'name':name,'words':size} for name,size in regions]
    doc['body']=init+doc['body']
    for node in doc['body']:
        if node.get('loop')=='zq':
            first=next(i for i,n in enumerate(node['body'])
                       if n.get('op')=='copy' and n.get('dst')==B.s_(B.S_BEST))
            node['body']=node['body'][first:]
        if node.get('loop')!='c':continue
        old=node['body'];new=[];replaced=0
        for item in old:
            if item.get('loop')=='j' and len(item['body'])==1 and item['body'][0].get('loop')=='ic':
                new += seed_centroids(N,K,k);replaced+=1
            elif item.get('loop')=='km':
                comp=item['body'][1]
                assert comp['loop']=='j' and comp['body'][-1]['loop']=='nm'
                # A100 updates a center only when more than one row is assigned.
                comp['body'][-2:] = [
                    I('cmp',rs(6),B.k_(B.KONE),R('ccA',j=1)),
                    I('select',B.s_(B.S_DEN),rs(6),R('ccA',j=1),B.k_(B.KONE)),
                    L('nm',K,[I('div',rs(2),R('csA',j=K,nm=1),B.s_(B.S_DEN)),
                        I('select',R('muA',j=K,nm=1),rs(6),rs(2),R('muA',j=K,nm=1))])]
                new.append(item)
            else:new.append(item)
        assert replaced==1
        node['body']=new
        for leaf in BC.nodes(node['body']):
            if leaf.get('op')=='add' and B.s_(B.S_CNT) in leaf.get('src',[]):
                leaf['src']=[R('rng_eps',0) if src==B.k_(B.KTINY) else src for src in leaf['src']]
    cfg.update(mixture_seed_by_class=list(range(900,910)),responsibility_epsilon=1e-9,
               exp_squarings=10,exp_saturation_delta=128,
               empty_cluster_rule='retain previous centroid unless assigned count > 1')
    doc['metadata']['differences_from_A100']=[s for s in doc['metadata']['differences_from_A100']
                                             if not s.startswith('Fixed global row')]
    doc['metadata']['seed_implementation']='First k CPU randperm entries reproduced by exact-FP32 binary remainder and a sparse Fisher-Yates swap map; masked class-relative row selection.'
    doc['metadata']['prequalification_corrections']=[
        'Per-class randperm seeds 900+c and empty-cluster retention now match the submitted algorithm.',
        'Component responsibility epsilon corrected to 1e-9.',
        'Removed the historical bag wrapper score standardization for the single mixture member.',
        'Primitive exp saturates at an underflowing argument and uses ten squarings to avoid overflow.',
        'Corrections selected from source comparison and primitive validation before reading any full-run test accuracy.']
    return doc


def rng_document(components=8):
    regions,init=setup_regions(components)
    regions += [('counts',10),('jval',components),('picks',10*components)]
    init += [L('ri',10,[I('recv',R('counts',ri=1))]),
             L('zi',10*components,[I('set',R('picks',zi=1),0)])]
    init += [I('set',R('jval',i),B.raw(i)) for i in range(components)]
    body=init+[L('c',10,rank_selection(R('counts',c=1),components)+[
        L('save',components,[I('copy',R('picks',c=components,save=1),R('rng_pick',save=1))])]),
        L('out',10*components,[I('send',R('picks',out=1))])]
    return make_program(regions,body,{'test':'seeded permutation prefix'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reduced',action='store_true');p.add_argument('--rng-only',action='store_true')
    a=p.parse_args()
    kw=dict(N=60,Q=8,side=6,kernel=3,block=4,stride=1,L1=2,L2=2,
            K=4,components=2,filter_rounds=3,pca_rounds=3) if a.reduced else {}
    doc=rng_document() if a.rng_only else build(**kw)
    data=json.dumps(doc,sort_keys=True,separators=(',',':')).encode()
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_bytes(gzip.compress(data,mtime=0))
    print(json.dumps({'program_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}),flush=True)
