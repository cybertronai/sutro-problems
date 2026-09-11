"""Verified input-independent literal tables and static address compression.

Lookup syntax resolves while expanding the program, never from runtime memory.
Only embedded uint32 words and one pinned seed-only schedule generator exist.
"""
import base64
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent


def packed_table(values):
    values=np.ascontiguousarray(values,dtype='<u4').reshape(-1)
    raw=values.tobytes()
    return {'kind':'u32-zlib-base64','length':len(values),'sha256':hashlib.sha256(raw).hexdigest(),
            'data':base64.b64encode(zlib.compress(raw)).decode()}


def lookup(table,offset=0,**coefficients):
    return {'table':table,'offset':offset,'coefficients':coefficients}


class Tables:
    def __init__(self,definitions):
        self.definitions=definitions
        self.embedded={};self.epoch_cache=OrderedDict();self.schedule_sequences={}
        self.family_histograms={}
        self.schedule_module=None
        self.identities={}
        self.mask_sequences={}
        for name,spec in definitions.items():
            if not isinstance(name,str) or not name:raise ValueError('Invalid table name')
            if spec['kind']=='u32-zlib-base64':
                if set(spec)!={'kind','length','sha256','data'}:raise ValueError('Unexpected literal table fields')
                if type(spec['length']) is not int or not 0<spec['length']<=32_000_000:raise ValueError('Literal table size')
                decoder=zlib.decompressobj()
                raw=decoder.decompress(base64.b64decode(spec['data'],validate=True),spec['length']*4+1)
                if not decoder.eof or decoder.unused_data:raise ValueError('Invalid compressed literal length')
                if len(raw)!=spec['length']*4 or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Literal table identity')
                self.embedded[name]=np.frombuffer(raw,dtype='<u4')
            elif spec['kind']=='seeded-cnn-v1':
                required={'kind','field','seeds','epochs','n_train','side','augmentation','generator_sha256'}
                if spec['field']=='head_masks':required|={'head_width','dropout'}
                if set(spec) not in (required,required|{'epoch_manifests'}):
                    raise ValueError('Unexpected seeded-table fields')
                if spec['field'] not in ('order','indices','coefficients','head_masks'):raise ValueError('Unknown seeded-table field')
                if spec['field']=='head_masks' and (type(spec['head_width']) is not int or spec['head_width']<1 or spec['dropout']!=.2):raise ValueError('Unsupported fixed head dropout')
                if not isinstance(spec['seeds'],list) or not spec['seeds'] or any(type(s) is not int or s<0 for s in spec['seeds']):
                    raise ValueError('Invalid fixed seeds')
                if any(type(spec[k]) is not int or spec[k]<1 for k in ('epochs','n_train','side')):raise ValueError('Invalid schedule dimensions')
                if spec['augmentation'] not in ('none','mild_affine'):raise ValueError('Unsupported augmentation')
                manifests=spec.get('epoch_manifests')
                if manifests is not None:
                    if len(manifests)!=len(spec['seeds']) or any(len(rows)!=spec['epochs'] for rows in manifests):raise ValueError('Manifest shape')
                    for rows in manifests:
                        for epoch,row in enumerate(rows,1):
                            if row.get('epoch')!=epoch:raise ValueError('Manifest epoch')
                            for field in ('order','theta','mask','indices','coefficients'):
                                value=row.get(field+'_sha256','')
                                if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Manifest digest')
                            if spec['field']=='head_masks':
                                value=row.get('head_mask_sha256','')
                                if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Dropout manifest digest')
                self.identities[id(spec)]=(tuple(spec['seeds']),spec['epochs'],spec['n_train'],spec['side'],spec['augmentation'],
                    hashlib.sha256(json.dumps(manifests,sort_keys=True).encode()).hexdigest(),spec.get('head_width'),spec.get('dropout'))
                path=HERE/'ordered_backend/schedule.py'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['generator_sha256']:raise ValueError('Schedule source changed')
                if self.schedule_module is None:
                    module_spec=importlib.util.spec_from_file_location('ir_seed_schedule',path)
                    self.schedule_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(self.schedule_module)
            else:raise ValueError('Unsupported static table kind')

    def length(self,name):
        spec=self.definitions[name]
        if name in self.embedded:return len(self.embedded[name])
        factor=spec['head_width'] if spec['field']=='head_masks' else (1 if spec['field']=='order' else spec['side']**2*4)
        return len(spec['seeds'])*spec['epochs']*spec['n_train']*factor

    def validate_lookup(self,expression,scope,literal=False):
        if not isinstance(expression,dict) or set(expression)!={'table','offset','coefficients'}:raise ValueError('Malformed static lookup')
        name=expression['table']
        if name not in self.definitions:raise ValueError('Unknown literal table')
        low=high=expression['offset']
        if type(low) is not int or not isinstance(expression['coefficients'],dict):raise ValueError('Malformed literal index')
        for variable,coefficient in expression['coefficients'].items():
            if variable not in scope or type(coefficient) is not int:raise ValueError('Malformed literal index coefficient')
            first,count=scope[variable]
            a,b=coefficient*first,coefficient*(first+max(count,1)-1)
            low+=min(a,b);high+=max(a,b)
        if all(count for _,count in scope.values()) and not 0<=low<=high<self.length(name):raise ValueError('Literal index outside table')
        if literal:return 0,0xffffffff
        if name in self.embedded:
            values=self.embedded[name]
            return int(values.min()),int(values.max())
        spec=self.definitions[name]
        if spec['field']=='order':return 0,spec['n_train']-1
        if spec['field']=='indices':return 0,spec['n_train']*spec['side']**2
        raise ValueError('FP32 coefficients cannot be used as static scratch addresses')

    def epoch(self,spec,member,epoch):
        base=self.identities[id(spec)]
        key=(base,member,epoch)
        if key in self.epoch_cache and spec['field'] in self.epoch_cache[key]:
            self.epoch_cache.move_to_end(key)
            return self.epoch_cache[key]
        sequence_key=(base,member)
        if spec['field']=='head_masks':
            state=self.mask_sequences.get(sequence_key)
            if state is None or state[0]>epoch:
                state=[0,iter(self.schedule_module.head_masks(spec['seeds'][member],spec['n_train'],spec['epochs'],spec['head_width'],spec['dropout']))]
                self.mask_sequences[sequence_key]=state
            while state[0]<=epoch:
                mask=next(state[1]);state[0]+=1
            if 'epoch_manifests' in spec:
                actual=self.schedule_module.array_hash(mask)
                if actual!=spec['epoch_manifests'][member][epoch]['head_mask_sha256']:raise ValueError('Head dropout bytes differ from pinned GPU manifest')
            result={'head_masks':mask.reshape(-1).view(np.uint32)}
            self.epoch_cache[key]=result
            while len(self.epoch_cache)>2:self.epoch_cache.popitem(last=False)
            return result
        if sequence_key not in self.schedule_sequences:
            self.schedule_sequences[sequence_key]=list(self.schedule_module.schedules(
                spec['seeds'][member],spec['n_train'],spec['epochs'],spec['side'],spec['augmentation']))
        schedule=self.schedule_sequences[sequence_key][epoch]
        if spec['field']=='order':
            if 'epoch_manifests' in spec and self.schedule_module.array_hash(schedule['order'])!=spec['epoch_manifests'][member][epoch]['order_sha256']:
                raise ValueError('Seed-only order bytes differ from pinned GPU manifest')
            result={'order':schedule['order'].reshape(-1)}
            self.epoch_cache[key]=result
            while len(self.epoch_cache)>2:self.epoch_cache.popitem(last=False)
            return result
        indices,coefficients=self.schedule_module.maps(schedule,spec['side'])
        assert indices.min()>=0 and indices.max()<=spec['n_train']*spec['side']**2
        if 'epoch_manifests' in spec:
            actual=self.schedule_module.manifest(schedule,indices,coefficients)
            expected={k:v for k,v in spec['epoch_manifests'][member][epoch].items() if k!='head_mask_sha256'}
            if actual!=expected:raise ValueError('Seed-only schedule bytes differ from pinned GPU manifest')
        result={'order':schedule['order'].reshape(-1),'indices':indices.reshape(-1),
                'coefficients':coefficients.reshape(-1).view(np.uint32)}
        self.epoch_cache[key]=result
        while len(self.epoch_cache)>2:self.epoch_cache.popitem(last=False)
        return result

    def values(self,name,index):
        index=np.asarray(index,dtype=np.int64)
        if name in self.embedded:return self.embedded[name][index]
        spec=self.definitions[name]
        width=spec['n_train']*(spec['head_width'] if spec['field']=='head_masks' else (1 if spec['field']=='order' else spec['side']**2*4))
        groups=index//width
        result=np.empty(index.shape,np.uint32)
        for group in np.unique(groups):
            member,epoch=divmod(int(group),spec['epochs'])
            mask=groups==group
            result[mask]=self.epoch(spec,member,epoch)[spec['field']][index[mask]%width]
        return result

    def resolve(self,expression,environment):
        index=expression['offset']
        for variable,coefficient in expression['coefficients'].items():index=index+coefficient*environment[variable]
        return self.values(expression['table'],index)

    def verify_literal_manifests(self):
        # Address histograms necessarily visit all augmentation epochs. Dropout
        # is used only by set-immediates, so explicitly verify its retained bytes.
        for spec in self.definitions.values():
            if spec['kind']=='seeded-cnn-v1' and spec['field']=='head_masks' and 'epoch_manifests' in spec:
                for member in range(len(spec['seeds'])):
                    for epoch in range(spec['epochs']):self.epoch(spec,member,epoch)

    def address_histogram(self,operand,scope,words):
        expression=operand['lookup']
        spec=self.definitions[expression['table']]
        # Four bilinear neighbors share every loop index. Resolve them together
        # and retain four independent exact histograms, rather than regenerate
        # each epoch's fixed map once for each syntactically separate mul leaf.
        family=(spec['kind']=='seeded-cnn-v1' and spec['field']=='indices'
                and not operand['coefficients']
                and all(c%4==0 for c in expression['coefficients'].values()))
        if family:
            neighbor=expression['offset']%4
            expression={**expression,'offset':expression['offset']-neighbor}
            key=(json.dumps(expression,sort_keys=True),operand['offset'],scope,words)
            if key in self.family_histograms:return self.family_histograms[key][neighbor]
        variables=set(operand['coefficients'])|set(expression['coefficients'])
        active=[(var,first,count) for var,(first,count) in scope if var in variables]
        multiplier=math.prod(count for var,(_,count) in scope if var not in variables)
        total=math.prod(count for _,_,count in active)
        result=np.zeros((4,words) if family else words,np.int64)
        scope_dict=dict(scope)
        epoch_width=spec.get('n_train',0)*spec.get('side',0)**2*4
        # Common compiler form: member/epoch pick the fixed map, all remaining
        # indices select its pixels. Enumerate local map positions once and
        # stream actual epochs directly. This avoids sorting hundreds of millions
        # of repeated epoch IDs; it changes no address multiplicity.
        direct=(family and 'member' in scope_dict and 'epoch' in scope_dict
            and expression['coefficients'].get('member')==spec['epochs']*epoch_width
            and expression['coefficients'].get('epoch')==epoch_width)
        if direct:
            local=[item for item in active if item[0] not in ('member','epoch')]
            local_count=math.prod(count for _,_,count in local)
            if local_count<=2_000_000:
                linear=np.arange(local_count,dtype=np.int64);positions=np.full(local_count,expression['offset'],np.int64)
                for variable,start,count in reversed(local):
                    positions+=(linear%count+start)*expression['coefficients'][variable];linear//=count
                if positions.min()>=0 and positions.max()<epoch_width and np.all(positions%4==0):
                    positions//=4
                    mfirst,mcount=scope_dict['member'];efirst,ecount=scope_dict['epoch']
                    for member in range(mfirst,mfirst+mcount):
                        for epoch in range(efirst,efirst+ecount):
                            values=self.epoch(spec,member,epoch)['indices'].reshape(-1,4)[positions]
                            for k in range(4):result[k]+=np.bincount(values[:,k]+operand['offset'],minlength=words).astype(np.int64)*multiplier
                    self.family_histograms[key]=result
                    return result[neighbor]
        for first in range(0,total,262144):
            linear=np.arange(first,min(total,first+262144),dtype=np.int64)
            environment={}
            for variable,start,count in reversed(active):
                environment[variable]=linear%count+start
                linear=linear//count
            if family:
                index=expression['offset']
                for variable,coefficient in expression['coefficients'].items():index=index+coefficient*environment[variable]
                address=operand['offset']+self.values(expression['table'],np.asarray(index)[...,None]+np.arange(4)).astype(np.int64)
                if address.min()<0 or address.max()>=words:raise ValueError('Resolved static address outside region')
                for k in range(4):result[k]+=np.bincount(address[...,k].reshape(-1),minlength=words).astype(np.int64)*multiplier
                continue
            address=operand['offset']+self.resolve(expression,environment).astype(np.int64)
            for variable,coefficient in operand['coefficients'].items():address=address+coefficient*environment[variable]
            if np.size(address)==1:address=np.full(min(total-first,262144),np.asarray(address).item(),np.int64)
            if address.min()<0 or address.max()>=words:raise ValueError('Resolved static address outside region')
            result+=np.bincount(address,minlength=words).astype(np.int64)*multiplier
        if family:
            self.family_histograms[key]=result
            return result[neighbor]
        return result
