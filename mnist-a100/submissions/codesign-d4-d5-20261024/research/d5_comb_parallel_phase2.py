"""Collect decoder partial gradients/BN sums with parallel row reductions.

Preserve17parameter gradients and3normalization sums; floating reduction order
changes. Tensor tiles replace the phase2 serial loop over partial row blocks.
"""
def transform(source):
 import ast,copy
 tree=ast.parse(source)
 kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='comb_bwd_kernel')
 annotation=copy.deepcopy(next(a.annotation for a in kernel.args.args if a.arg=='PHASE'))
 kernel.args.args.extend([ast.arg(arg='NR',annotation=annotation),ast.arg(arg='NE',annotation=copy.deepcopy(annotation))])
 branch=next(n for n in ast.walk(kernel)if isinstance(n,ast.If)and ast.unparse(n.test)=='PHASE == 1')
 end=next(i for i,n in enumerate(branch.orelse)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='mdv'for t in n.targets))
 replacement='''CH=tl.cdiv(17*S,NPROG)
e=pid*CH+tl.arange(0,NE)
em=e<tl.minimum((pid+1)*CH,17*S)
rk=tl.arange(0,NR)
rm=rk<NRS
part=tl.load(WS+rk[:,None]*20*S+e[None,:],mask=rm[:,None]&em[None,:],other=0.)
tl.store(DP+e,tl.sum(part,0),mask=em)
off=rk[:,None]*20*S+c[None,:]
mask=rm[:,None]&cm[None,:]
v1=tl.load(WS+off+17*S,mask=mask,other=0.)
v2=tl.load(WS+off+18*S,mask=mask,other=0.)
v3=tl.load(WS+off+19*S,mask=mask,other=0.)
s1,s2,s3=tl.reduce((v1,v2,v3),0,comb_add3)
'''
 branch.orelse=ast.parse(replacement).body+branch.orelse[end:]
 helper=ast.parse('@triton.jit\ndef comb_add3(_0,_1,_2,_3,_4,_5):\n return _0+_3,_1+_4,_2+_5\n').body[0];tree.body.insert(tree.body.index(kernel),helper)
 host=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='comb_bwd')
 calls=[n for n in ast.walk(host)if isinstance(n,ast.Call)and isinstance(n.func,ast.Subscript)and isinstance(n.func.value,ast.Name)and n.func.value.id=='comb_bwd_kernel'];assert len(calls)==1
 calls[0].keywords.extend([ast.keyword(arg='NR',value=ast.parse('triton.next_power_of_2(nrs)',mode='eval').body),ast.keyword(arg='NE',value=ast.parse('triton.next_power_of_2(triton.cdiv(17*S,n*nrs))',mode='eval').body)])
 return ast.unparse(ast.fix_missing_locations(tree))+'\n'
