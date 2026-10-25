"""Per-call debiased EMA of all learner parameters, updated with Adam.

The shadow starts at zero, resets for every call, and never affects gradients.
Inference uses EMA/(1-decay**STEPS); no cross-call learned state is retained.
"""
def transform(source,decay=.98):
 assert 0<=decay<1
 old='  o.ostep=torch.compile(o.opt.step)';assert source.count(old)==1
 source=source.replace(old,'  o.ema=[torch.zeros_like(p)for p in o.params]\n  o.ostep=torch.compile(o.ema_step)')
 old=' def step(o):';assert source.count(old)==1
 method=f''' def ema_step(o):
  o.opt.step()
  with torch.no_grad():torch._foreach_lerp_(o.ema,o.params,{1-decay})
'''
 source=source.replace(old,method+old)
 old='  o.counter.zero_()';assert source.count(old)==1
 source=source.replace(old,'  torch._foreach_zero_(o.ema)\n'+old)
 old='  both=net.predict(q)';assert source.count(old)==1
 source=source.replace(old,f'  torch._foreach_copy_(net.params,torch._foreach_div(net.ema,1-{decay}**STEPS))\n'+old)
 return source
