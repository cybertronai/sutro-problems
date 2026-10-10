import torch
import centered_halfnorm_small_members as centered_halfnorm
import embedding_graph,learned_ridge_head
import persistent_bf16_deep3_batchnorm16_staged_back128 as teacher_model
import persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3 as student_model
import persistent_bf16_deep4_batchnorm_staged_back128_ltsearch_head16_residual as deep_student
CONFIG=dict(lr1=.4,lr2=.4,head_lr=.4,momentum=.95,dropout=.1,weight_decay=.001,
 mixup=.2,input_scale=2.,initialization='data_sample',dictionary_scale=0.,
 data_noise=.25,swa_fraction=.25,schedule='cosine')
def classify(x,y,q):
 queries=torch.cat((x,q));N=len(queries)
 def fit(xx,yy,steps,is_student=False,seed=42,deep=False):
  backend=(deep_student if deep else student_model) if is_student else teacher_model
  E,width,batch=(4,512,2048) if is_student else (8,256,1024)
  backend.classify(xx,yy,queries,**dict(CONFIG,steps=steps,members=E,width=width,
   batch=batch,views=2,seed=seed,input_noise=0. if is_student else .15))
  state=backend.extension().debug_last_state();z=state[30][:E*N].contiguous();temp=100.
  if is_student:
   z=centered_halfnorm.transform(z,state[11][:E*N].contiguous(),E,len(x));return z
  p=learned_ridge_head.extension().knn(z,y,E,5,temp)
  p=(p/p.sum(1,keepdim=True).clamp_min(1e-20)).contiguous()
  return embedding_graph.extension().run(z,y,p,E,5,temp,.9,3)
 teacher=(fit(x,y,1000).clone()+fit(x,y,250,seed=1042))/2
 confidence,pseudo=teacher.max(1);ids=torch.nonzero(confidence>=.999,as_tuple=False).flatten()
 order=torch.arange(len(q),device=q.device)
 if ids.numel():ids=ids[order%ids.numel()];px,py=q[ids],pseudo[ids]
 else:px,py=x[order%len(x)],y[order%len(x)]
 xx,yy=torch.cat((x,px)),torch.cat((y,py))
 z0=fit(xx,yy,3000,True).clone();z1=fit(xx,yy,3000,True,deep=True)
 z1=(z1.float()*(z0.float().norm(dim=1,keepdim=True)/z1.float().norm(dim=1,keepdim=True).clamp_min(1e-12))).bfloat16()
 z=torch.cat((z0,z1),1).contiguous()
 p=learned_ridge_head.extension().knn(z,y,1,5,10.);p=(p/p.sum(1,keepdim=True).clamp_min(1e-20)).contiguous()
 return embedding_graph.extension().run(z,y,p,1,5,10.,.9,3).argmax(1)
