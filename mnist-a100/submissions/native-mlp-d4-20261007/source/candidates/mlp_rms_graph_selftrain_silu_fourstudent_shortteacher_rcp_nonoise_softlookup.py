import torch,embedding_graph,learned_ridge_head
from candidates.mlp_rms_d3_knn_fast import CONFIG,model
import persistent_bf16_deep3_rmsnorm_silu_fastexp_rcp_hoisted_ltsearch as student

def classify(x,y,q):
 queries=torch.cat((x,q));N=len(queries)
 def fit(xx,yy,steps,width=256,backend=model,batch=1024,members=8,seed=42):
  backend.classify(xx,yy,queries,**dict(CONFIG,steps=steps,width=width,head_lr=.4,batch=batch,members=members,seed=seed,input_noise=(0. if backend is student else CONFIG["input_noise"])))
  s=backend.extension().debug_last_state();z=s[30][:members*N].contiguous()
  if backend is student:
   a=z.reshape(members,N,width).permute(1,0,2).reshape(N,-1)
   b=s[11][:members*N].reshape(members,N,width).permute(1,0,2).reshape(N,-1)
   z=torch.cat((a,b),1).contiguous();members=1
  temperature=10. if backend is student else 100.
  p=learned_ridge_head.extension().knn(z,y,members,5,temperature);p=(p/p.sum(1,keepdim=True).clamp_min(1e-20)).contiguous()
  return embedding_graph.extension().run(z,y,p,members,5,temperature,.9,3)
 teacher=(fit(x,y,1000,seed=42).clone()+fit(x,y,250,seed=1042))/2;confidence,pseudo=teacher.max(1);keep=confidence>=0.999
 ids=torch.nonzero(keep,as_tuple=False).flatten();order=torch.arange(len(q),device=q.device)
 if ids.numel():ids=ids[order%ids.numel()];px,py=q[ids],pseudo[ids]
 else:px,py=x[order%len(x)],y[order%len(x)]
 return fit(torch.cat((x,px)),torch.cat((y,py)),1000,512,backend=student,batch=2048,members=4).argmax(1)
