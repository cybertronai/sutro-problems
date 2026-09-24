import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, topology
X=pool(); order=study.draw_order(2026092301); vis=order[:20000]
act=np.where(X.var(0)>1e-4)[0]; d=len(act); Xa=X[:,act]; mu=Xa.mean(0)
C=np.cov(Xa-mu,rowvar=False); w,V=np.linalg.eigh(C)
Wz=V@np.diag(1/np.sqrt(w+1e-3*w.mean()))@V.T
Q=rand_orth(777,d); M=Wz@Q.T
Y0=((Xa-mu)@M)[vis]; m=Y0.mean(0)
Cy=np.cov(Y0-m,rowvar=False); wy,Vy=np.linalg.eigh(Cy)
Wl=Vy@np.diag(1/np.sqrt(np.clip(wy,1e-9,None)))@Vy.T
Y=(Y0-m)@Wl; Tot=M@Wl
B=np.load('/tmp/whiten/B.npy')
P=Tot@B; a=np.abs(P/np.linalg.norm(P,axis=0,keepdims=True)); am=a.argmax(0)
Xrec=np.clip(Y@B,0,None)                      # recovered nonneg coordinates
F=np.zeros((10000,81),np.float32)
F[:,:d]=Xrec[:10000]
F[:,d:]=np.random.default_rng(0).normal(0,1e-6,(10000,81-d)).astype(np.float32)
pixel_of_feature=np.full(81,-1,np.int64)
pixel_of_feature[:d]=act[am]
left=[p for p in range(81) if p not in set(pixel_of_feature[:d].tolist())]
k=0
for j in range(81):
    if pixel_of_feature[j]<0: pixel_of_feature[j]=left[k]; k+=1
# duplicates: give the later duplicate a leftover pixel so the map is a bijection
seen={}; 
for j in range(81):
    p=pixel_of_feature[j]
    if p in seen: pixel_of_feature[j]=left[k]; k+=1
    else: seen[p]=j
lay=topology.recover_layout(F)
f,c=edge_preservation(lay,pixel_of_feature)
print(f"recover_layout on min-volume-cone reconstruction: {c}/144 true lattice edges preserved ({f:.3f}); chance 6.4/144; raw-protocol reference 133/144")
