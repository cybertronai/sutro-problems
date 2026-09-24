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
Y=(Y0-m)@Wl; Tot=M@Wl; B=np.load('/tmp/whiten/B.npy')
P=Tot@B; a=np.abs(P/np.linalg.norm(P,axis=0,keepdims=True)); am=a.argmax(0)
TRUE=np.zeros((81,81),bool)
for i,j in TRUE_EDGES: TRUE[i,j]=TRUE[j,i]=True
E=int(sum(TRUE[act[i],act[j]] for i in range(d) for j in range(i+1,d)))
def prec(F,pix):
    A=np.sqrt(np.clip(F,0,None))
    S=topology.partial_correlation(A)
    iu=np.triu_indices(F.shape[1],1)
    ta=TRUE[pix[iu[0]],pix[iu[1]]]
    return ta[np.argsort(-S[iu])[:E]].mean()
print("true edges among %d active pixels: %d ; chance precision = %.3f"%(d,E,E/(d*(d-1)/2)))
print("A  true pixels (upper bound)                : %.3f"%prec(Xa[vis[:10000]],act))
print("C  whitened+rotated features                : %.3f"%prec(Y[:10000],act[np.arange(d)]))
print("D  min-volume-cone reconstruction           : %.3f"%prec(np.clip(Y[:10000]@B,0,None),act[am]))
