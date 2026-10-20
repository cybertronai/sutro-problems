__device__ __forceinline__ float dot64(float v,float*s){int lane=threadIdx.x%32;for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)s[threadIdx.x/32]=v;__syncthreads();if(threadIdx.x==0)s[2]=s[0]+s[1];__syncthreads();return s[2];}
extern "C" __global__ void hull_cg(const float*A,const float*B,float*X,int d,int iterations){
 __shared__ float mat[64*65],p[64],s[3];int t=threadIdx.x,b=blockIdx.x;
 for(int i=t;i<64*64;i+=64){int row=i/64,col=i%64;mat[row*65+col]=(row<d&&col<d)?A[(b*d+row)*d+col]:0.f;}
 float x=0.f,r=t<d?B[b*d+t]:0.f;p[t]=r;__syncthreads();
 float rho=dot64(r*r,s),initial=rho;
 for(int it=0;it<iterations;++it){
  if(rho<=initial*1.e-12f)break;
  float ap=0.f;for(int j=0;j<64;++j)ap=fmaf(mat[t*65+j],p[j],ap);
  float den=dot64(p[t]*ap,s);float alpha=rho/fmaxf(den,1.e-30f);
  x=fmaf(alpha,p[t],x);r=fmaf(-alpha,ap,r);
  float next=dot64(r*r,s),beta=next/fmaxf(rho,1.e-30f);
  p[t]=fmaf(beta,p[t],r);__syncthreads();rho=next;
 }
 if(t<d)X[b*d+t]=x;
}
