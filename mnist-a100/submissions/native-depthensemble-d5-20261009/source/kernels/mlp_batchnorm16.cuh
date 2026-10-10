// Fixed-scale per-feature batch normalization, separate ensemble members.
namespace mlp_bn16 {
__global__ void forward(const B*u,B*z,int b,int m,float dropout,unsigned seed){
 int t=threadIdx.x,c=(blockIdx.x%((m+15)/16))*16+(t&15),e=blockIdx.x/((m+15)/16),group=t>>4;
 __shared__ float sums[16][16],squares[16][16],mean[16],inv[16];float s=0,q=0;
 for(int r=group;r<b;r+=16)if(c<m){float x=__bfloat162float(u[(e*b+r)*m+c]);s+=x;q+=x*x;}
 sums[group][t&15]=s;squares[group][t&15]=q;__syncthreads();
 if(t<16){s=0;q=0;for(int k=0;k<16;k++){s+=sums[k][t];q+=squares[k][t];}mean[t]=s/b;inv[t]=rsqrtf(fmaxf(q/b-mean[t]*mean[t],0.f)+1e-5f);}__syncthreads();
 for(int r=group;r<b;r+=16)if(c<m){float v=fmaxf((__bfloat162float(u[(e*b+r)*m+c])-mean[t&15])*inv[t&15],0.f);B rounded=__float2bfloat16_rn(v);if(dropout>0){float random=(pb_hash(r*m+c+seed+e*101)+1.f)/4294967296.f;rounded=__float2bfloat16_rn(random<dropout?0:__bfloat162float(rounded)/(1-dropout));}z[(e*b+r)*m+c]=rounded;}
}
__global__ void backward(B*g,const B*z,const B*u,int b,int m,float gain){
 int t=threadIdx.x,c=(blockIdx.x%((m+15)/16))*16+(t&15),e=blockIdx.x/((m+15)/16),group=t>>4;
 __shared__ float sums[16][16],squares[16][16],mean[16],inv[16],gm[16],gx[16];float s=0,q=0;
 for(int r=group;r<b;r+=16)if(c<m){float x=__bfloat162float(u[(e*b+r)*m+c]);s+=x;q+=x*x;}
 sums[group][t&15]=s;squares[group][t&15]=q;__syncthreads();
 if(t<16){s=0;q=0;for(int k=0;k<16;k++){s+=sums[k][t];q+=squares[k][t];}mean[t]=s/b;inv[t]=rsqrtf(fmaxf(q/b-mean[t]*mean[t],0.f)+1e-5f);}__syncthreads();s=0;q=0;
 for(int r=group;r<b;r+=16)if(c<m){int i=(e*b+r)*m+c;float v=__bfloat162float(z[i])>0?gain*__bfloat162float(g[i]):0.f;s+=v;q+=v*(__bfloat162float(u[i])-mean[t&15])*inv[t&15];}
 sums[group][t&15]=s;squares[group][t&15]=q;__syncthreads();
 if(t<16){s=0;q=0;for(int k=0;k<16;k++){s+=sums[k][t];q+=squares[k][t];}gm[t]=s/b;gx[t]=q/b;}__syncthreads();
 for(int r=group;r<b;r+=16)if(c<m){int i=(e*b+r)*m+c;float v=__bfloat162float(z[i])>0?gain*__bfloat162float(g[i]):0.f;float normalized=(__bfloat162float(u[i])-mean[t&15])*inv[t&15];g[i]=__float2bfloat16_rn(inv[t&15]*(v-gm[t&15]-normalized*gx[t&15]));}
}
inline void act(const B*u,B*z,int rows,int m,int b,float p,unsigned seed,cudaStream_t stream){forward<<<(rows/b)*((m+15)/16),256,0,stream>>>(u,z,b,m,p,seed);}
inline void back(B*g,const B*z,const B*u,int rows,int m,int b,float gain,cudaStream_t stream){backward<<<(rows/b)*((m+15)/16),256,0,stream>>>(g,z,u,b,m,gain);}
}
