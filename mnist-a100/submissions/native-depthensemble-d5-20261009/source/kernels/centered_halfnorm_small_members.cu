#include <cuda_runtime.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
// Mean fitting uses only first ntrue rows of each member, never query/pseudo truth.
__global__ void partial_mean(const B*last,const B*middle,float*partial,int N,int ntrue,int E,int chunks){__shared__ float sums[256];int c=blockIdx.x*32+(threadIdx.x&31),group=blockIdx.y,chunk=blockIdx.z,warp=threadIdx.x>>5;const B*src=(group<E?last:middle)+(group%E)*N*512;float sum=0;int end=min(ntrue,(chunk+1)*1024);for(int r=chunk*1024+warp;r<end;r+=8)sum+=__bfloat162float(src[r*512+c]);sums[threadIdx.x]=sum;__syncthreads();if(warp==0){float x=sums[c%32];for(int w=1;w<8;w++)x+=sums[w*32+c%32];partial[(group*chunks+chunk)*512+c]=x;}}
__global__ void finish_mean(const float*partial,float*means,int ntrue,int groups,int chunks){int i=blockIdx.x*256+threadIdx.x;if(i>=groups*512)return;int g=i/512,c=i%512;float x=0;for(int k=0;k<chunks;k++)x+=partial[(g*chunks+k)*512+c];means[i]=x/ntrue;}
template<int E>__global__ void transform(const B*last,const B*middle,const float*means,B*out,int N){constexpr int G=2*E,Count=(G+7)/8;int r=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;__shared__ float norms[G],average;float v[Count][16];
 #pragma unroll
 for(int k=0;k<Count;k++){int g=warp+k*8;if(g<G){const B*src=(g<E?last:middle)+(g%E)*N*512+r*512;float sum=0;
 #pragma unroll
 for(int j=0;j<16;j++){int c=lane+j*32;v[k][j]=__bfloat162float(src[c])-means[g*512+c];sum+=v[k][j]*v[k][j];}
 for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);if(lane==0)norms[g]=fmaxf(sqrtf(sum),1e-12f);}}
 __syncthreads();if(threadIdx.x==0){float x=0;for(int g=0;g<G;g++)x+=norms[g];average=x/G;}__syncthreads();
 #pragma unroll
 for(int k=0;k<Count;k++){int g=warp+k*8;if(g<G){float gain=sqrtf(average/norms[g]);
 #pragma unroll
 for(int j=0;j<16;j++){int c=lane+j*32;out[(r*G+g)*512+c]=__float2bfloat16_rn(v[k][j]*gain);}}}
}

cudaError_t centered_halfnorm_small_members_into(const B*last,const B*middle,B*out,float*means,float*partial,int E,int N,int ntrue,cudaStream_t s){int chunks=(ntrue+1023)/1024;partial_mean<<<dim3(16,2*E,chunks),256,0,s>>>(last,middle,partial,N,ntrue,E,chunks);finish_mean<<<(2*E*512+255)/256,256,0,s>>>(partial,means,ntrue,2*E,chunks);if(E==1)transform<1><<<N,256,0,s>>>(last,middle,means,out,N);else if(E==2)transform<2><<<N,256,0,s>>>(last,middle,means,out,N);else if(E==4)transform<4><<<N,256,0,s>>>(last,middle,means,out,N);else transform<8><<<N,256,0,s>>>(last,middle,means,out,N);return cudaGetLastError();}
cudaError_t centered_halfnorm_small_members_resources(int E,cudaFuncAttributes*attributes){if(E==1)return cudaFuncGetAttributes(attributes,transform<1>);if(E==2)return cudaFuncGetAttributes(attributes,transform<2>);if(E==4)return cudaFuncGetAttributes(attributes,transform<4>);return cudaFuncGetAttributes(attributes,transform<8>);}
