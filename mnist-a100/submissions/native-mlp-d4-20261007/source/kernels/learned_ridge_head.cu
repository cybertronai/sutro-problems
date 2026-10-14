#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>
using B=__nv_bfloat16;
static void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("ridge cuBLAS failed");}
__global__ void pack(const B*z,B*f,int N,int h,int E){int r=blockIdx.x,t=threadIdx.x,D=E*h;__shared__ float sums[8];float sum=0;for(int j=t;j<D;j+=256){float v=__bfloat162float(z[((j/h)*N+r)*h+j%h]);sum+=v*v;}for(int k=16;k;k>>=1)sum+=__shfl_down_sync(0xffffffff,sum,k);if((t&31)==0)sums[t/32]=sum;__syncthreads();if(t==0){sum=0;for(int j=0;j<8;j++)sum+=sums[j];sums[0]=rsqrtf(fmaxf(sum,1e-20f));}__syncthreads();for(int j=t;j<D;j+=256)f[r*D+j]=__float2bfloat16_rn(__bfloat162float(z[((j/h)*N+r)*h+j%h])*sums[0]);}
__global__ void votes(const float*s,const long*y,float*out,int n,int k,float temperature){
 int r=blockIdx.x,t=threadIdx.x;float best[15];int indices[15];for(int j=0;j<k;j++){best[j]=-INFINITY;indices[j]=-1;}
 for(int i=t;i<n;i+=128){float v=s[r*n+i];if(v>best[k-1]){int j=k-1;while(j>0&&v>best[j-1]){best[j]=best[j-1];indices[j]=indices[j-1];j--;}best[j]=v;indices[j]=i;}}
 __shared__ float candidates[128],score[10];__shared__ int ids[128];if(t<10)score[t]=0;__syncthreads();
 for(int j=0;j<k;j++){
  candidates[t]=best[0];ids[t]=indices[0];__syncthreads();
  if(t==0){int winner=0;for(int i=1;i<128;i++)if(candidates[i]>candidates[winner])winner=i;int id=ids[winner];score[y[id]]+=expf((candidates[winner]-1)*temperature);ids[0]=winner;}
  __syncthreads();if(t==ids[0]){for(int i=1;i<k;i++){best[i-1]=best[i];indices[i-1]=indices[i];}best[k-1]=-INFINITY;indices[k-1]=-1;}__syncthreads();
 }
 if(t<10)out[r*10+t]=score[t];
}
void neighbor(const B*z,const long*y,B*f,float*s,float*out,int N,int n,int h,int E,int k,float temperature,cudaStream_t stream){
 static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));int D=E*h;float one=1,zero=0;pack<<<N,256,0,stream>>>(z,f,N,h,E);
 cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,N-n,D,&one,f,CUDA_R_16BF,D,f+n*D,CUDA_R_16BF,D,&zero,s,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));votes<<<N-n,128,0,stream>>>(s,y,out,n,k,temperature);
}
