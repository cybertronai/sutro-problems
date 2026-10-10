#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
using B=__nv_bfloat16;
static void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("graph GEMM");}
__global__ void pack(const B*z,B*f,int N,int h,int E){int r=blockIdx.x,t=threadIdx.x,D=E*h;__shared__ float sums[8];float sum=0;for(int j=t;j<D;j+=256){float v=__bfloat162float(z[((j/h)*N+r)*h+j%h]);sum+=v*v;}for(int k=16;k;k>>=1)sum+=__shfl_down_sync(0xffffffff,sum,k);if((t&31)==0)sums[t/32]=sum;__syncthreads();if(t==0){sum=0;for(int j=0;j<8;j++)sum+=sums[j];sums[0]=rsqrtf(fmaxf(sum,1e-20f));}__syncthreads();for(int j=t;j<D;j+=256)f[r*D+j]=__float2bfloat16_rn(__bfloat162float(z[((j/h)*N+r)*h+j%h])*sums[0]);}

__global__ void adjacency(const float*s,int*idsOut,float*wOut,int N,int n,int k,float temp){int r=blockIdx.x+n,t=threadIdx.x;float best[15];int id[15];for(int j=0;j<k;j++){best[j]=-INFINITY;id[j]=-1;}for(int i=t;i<N;i+=128){if(i==r)continue;float v=s[blockIdx.x*N+i];if(v>best[k-1]){int j=k-1;while(j>0&&v>best[j-1]){best[j]=best[j-1];id[j]=id[j-1];j--;}best[j]=v;id[j]=i;}}
__shared__ float vals[128],weights[15],maxv;__shared__ int ids[128],winning;
for(int j=0;j<k;j++){vals[t]=best[0];ids[t]=id[0];__syncthreads();if(t==0){int win=0;for(int i=1;i<128;i++)if(vals[i]>vals[win])win=i;winning=win;idsOut[r*k+j]=ids[win];if(j==0)maxv=vals[win];weights[j]=expf(temp*(vals[win]-maxv));}__syncthreads();if(t==winning){for(int i=1;i<k;i++){best[i-1]=best[i];id[i-1]=id[i];}best[k-1]=-INFINITY;id[k-1]=-1;}__syncthreads();}
if(t<k){float sum=0;for(int j=0;j<k;j++)sum+=weights[j];wOut[r*k+t]=weights[t]/sum;}}
__global__ void initial(const long*y,const float*p,float*out,int N,int n){int i=blockIdx.x*256+threadIdx.x;if(i<N*10){int r=i/10,c=i%10;out[i]=r<n?float(y[r]==c):p[(r-n)*10+c];}}
__global__ void diffuse(const int*ids,const float*w,const long*y,const float*p,const float*old,float*out,int N,int n,int k,float alpha){int i=blockIdx.x*256+threadIdx.x;if(i>=N*10)return;int r=i/10,c=i%10;if(r<n){out[i]=float(y[r]==c);return;}float sum=0;for(int j=0;j<k;j++)sum+=w[r*k+j]*old[ids[r*k+j]*10+c];out[i]=alpha*sum+(1-alpha)*p[(r-n)*10+c];}
void graph(const B*z,const long*y,const float*p,B*f,float*s,int*ids,float*w,float*a,float*b,int N,int n,int h,int E,int k,float temp,float alpha,int steps,cudaStream_t stream){static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));pack<<<N,256,0,stream>>>(z,f,N,h,E);int D=E*h;float one=1,zero=0;cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N-n,D,&one,f,CUDA_R_16BF,D,f+n*D,CUDA_R_16BF,D,&zero,s,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));adjacency<<<N-n,128,0,stream>>>(s,ids,w,N,n,k,temp);initial<<<(N*10+255)/256,256,0,stream>>>(y,p,a,N,n);for(int i=0;i<steps;i++){diffuse<<<(N*10+255)/256,256,0,stream>>>(ids,w,y,p,a,b,N,n,k,alpha);float*t=a;a=b;b=t;}if(steps%2)cudaMemcpyAsync(b,a,N*10*4,cudaMemcpyDeviceToDevice,stream);}
