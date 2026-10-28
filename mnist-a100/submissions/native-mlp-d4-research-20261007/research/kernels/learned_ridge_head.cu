#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <cmath>
using B=__nv_bfloat16;
static void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("ridge cuBLAS failed");}
static void cs(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("ridge cuSOLVER failed");}
__global__ void pack(const B*z,B*f,int N,int h,int E){int r=blockIdx.x,t=threadIdx.x,D=E*h;__shared__ float sums[8];float sum=0;for(int j=t;j<D;j+=256){float v=__bfloat162float(z[((j/h)*N+r)*h+j%h]);sum+=v*v;}for(int k=16;k;k>>=1)sum+=__shfl_down_sync(0xffffffff,sum,k);if((t&31)==0)sums[t/32]=sum;__syncthreads();if(t==0){sum=0;for(int j=0;j<8;j++)sum+=sums[j];sums[0]=rsqrtf(fmaxf(sum,1e-20f));}__syncthreads();for(int j=t;j<D;j+=256)f[r*D+j]=__float2bfloat16_rn(__bfloat162float(z[((j/h)*N+r)*h+j%h])*sums[0]);}
__global__ void targets(const long*y,B*t,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n*16)t[i]=__float2bfloat16_rn(i%16==y[i/16]?1.f:0.f);}
__global__ void diag(float*g,int D,float v){int i=blockIdx.x*256+threadIdx.x;if(i<D)g[i*D+i]+=v;}
__global__ void weights(const float*b,B*w,int D){int i=blockIdx.x*256+threadIdx.x;if(i<D*16)w[i]=__float2bfloat16_rn(b[(i%16)*D+i/16]);}
void ridge(const B*z,const long*y,B*f,B*t,float*g,float*b,B*w,float*out,int*info,int N,int n,int h,int E,float lambda,cudaStream_t stream){
 static cublasHandle_t handle=nullptr;static cusolverDnHandle_t solver=nullptr;if(!handle){cb(cublasCreate(&handle));cs(cusolverDnCreate(&solver));}cb(cublasSetStream(handle,stream));cs(cusolverDnSetStream(solver,stream));int D=E*h;float one=1,zero=0;
 pack<<<N,256,0,stream>>>(z,f,N,h,E);targets<<<(n*16+255)/256,256,0,stream>>>(y,t,n);
 // Row-major F is column-major D x N. Form F^T F and F^T Y.
 cb(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,D,D,n,&one,f,CUDA_R_16BF,D,f,CUDA_R_16BF,D,&zero,g,CUDA_R_32F,D,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 cb(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,D,16,n,&one,f,CUDA_R_16BF,D,t,CUDA_R_16BF,16,&zero,b,CUDA_R_32F,D,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));diag<<<(D+255)/256,256,0,stream>>>(g,D,lambda);
 int length=0;cs(cusolverDnSpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,D,g,D,&length));float*scratch=nullptr;cudaMallocAsync(&scratch,length*sizeof(float),stream);cs(cusolverDnSpotrf(solver,CUBLAS_FILL_MODE_LOWER,D,g,D,scratch,length,info));cs(cusolverDnSpotrs(solver,CUBLAS_FILL_MODE_LOWER,D,16,g,D,b,D,info+1));cudaFreeAsync(scratch,stream);
 weights<<<(D*16+255)/256,256,0,stream>>>(b,w,D);cb(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,16,N-n,D,&one,w,CUDA_R_16BF,16,f+n*D,CUDA_R_16BF,D,&zero,out,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
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
__global__ void blend_layers(const B*a,const B*b,B*out,int h,float alpha){
 int r=blockIdx.x,t=threadIdx.x;float sa=0,sb=0;__shared__ float aa[8],bb[8];
 for(int j=t;j<h;j+=256){float x=__bfloat162float(a[r*h+j]),y=__bfloat162float(b[r*h+j]);sa+=x*x;sb+=y*y;}
 for(int k=16;k;k>>=1){sa+=__shfl_down_sync(0xffffffff,sa,k);sb+=__shfl_down_sync(0xffffffff,sb,k);}if((t&31)==0){aa[t/32]=sa;bb[t/32]=sb;}__syncthreads();
 if(t==0){sa=0;sb=0;for(int i=0;i<8;i++){sa+=aa[i];sb+=bb[i];}aa[0]=sqrtf(alpha)*rsqrtf(fmaxf(sa,1e-20f));bb[0]=sqrtf(1-alpha)*rsqrtf(fmaxf(sb,1e-20f));}__syncthreads();
 for(int j=t;j<h;j+=256){out[r*2*h+j]=__float2bfloat16_rn(__bfloat162float(a[r*h+j])*aa[0]);out[r*2*h+h+j]=__float2bfloat16_rn(__bfloat162float(b[r*h+j])*bb[0]);}
}
void mix_layers(const B*a,const B*b,B*out,int rows,int h,float alpha,cudaStream_t stream){blend_layers<<<rows,256,0,stream>>>(a,b,out,h,alpha);}
// A supervised low-rank metric: append normalized ridge class coordinates.
__global__ void metric_blend(const B*f,const B*p,B*out,int D,float alpha){
 int r=blockIdx.x,t=threadIdx.x;__shared__ float scale;
 if(t==0){float sum=0;for(int j=0;j<10;j++){float v=__bfloat162float(p[r*16+j]);sum+=v*v;}scale=sqrtf(alpha)*rsqrtf(fmaxf(sum,1e-20f));}__syncthreads();
 for(int j=t;j<D+16;j+=256){float v=j<D?__bfloat162float(f[r*D+j])*sqrtf(1-alpha):(j<D+10?__bfloat162float(p[r*16+j-D])*scale:0.f);out[r*(D+16)+j]=__float2bfloat16_rn(v);}
}
void ridge_metric(const B*f,const float*b,B*w,B*p,B*out,int N,int D,float alpha,cudaStream_t stream){
 static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));weights<<<(D*16+255)/256,256,0,stream>>>(b,w,D);float one=1,zero=0;
 cb(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,16,N,D,&one,w,CUDA_R_16BF,16,f,CUDA_R_16BF,D,&zero,p,CUDA_R_16BF,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));metric_blend<<<N,256,0,stream>>>(f,p,out,D,alpha);
}
__global__ void blend_probabilities(const float*memory,const B*logits,float*out,int n,int N,int E,int C,int off,float alpha){
 int r=blockIdx.x*128+threadIdx.x;if(r>=n)return;float a[10],mx=-INFINITY,total=0,memtotal=0;
 for(int c=0;c<10;c++){float v=0;for(int e=0;e<E;e++)v+=__bfloat162float(logits[(e*N+off+r)*C+c]);a[c]=v/E;mx=fmaxf(mx,a[c]);memtotal+=memory[r*10+c];}
 for(int c=0;c<10;c++){a[c]=expf(a[c]-mx);total+=a[c];}
 for(int c=0;c<10;c++)out[r*10+c]=(1-alpha)*a[c]/total+alpha*memory[r*10+c]/fmaxf(memtotal,1e-30f);
}
void probability_blend(const float*memory,const B*logits,float*out,int n,int N,int E,int C,int off,float alpha,cudaStream_t stream){blend_probabilities<<<(n+127)/128,128,0,stream>>>(memory,logits,out,n,N,E,C,off,alpha);}

__global__ void votes_loo(const float*s,const long*y,float*out,int n,int k,float temperature){
 int r=blockIdx.x,t=threadIdx.x;float best[15];int indices[15];for(int j=0;j<k;j++){best[j]=-INFINITY;indices[j]=-1;}
 for(int i=t;i<n;i+=128){if(r<n&&i==r)continue;float v=s[r*n+i];if(v>best[k-1]){int j=k-1;while(j>0&&v>best[j-1]){best[j]=best[j-1];indices[j]=indices[j-1];j--;}best[j]=v;indices[j]=i;}}
 __shared__ float candidates[128],score[10];__shared__ int ids[128];if(t<10)score[t]=0;__syncthreads();
 for(int j=0;j<k;j++){
  candidates[t]=best[0];ids[t]=indices[0];__syncthreads();
  if(t==0){int winner=0;for(int i=1;i<128;i++)if(candidates[i]>candidates[winner])winner=i;int id=ids[winner];score[y[id]]+=expf((candidates[winner]-1)*temperature);ids[0]=winner;}
  __syncthreads();if(t==ids[0]){for(int i=1;i<k;i++){best[i-1]=best[i];indices[i-1]=indices[i];}best[k-1]=-INFINITY;indices[k-1]=-1;}__syncthreads();
 }
 if(t<10)out[r*10+t]=score[t];
}

void neighbor_all(const B*z,const long*y,B*f,float*similarity,float*out,int N,int n,int h,int E,int k,float temperature,cudaStream_t stream){
 static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));int D=E*h;float one=1,zero=0;pack<<<N,256,0,stream>>>(z,f,N,h,E);
 cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,N,D,&one,f,CUDA_R_16BF,D,f,CUDA_R_16BF,D,&zero,similarity,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));votes_loo<<<N,128,0,stream>>>(similarity,y,out,n,k,temperature);
}
__global__ void calibration_pack(const float*memory,const B*logits,B*out,int N,int E,int C,float scale){
 int r=blockIdx.x*128+threadIdx.x;if(r>=N)return;float a[10],mx=-INFINITY,sum=0,msum=0;
 for(int c=0;c<10;c++){a[c]=0;for(int e=0;e<E;e++)a[c]+=__bfloat162float(logits[(e*N+r)*C+c])/E;mx=fmaxf(mx,a[c]);msum+=memory[r*10+c];}
 for(int c=0;c<10;c++){a[c]=expf(a[c]-mx);sum+=a[c];}
 for(int c=0;c<32;c++){float v=c<10?(msum>0?scale*memory[r*10+c]/msum:scale*.1f):(c<20?a[c-10]/sum:(c==20?1.f:0.f));out[r*32+c]=__float2bfloat16_rn(v);}
}
void calibration(const float*memory,const B*logits,B*out,int N,int E,int C,float scale,cudaStream_t stream){calibration_pack<<<(N+127)/128,128,0,stream>>>(memory,logits,out,N,E,C,scale);}

// Retrieval keys are normalized; values retain the original activation scale.
__global__ void retrieve_ids(const float*s,int*ids,float*weights,int n,int k,float temperature){
 int r=blockIdx.x,t=threadIdx.x;float best[15];int idx[15];for(int j=0;j<k;j++){best[j]=-INFINITY;idx[j]=-1;}
 for(int i=t;i<n;i+=128){if(r<n&&i==r)continue;float v=s[r*n+i];if(v>best[k-1]){int j=k-1;while(j>0&&v>best[j-1]){best[j]=best[j-1];idx[j]=idx[j-1];j--;}best[j]=v;idx[j]=i;}}
 __shared__ float candidates[128];__shared__ int indices[128],winner;__shared__ float maximum;
 for(int j=0;j<k;j++){candidates[t]=best[0];indices[t]=idx[0];__syncthreads();if(t==0){winner=0;for(int i=1;i<128;i++)if(candidates[i]>candidates[winner])winner=i;ids[r*k+j]=indices[winner];if(j==0)maximum=candidates[winner];weights[r*k+j]=expf((candidates[winner]-maximum)*temperature);}
 __syncthreads();if(t==winner){for(int i=1;i<k;i++){best[i-1]=best[i];idx[i-1]=idx[i];}best[k-1]=-INFINITY;idx[k-1]=-1;}__syncthreads();}
 if(t==0){float sum=0;for(int j=0;j<k;j++)sum+=weights[r*k+j];for(int j=0;j<k;j++)weights[r*k+j]/=fmaxf(sum,1e-30f);}
}
__global__ void inject_values(const B*z,const int*ids,const float*weights,B*out,int N,int h,int E,int k,float alpha){
 int r=blockIdx.x,e=blockIdx.y;for(int j=threadIdx.x;j<h;j+=256){float v=0;for(int a=0;a<k;a++)v+=weights[r*k+a]*__bfloat162float(z[(e*N+ids[r*k+a])*h+j]);out[(e*N+r)*h+j]=__float2bfloat16_rn(__bfloat162float(z[(e*N+r)*h+j])+alpha*v);}
}
void memory_ids(const B*z,B*f,float*s,int*ids,float*w,int N,int n,int h,int E,int k,float temperature,cudaStream_t stream){
 static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));int D=E*h;float one=1,zero=0;pack<<<N,256,0,stream>>>(z,f,N,h,E);cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,N,D,&one,f,CUDA_R_16BF,D,f,CUDA_R_16BF,D,&zero,s,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));retrieve_ids<<<N,128,0,stream>>>(s,ids,w,n,k,temperature);
}
void residual_project(const B*z,const int*ids,const float*w,const B*head,B*out,float*logits,int N,int h,int E,int C,int k,float alpha,cudaStream_t stream){
 inject_values<<<dim3(N,E),256,0,stream>>>(z,ids,w,out,N,h,E,k,alpha);static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));float one=1,zero=0;
 cb(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,C,N,h,&one,head,CUDA_R_16BF,C,(long long)h*C,out,CUDA_R_16BF,h,(long long)N*h,&zero,logits,CUDA_R_32F,C,(long long)N*C,E,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__global__ void diffusion_init(const long*y,const float*prior,float*p,int N,int n){int i=blockIdx.x*256+threadIdx.x;if(i<N*10){int r=i/10,c=i%10;p[i]=r<n?(y[r]==c?1.f:0.f):prior[(r-n)*10+c];}}
__global__ void diffusion_step(const int*ids,const float*w,const long*y,const float*prior,const float*in,float*out,int N,int n,int k,float alpha){int i=blockIdx.x*256+threadIdx.x;if(i<N*10){int r=i/10,c=i%10;if(r<n){out[i]=y[r]==c?1.f:0.f;return;}float v=0;for(int j=0;j<k;j++)v+=w[r*k+j]*in[ids[r*k+j]*10+c];out[i]=(1-alpha)*prior[(r-n)*10+c]+alpha*v;}}
void diffuse(const int*ids,const float*w,const long*y,const float*prior,float*p0,float*p1,int N,int n,int k,float alpha,int steps,cudaStream_t stream){diffusion_init<<<(N*10+255)/256,256,0,stream>>>(y,prior,p0,N,n);for(int t=0;t<steps;t++){diffusion_step<<<(N*10+255)/256,256,0,stream>>>(ids,w,y,prior,p0,p1,N,n,k,alpha);float*tmp=p0;p0=p1;p1=tmp;}}

__global__ void group_views(const float*s,float*g,int n,int nq){int i=blockIdx.x*256+threadIdx.x;if(i<n*nq){int r=i/n,c=i%n;g[i]=fmaxf(s[r*2*n+c],s[r*2*n+n+c]);}}
void grouped_neighbor(const B*z,const long*y,B*f,float*s,float*g,float*out,int N,int n,int h,int E,int k,float temperature,cudaStream_t stream){static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,stream));int D=E*h,nq=N-2*n;float one=1,zero=0;pack<<<N,256,0,stream>>>(z,f,N,h,E);cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,2*n,nq,D,&one,f,CUDA_R_16BF,D,f+2*n*D,CUDA_R_16BF,D,&zero,s,CUDA_R_32F,2*n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));group_views<<<(n*nq+255)/256,256,0,stream>>>(s,g,n,nq);votes<<<nq,128,0,stream>>>(g,y,out,n,k,temperature);}
