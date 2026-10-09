#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>
using B=__nv_bfloat16;
void graph(const B*,const long*,const float*,B*,float*,int*,float*,float*,float*,int,int,int,int,int,float,float,int,cudaStream_t);
torch::Tensor run(torch::Tensor z,torch::Tensor y,torch::Tensor p,int E,int k,double temp,double alpha,int steps){TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&p.is_cuda()&&z.is_contiguous()&&y.is_contiguous()&&p.is_contiguous()&&z.scalar_type()==torch::kBFloat16&&y.scalar_type()==torch::kInt64&&p.scalar_type()==torch::kFloat32&&k>=1&&k<=15&&steps>=0&&alpha>=0&&alpha<=1);TORCH_CHECK(E>0&&z.dim()==2&&y.dim()==1&&p.dim()==2&&z.size(0)%E==0&&z.device()==y.device()&&z.device()==p.device()&&std::isfinite(temp)&&temp>=0);c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1);TORCH_CHECK(n>0&&n<N&&h>0&&k<N&&p.size(0)==N-n&&p.size(1)==10);auto f=torch::empty({N,E*h},z.options()),s=torch::empty({N-n,N},p.options()),ids=torch::empty({N,k},y.options().dtype(torch::kInt32)),w=torch::empty({N,k},p.options()),a=torch::empty({N,10},p.options()),b=torch::empty_like(a);graph((B*)z.data_ptr(),y.data_ptr<long>(),p.data_ptr<float>(),(B*)f.data_ptr(),s.data_ptr<float>(),ids.data_ptr<int>(),w.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),N,n,h,E,k,temp,alpha,steps,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return a.slice(0,n,N);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
