#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>
using B=__nv_bfloat16;
void graph(const B*,const long*,const float*,B*,float*,int*,float*,float*,float*,int,int,int,int,int,float,float,int,cudaStream_t);
at::Tensor run(at::Tensor z,at::Tensor y,at::Tensor p,int E,int k,double temp,double alpha,int steps){TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&p.is_cuda()&&z.is_contiguous()&&y.is_contiguous()&&p.is_contiguous()&&z.scalar_type()==at::kBFloat16&&y.scalar_type()==at::kLong&&p.scalar_type()==at::kFloat&&k>=1&&k<=15&&steps>=0&&alpha>=0&&alpha<=1);TORCH_CHECK(E>0&&z.dim()==2&&y.dim()==1&&p.dim()==2&&z.size(0)%E==0&&z.device()==y.device()&&z.device()==p.device()&&std::isfinite(temp)&&temp>=0);c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1);TORCH_CHECK(n>0&&n<N&&h>0&&k<N&&p.size(0)==N-n&&p.size(1)==10);auto f=at::empty({N,E*h},z.options()),s=at::empty({N-n,N},p.options()),ids=at::empty({N,k},y.options().dtype(at::kInt)),w=at::empty({N,k},p.options()),a=at::empty({N,10},p.options()),b=at::empty_like(a);graph((B*)z.data_ptr(),y.data_ptr<long>(),p.data_ptr<float>(),(B*)f.data_ptr(),s.data_ptr<float>(),ids.data_ptr<int>(),w.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),N,n,h,E,k,temp,alpha,steps,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return a.slice(0,n,N);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
