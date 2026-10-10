#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
void neighbor(const __nv_bfloat16*,const long*,__nv_bfloat16*,float*,float*,int,int,int,int,int,float,cudaStream_t);
at::Tensor knn(at::Tensor z,at::Tensor y,int E,int k,double temperature){
 TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&z.device()==y.device()&&z.scalar_type()==at::kBFloat16&&y.scalar_type()==at::kLong&&z.is_contiguous()&&y.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&k>=1&&k<=15&&temperature>0);
 c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1),D=E*h;TORCH_CHECK(n>0&&n<N&&n>=k&&D<=8192);auto fp=z.options().dtype(at::kFloat);auto f=at::empty({N,D},z.options()),s=at::empty({N-n,n},fp),o=at::empty({N-n,10},fp);
 neighbor((const __nv_bfloat16*)z.data_ptr(),y.data_ptr<long>(),(__nv_bfloat16*)f.data_ptr(),s.data_ptr<float>(),o.data_ptr<float>(),N,n,h,E,k,temperature,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("knn",&knn);}
