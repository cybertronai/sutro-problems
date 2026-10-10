#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
cudaError_t centered_halfnorm_small_members_into(const B*,const B*,B*,float*,float*,int,int,int,cudaStream_t);
cudaError_t centered_halfnorm_small_members_resources(int,cudaFuncAttributes*);
void into(at::Tensor last,at::Tensor middle,at::Tensor out,at::Tensor means,at::Tensor partial,int E,int ntrue){for(auto t:{last,middle,out})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16,"contiguous CUDA BF16");TORCH_CHECK((E==1||E==2||E==4||E==8)&&last.dim()==2&&last.size(1)==512&&last.size(0)%E==0&&last.sizes()==middle.sizes(),"shape/E");int N=last.size(0)/E,chunks=(ntrue+1023)/1024;TORCH_CHECK(ntrue>0&&ntrue<=N&&out.dim()==2&&out.size(0)==N&&out.size(1)==2*E*512,"output/true prefix");for(auto t:{means,partial})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kFloat,"FP32 workspace");TORCH_CHECK(means.numel()>=2*E*512&&partial.numel()>=2*E*chunks*512,"workspace");for(auto t:{middle,out,means,partial})TORCH_CHECK(t.device()==last.device(),"device");c10::cuda::CUDAGuard guard(last.device());auto s=at::cuda::getCurrentCUDAStream();auto l=reinterpret_cast<const B*>(last.data_ptr()),m=reinterpret_cast<const B*>(middle.data_ptr());TORCH_CHECK(centered_halfnorm_small_members_into(l,m,reinterpret_cast<B*>(out.data_ptr()),means.data_ptr<float>(),partial.data_ptr<float>(),E,N,ntrue,s)==cudaSuccess,"launch");}

std::vector<int64_t> resources(int E){TORCH_CHECK(E==1||E==2||E==4||E==8,"E");cudaFuncAttributes a;TORCH_CHECK(centered_halfnorm_small_members_resources(E,&a)==cudaSuccess,"resources");return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes)};}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("into",&into);m.def("resources",&resources);}
