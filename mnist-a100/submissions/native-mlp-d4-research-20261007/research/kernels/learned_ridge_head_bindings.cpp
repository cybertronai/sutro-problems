#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
void ridge(const __nv_bfloat16*,const long*,__nv_bfloat16*,__nv_bfloat16*,float*,float*,__nv_bfloat16*,float*,int*,int,int,int,int,float,cudaStream_t);
void neighbor(const __nv_bfloat16*,const long*,__nv_bfloat16*,float*,float*,int,int,int,int,int,float,cudaStream_t);
void mix_layers(const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*,int,int,float,cudaStream_t);
torch::Tensor layers(torch::Tensor a,torch::Tensor b,double alpha){
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.scalar_type()==torch::kBFloat16&&b.scalar_type()==torch::kBFloat16&&a.is_contiguous()&&b.is_contiguous()&&a.dim()==2&&a.sizes()==b.sizes()&&alpha>=0&&alpha<=1);
 c10::cuda::CUDAGuard guard(a.device());auto o=torch::empty({a.size(0),2*a.size(1)},a.options());mix_layers((const __nv_bfloat16*)a.data_ptr(),(const __nv_bfloat16*)b.data_ptr(),(__nv_bfloat16*)o.data_ptr(),a.size(0),a.size(1),alpha,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;
}
torch::Tensor knn(torch::Tensor z,torch::Tensor y,int E,int k,double temperature){
 TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&z.device()==y.device()&&z.scalar_type()==torch::kBFloat16&&y.scalar_type()==torch::kInt64&&z.is_contiguous()&&y.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&k>=1&&k<=15&&temperature>0);
 c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1),D=E*h;TORCH_CHECK(n>0&&n<N&&n>=k&&D<=8192);auto fp=z.options().dtype(torch::kFloat32);auto f=torch::empty({N,D},z.options()),s=torch::empty({N-n,n},fp),o=torch::empty({N-n,10},fp);
 neighbor((const __nv_bfloat16*)z.data_ptr(),y.data_ptr<long>(),(__nv_bfloat16*)f.data_ptr(),s.data_ptr<float>(),o.data_ptr<float>(),N,n,h,E,k,temperature,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;
}
std::vector<torch::Tensor> fit(torch::Tensor z,torch::Tensor y,int E,double lambda){
 TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&z.device()==y.device()&&z.scalar_type()==torch::kBFloat16&&y.scalar_type()==torch::kInt64&&z.is_contiguous()&&y.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&lambda>0);
 c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1),D=E*h;TORCH_CHECK(n>0&&n<N&&D<=4096);
 auto f=torch::empty({N,D},z.options()),t=torch::empty({n,16},z.options()),w=torch::empty({D,16},z.options());auto fp=z.options().dtype(torch::kFloat32);auto g=torch::empty({D,D},fp),b=torch::empty({16,D},fp),o=torch::empty({N-n,16},fp),info=torch::empty({2},y.options().dtype(torch::kInt32));
 ridge((const __nv_bfloat16*)z.data_ptr(),y.data_ptr<long>(),(__nv_bfloat16*)f.data_ptr(),(__nv_bfloat16*)t.data_ptr(),g.data_ptr<float>(),b.data_ptr<float>(),(__nv_bfloat16*)w.data_ptr(),o.data_ptr<float>(),info.data_ptr<int>(),N,n,h,E,lambda,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return {o,info,f,g,b};
}
void ridge_metric(const __nv_bfloat16*,const float*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,int,int,float,cudaStream_t);
torch::Tensor metric(torch::Tensor f,torch::Tensor b,double alpha){
 TORCH_CHECK(f.is_cuda()&&b.is_cuda()&&f.device()==b.device()&&f.scalar_type()==torch::kBFloat16&&b.scalar_type()==torch::kFloat32&&f.is_contiguous()&&b.is_contiguous()&&f.dim()==2&&b.dim()==2&&b.size(0)==16&&b.size(1)==f.size(1)&&alpha>=0&&alpha<=1);
 c10::cuda::CUDAGuard guard(f.device());int N=f.size(0),D=f.size(1);auto w=torch::empty({D,16},f.options()),p=torch::empty({N,16},f.options()),out=torch::empty({N,D+16},f.options());ridge_metric((const __nv_bfloat16*)f.data_ptr(),b.data_ptr<float>(),(__nv_bfloat16*)w.data_ptr(),(__nv_bfloat16*)p.data_ptr(),(__nv_bfloat16*)out.data_ptr(),N,D,alpha,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void probability_blend(const float*,const B*,float*,int,int,int,int,int,float,cudaStream_t);
torch::Tensor fuse(torch::Tensor memory,torch::Tensor logits,int E,int off,double alpha){
 TORCH_CHECK(memory.is_cuda()&&logits.is_cuda()&&memory.device()==logits.device()&&memory.scalar_type()==torch::kFloat32&&logits.scalar_type()==torch::kBFloat16&&memory.is_contiguous()&&logits.is_contiguous()&&memory.dim()==2&&memory.size(1)==10&&logits.dim()==2&&E>0&&logits.size(0)%E==0&&logits.size(1)>=10&&off>=0&&off+memory.size(0)<=logits.size(0)/E&&alpha>=0&&alpha<=1);
 c10::cuda::CUDAGuard guard(memory.device());auto out=torch::empty_like(memory);probability_blend(memory.data_ptr<float>(),(B*)logits.data_ptr(),out.data_ptr<float>(),memory.size(0),logits.size(0)/E,E,logits.size(1),off,alpha,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void neighbor_all(const B*,const long*,B*,float*,float*,int,int,int,int,int,float,cudaStream_t);
torch::Tensor memory_all(torch::Tensor z,torch::Tensor y,int E,int k,double temperature){
 TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&z.device()==y.device()&&z.scalar_type()==torch::kBFloat16&&y.scalar_type()==torch::kInt64&&z.is_contiguous()&&y.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&k>=1&&k<=15&&temperature>0);
 c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,n=y.numel(),h=z.size(1),D=E*h;TORCH_CHECK(n>k&&n<N&&D<=4096);auto fp=z.options().dtype(torch::kFloat32);auto f=torch::empty({N,D},z.options()),s=torch::empty({N,n},fp),out=torch::empty({N,10},fp);neighbor_all((B*)z.data_ptr(),y.data_ptr<long>(),(B*)f.data_ptr(),s.data_ptr<float>(),out.data_ptr<float>(),N,n,h,E,k,temperature,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void calibration(const float*,const B*,B*,int,int,int,float,cudaStream_t);
torch::Tensor calibration_features(torch::Tensor memory,torch::Tensor logits,int E,double scale){
 TORCH_CHECK(memory.is_cuda()&&logits.is_cuda()&&memory.device()==logits.device()&&memory.scalar_type()==torch::kFloat32&&logits.scalar_type()==torch::kBFloat16&&memory.is_contiguous()&&logits.is_contiguous()&&memory.dim()==2&&memory.size(1)==10&&logits.dim()==2&&E>0&&logits.size(0)==E*memory.size(0)&&logits.size(1)>=10&&scale>=0);
 c10::cuda::CUDAGuard guard(memory.device());auto out=torch::empty({memory.size(0),32},logits.options());calibration(memory.data_ptr<float>(),(B*)logits.data_ptr(),(B*)out.data_ptr(),memory.size(0),E,logits.size(1),scale,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

void memory_ids(const B*,B*,float*,int*,float*,int,int,int,int,int,float,cudaStream_t);
std::vector<torch::Tensor> retrieve(torch::Tensor z,int E,int n,int k,double temperature){
 TORCH_CHECK(z.is_cuda()&&z.scalar_type()==torch::kBFloat16&&z.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&n>k&&k>0&&k<=15&&temperature>0);c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,h=z.size(1);TORCH_CHECK(N>=n&&E*h<=4096);auto f=torch::empty({N,E*h},z.options()),s=torch::empty({N,n},z.options().dtype(torch::kFloat32)),ids=torch::empty({N,k},z.options().dtype(torch::kInt32)),w=torch::empty({N,k},s.options());memory_ids((B*)z.data_ptr(),(B*)f.data_ptr(),s.data_ptr<float>(),ids.data_ptr<int>(),w.data_ptr<float>(),N,n,h,E,k,temperature,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,w};
}
void residual_project(const B*,const int*,const float*,const B*,B*,float*,int,int,int,int,int,float,cudaStream_t);
std::vector<torch::Tensor> inject(torch::Tensor z,torch::Tensor ids,torch::Tensor w,torch::Tensor head,int E,double alpha){
 TORCH_CHECK(z.is_cuda()&&z.scalar_type()==torch::kBFloat16&&z.is_contiguous()&&z.dim()==2&&E>0&&z.size(0)%E==0&&ids.is_cuda()&&ids.device()==z.device()&&ids.scalar_type()==torch::kInt32&&ids.is_contiguous()&&ids.dim()==2&&w.device()==z.device()&&w.scalar_type()==torch::kFloat32&&w.is_contiguous()&&w.sizes()==ids.sizes()&&head.device()==z.device()&&head.scalar_type()==torch::kBFloat16&&head.is_contiguous()&&head.dim()==2);c10::cuda::CUDAGuard guard(z.device());int N=z.size(0)/E,h=z.size(1),C=head.size(1),k=ids.size(1);TORCH_CHECK(ids.size(0)==N&&head.size(0)==E*h&&C>=10&&k>0&&k<=15&&alpha>=0);auto out=torch::empty_like(z),logits=torch::empty({E*N,C},z.options().dtype(torch::kFloat32));residual_project((B*)z.data_ptr(),ids.data_ptr<int>(),w.data_ptr<float>(),(B*)head.data_ptr(),(B*)out.data_ptr(),logits.data_ptr<float>(),N,h,E,C,k,alpha,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,logits};
}

void diffuse(const int*,const float*,const long*,const float*,float*,float*,int,int,int,float,int,cudaStream_t);
torch::Tensor diffusion(torch::Tensor ids,torch::Tensor w,torch::Tensor y,torch::Tensor prior,double alpha,int steps){
 TORCH_CHECK(ids.is_cuda()&&ids.scalar_type()==torch::kInt32&&ids.is_contiguous()&&ids.dim()==2&&w.device()==ids.device()&&w.scalar_type()==torch::kFloat32&&w.is_contiguous()&&w.sizes()==ids.sizes()&&y.device()==ids.device()&&y.scalar_type()==torch::kInt64&&y.is_contiguous()&&y.dim()==1&&prior.device()==ids.device()&&prior.scalar_type()==torch::kFloat32&&prior.is_contiguous()&&prior.dim()==2&&prior.size(1)==10&&ids.size(0)==y.numel()+prior.size(0)&&alpha>=0&&alpha<=1&&steps>=0&&steps<=100&&ids.size(1)>0&&ids.size(1)<=15);c10::cuda::CUDAGuard guard(ids.device());int N=ids.size(0),n=y.numel(),k=ids.size(1);auto p0=torch::empty({N,10},prior.options()),p1=torch::empty_like(p0);diffuse(ids.data_ptr<int>(),w.data_ptr<float>(),y.data_ptr<long>(),prior.data_ptr<float>(),p0.data_ptr<float>(),p1.data_ptr<float>(),N,n,k,alpha,steps,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return (steps%2?p1:p0).narrow(0,n,N-n);
}
void grouped_neighbor(const B*,const long*,B*,float*,float*,float*,int,int,int,int,int,float,cudaStream_t);
torch::Tensor view_knn(torch::Tensor z,torch::Tensor y,int E,int k,double temperature){TORCH_CHECK(z.is_cuda()&&y.is_cuda()&&z.device()==y.device()&&z.scalar_type()==torch::kBFloat16&&y.scalar_type()==torch::kInt64&&z.is_contiguous()&&y.is_contiguous()&&z.dim()==2&&y.dim()==1&&E>0&&z.size(0)%E==0&&k>0&&k<=15&&temperature>0);int N=z.size(0)/E,n=y.numel(),h=z.size(1),D=E*h,nq=N-2*n;TORCH_CHECK(n>=k&&nq>0&&D<=8192);c10::cuda::CUDAGuard guard(z.device());auto fp=z.options().dtype(torch::kFloat32);auto f=torch::empty({N,D},z.options()),s=torch::empty({nq,2*n},fp),g=torch::empty({nq,n},fp),out=torch::empty({nq,10},fp);grouped_neighbor((B*)z.data_ptr(),y.data_ptr<long>(),(B*)f.data_ptr(),s.data_ptr<float>(),g.data_ptr<float>(),out.data_ptr<float>(),N,n,h,E,k,temperature,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("view_knn",&view_knn);m.def("diffusion",&diffusion);m.def("retrieve",&retrieve);m.def("inject",&inject);m.def("fit",&fit);m.def("knn",&knn);m.def("layers",&layers);m.def("metric",&metric);m.def("fuse",&fuse);m.def("memory_all",&memory_all);m.def("calibration_features",&calibration_features);}
