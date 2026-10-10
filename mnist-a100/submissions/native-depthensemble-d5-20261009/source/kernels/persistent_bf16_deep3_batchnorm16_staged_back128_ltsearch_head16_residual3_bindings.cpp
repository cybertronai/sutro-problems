#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "persistent_bf16_deep3_batchnorm16_staged_back128_ltsearch_head16_residual3.h"
#include <map>
#include <sstream>
#include <iomanip>
struct PBWorkspace{cudaGraphExec_t graph;std::vector<at::Tensor> tensors;};
static thread_local PBWorkspace* pb_last=nullptr;
static std::map<std::string,PBWorkspace*>& pb_cache(){static std::map<std::string,PBWorkspace*> c;return c;}
at::Tensor classify(at::Tensor x,at::Tensor y,at::Tensor q,int m,int E,int batch,int views,int steps,double lr1,double lr2,double hlr,double momentum,int seed,bool graph,int schedule,double wd,double dropout,double noise,double input_scale,double mix,double init_gain,double head_gain,int init_kind,double swa_fraction,double dictionary_scale,double data_noise){
 TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&y.is_cuda()&&x.device()==y.device()&&x.device()==q.device()&&x.scalar_type()==at::kFloat&&q.scalar_type()==at::kFloat&&y.scalar_type()==at::kLong&&x.is_contiguous()&&y.is_contiguous()&&q.is_contiguous()&&x.dim()==2&&q.dim()==2&&x.size(1)==q.size(1)&&y.numel()==x.size(0)&&x.size(0)>0&&q.size(0)>0&&x.size(1)>0&&views>=1&&views<=8&&E>=1&&E<=16&&m>=128&&m%128==0&&m<=4096&&batch>0&&steps>=0&&schedule>=0&&schedule<=5&&wd>=0&&dropout>=0&&dropout<1&&noise>=0&&mix>=0&&mix<=.5&&init_gain>0&&head_gain>=0&&init_kind>=0&&init_kind<=7&&swa_fraction>=0&&swa_fraction<=1&&dictionary_scale>=0&&data_noise>=0&&data_noise<=1);
 c10::cuda::CUDAGuard device(x.device());int n=x.size(0),nq=q.size(0),original=x.size(1),d=((original+10+63)/64)*64,v=std::max(std::max(n,nq),std::min(batch,n)*views);auto sourcex=x,sourcey=y,sourceq=q;auto stream=at::cuda::getCurrentCUDAStream();
 std::ostringstream key;key<<std::setprecision(17)<<x.get_device()<<":"<<(uintptr_t)(cudaStream_t)stream<<":"<<n<<":"<<nq<<":"<<d<<":"<<original<<":"<<m<<":"<<E<<":"<<batch<<":"<<views<<":"<<steps<<":"<<lr1<<":"<<lr2<<":"<<hlr<<":"<<momentum<<":"<<seed<<":"<<schedule<<":"<<wd<<":"<<dropout<<":"<<noise<<":"<<input_scale<<":"<<mix<<":"<<init_gain<<":"<<head_gain<<":"<<init_kind<<":"<<swa_fraction<<":"<<dictionary_scale<<":"<<data_noise;
 auto found=pb_cache().find(key.str());PBWorkspace*ws=nullptr;
 if(!graph||found==pb_cache().end()){
  if(graph){x=at::empty_like(sourcex);y=at::empty_like(sourcey);q=at::empty_like(sourceq);}
  auto opts=x.options().dtype(at::kBFloat16);
  auto tx=at::empty({n,d},opts),tq=at::empty({nq,d},opts),w1=at::empty({E*m,d},opts),w2=at::empty({E*m,m},opts),h=at::empty({E*m,16},opts),u1=at::empty({E*v,m},opts),u2=at::empty_like(u1),z1=at::empty_like(u1),z2=at::empty_like(u1),du1=at::empty_like(u1),du2=at::empty_like(u1),dw1=at::empty_like(w1),dw2=at::empty_like(w2),dh=at::empty_like(h),err=at::empty({E*v,16},opts),logits=at::empty({E*nq,16},opts),v1=at::empty_like(w1),v2=at::empty_like(w2),vh=at::empty_like(h),aug=at::empty({E*std::min(batch,n)*views,d},opts),out=at::empty({nq},y.options());
  auto sw1=at::empty_like(w1,x.options()),sw2=at::empty_like(w2,x.options()),swh=at::empty_like(h,x.options());
  auto w3=at::empty_like(w2),u3=at::empty_like(u1),z3=at::empty_like(u1),du3=at::empty_like(u1),dw3=at::empty_like(w2),v3=at::empty_like(w2),sw3=at::empty_like(w2,x.options());
  auto res_z=at::empty_like(z1),res_g=at::empty_like(z1);
  auto targets=at::empty({n,10},opts),centroids=at::empty({10,d},opts);
  auto stats1=at::empty({E,4,m},x.options()),stats2=at::empty_like(stats1),stats3=at::empty_like(stats1);auto partial=at::empty({E,(v+255)/256,2,m},x.options());
  auto capture_stream=graph?at::cuda::getStreamFromPool(false,x.get_device()):stream;c10::cuda::CUDAStreamGuard guard(capture_stream);
  auto run=[&](const float*xx,const long*yy,const float*qq,int count){launch_pb(xx,yy,qq,out.data_ptr<long>(),(B*)tx.data_ptr(),(B*)tq.data_ptr(),(B*)w1.data_ptr(),(B*)w2.data_ptr(),(B*)h.data_ptr(),(B*)u1.data_ptr(),(B*)u2.data_ptr(),(B*)z1.data_ptr(),(B*)z2.data_ptr(),(B*)du1.data_ptr(),(B*)du2.data_ptr(),(B*)dw1.data_ptr(),(B*)dw2.data_ptr(),(B*)dh.data_ptr(),(B*)err.data_ptr(),(B*)logits.data_ptr(),(B*)v1.data_ptr(),(B*)v2.data_ptr(),(B*)vh.data_ptr(),(B*)aug.data_ptr(),(B*)targets.data_ptr(),(B*)centroids.data_ptr(),sw1.data_ptr<float>(),sw2.data_ptr<float>(),swh.data_ptr<float>(),(B*)w3.data_ptr(),(B*)u3.data_ptr(),(B*)z3.data_ptr(),(B*)du3.data_ptr(),(B*)dw3.data_ptr(),(B*)v3.data_ptr(),sw3.data_ptr<float>(),stats1.data_ptr<float>(),stats2.data_ptr<float>(),stats3.data_ptr<float>(),partial.data_ptr<float>(),(B*)res_z.data_ptr(),(B*)res_g.data_ptr(),n,nq,d,original,m,E,batch,views,count,schedule,lr1,lr2,hlr,momentum,wd,dropout,noise,input_scale,mix,init_gain,head_gain,init_kind,swa_fraction,dictionary_scale,data_noise,seed,capture_stream);};
  // Prime cuBLAS workspaces outside capture. All learned state is reset in run.
  if(graph){C10_CUDA_CHECK(cudaStreamSynchronize(stream));run(sourcex.data_ptr<float>(),sourcey.data_ptr<long>(),sourceq.data_ptr<float>(),1);C10_CUDA_CHECK(cudaStreamSynchronize(capture_stream));C10_CUDA_CHECK(cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal));}
  run(x.data_ptr<float>(),y.data_ptr<long>(),q.data_ptr<float>(),steps);C10_CUDA_KERNEL_LAUNCH_CHECK();if(!graph)return out;
  cudaGraph_t cg;C10_CUDA_CHECK(cudaStreamEndCapture(capture_stream,&cg));ws=new PBWorkspace;C10_CUDA_CHECK(cudaGraphInstantiate(&ws->graph,cg,nullptr,nullptr,0));C10_CUDA_CHECK(cudaGraphDestroy(cg));
  ws->tensors={x,y,q,tx,tq,w1,w2,h,u1,u2,z1,z2,du1,du2,dw1,dw2,dh,err,logits,v1,v2,vh,aug,sw1,sw2,swh,targets,centroids,w3,u3,z3,du3,dw3,v3,sw3,out,stats1,stats2,stats3,partial,res_z,res_g};pb_cache()[key.str()]=ws;
 }else ws=found->second;
 C10_CUDA_CHECK(cudaMemcpyAsync(ws->tensors[0].data_ptr(),sourcex.data_ptr(),sourcex.numel()*4,cudaMemcpyDeviceToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(ws->tensors[1].data_ptr(),sourcey.data_ptr(),sourcey.numel()*8,cudaMemcpyDeviceToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(ws->tensors[2].data_ptr(),sourceq.data_ptr(),sourceq.numel()*4,cudaMemcpyDeviceToDevice,stream));pb_last=ws;C10_CUDA_CHECK(cudaGraphLaunch(ws->graph,stream));auto result=at::empty({nq},sourcey.options());C10_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr(),ws->tensors[35].data_ptr(),nq*8,cudaMemcpyDeviceToDevice,stream));return result;
}
std::vector<at::Tensor> debug_last_state(){return pb_last->tensors;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("classify",&classify);m.def("debug_last_state",&debug_last_state);}
