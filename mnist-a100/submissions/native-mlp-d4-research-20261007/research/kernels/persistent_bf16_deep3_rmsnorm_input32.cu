#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
#include <cstdio>
#define LT_EXHAUSTIVE 0
#define PB_CLIP 0.03f
// Persistent BF16/cuBLAS control: every floating tensor except API input is BF16.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdexcept>
using B=__nv_bfloat16;
__device__ unsigned pb_hash(unsigned a){a^=a>>16;a*=0x7feb352d;a^=a>>15;a*=0x846ca68b;return a^(a>>16);}
__global__ void pb_input(const float*x,B*z,int n,int d,int original,float fixed_scale){int r=blockIdx.x,t=threadIdx.x;if(fixed_scale>=0){for(int j=t;j<d;j+=256)z[r*d+j]=__float2bfloat16_rn(j<original?x[r*original+j]*fixed_scale:0);return;}__shared__ float red[256];float v=0;for(int j=t;j<original;j+=256)v+=x[r*original+j]*x[r*original+j];red[t]=v;__syncthreads();for(int s=128;s;s>>=1){if(t<s)red[t]+=red[t+s];__syncthreads();}float scale=sqrtf(float(original)/fmaxf(red[0],1e-20f));for(int j=t;j<d;j+=256)z[r*d+j]=__float2bfloat16_rn(j<original?x[r*original+j]*scale:0);}
__global__ void pb_init(B*w,B*h,int d,int m,unsigned seed){int i=blockIdx.x*256+threadIdx.x;if(i<m*d){float a=(pb_hash(i+seed)+1.f)/4294967296.f,b=(pb_hash(i+seed+123456)+1.f)/4294967296.f;w[i]=__float2bfloat16_rn(sqrtf(-2*logf(a))*cosf(6.2831853f*b)/sqrtf(float(d)));}if(i<m*16)h[i]=__float2bfloat16_rn(0);}
// Gaussian, paired, signed Hadamard, Xavier normal, Kaiming normal, Xavier with ReLU gain.
__global__ void pb_init_weights(B*w,int d,int m,unsigned seed,float gain,int kind,int original=0){int i=blockIdx.x*256+threadIdx.x;if(i>=m*d)return;int physical=i;int row=i/d,col=i%d;if(original){if(col>=original){w[i]=__float2bfloat16_rn(0);return;}i=row*original+col;d=original;}float value;
 if(kind==2){int width=1;while(width<d)width*=2;unsigned signs=__popc((row%width)&col)+(pb_hash(col+(row/width)*100003+seed+451)&1)+(pb_hash(row+seed+917)&1);value=(signs&1)?-1.f:1.f;}
 else{int index=kind==1?(row%(m/2))*d+col:i;float a=(pb_hash(index+seed)+1.f)/4294967296.f,b=(pb_hash(index+seed+123456)+1.f)/4294967296.f;value=sqrtf(-2*logf(a))*cosf(6.2831853f*b);if(kind==1&&row>=m/2)value=-value;}
 float scaled=gain*value/sqrtf(float(d));if(kind==3||kind==5)scaled*=sqrtf((kind==5?4.f:2.f)*d/(d+m));else if(kind==4)scaled*=sqrtf(2.f);w[physical]=__float2bfloat16_rn(scaled);}
__global__ void pb_init_head(B*h,int m,unsigned seed,float gain){int i=blockIdx.x*256+threadIdx.x;if(i>=m*16)return;int logical=i/16*10+i%16;float a=(pb_hash(logical+seed)+1.f)/4294967296.f,b=(pb_hash(logical+seed+123456)+1.f)/4294967296.f;h[i]=__float2bfloat16_rn(i%16<10?gain*sqrtf(-2*logf(a))*cosf(6.2831853f*b)/sqrtf(float(m)):0.f);}
__global__ void pb_act(const B*u,B*z,int m,int batch,float dropout=0,unsigned seed=0){int r=blockIdx.x,t=threadIdx.x;__shared__ float red[5];float sum=0;for(int j=t;j<m;j+=128)sum+=__bfloat162float(u[r*m+j]);for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);if((t&31)==0)red[t>>5]=sum;__syncthreads();if(t==0){float a=0;for(int i=0;i<4;i++)a+=red[i];red[4]=a/m;}__syncthreads();for(int j=t;j<m;j+=128){B rounded=__float2bfloat16_rn(fmaxf(__bfloat162float(u[r*m+j])-red[4],0));if(dropout>0){float random=(pb_hash((r%batch)*m+j+seed+(r/batch)*101)+1.f)/4294967296.f;rounded=__float2bfloat16_rn(random<dropout?0:__bfloat162float(rounded)/(1-dropout));}z[r*m+j]=rounded;}}
__global__ void pb_back(B*g,const B*z,int m,float gain){int r=blockIdx.x,t=threadIdx.x;__shared__ float red[5];float sum=0;for(int j=t;j<m;j+=128)sum+=__bfloat162float(z[r*m+j])>0?gain*__bfloat162float(g[r*m+j]):0;for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);if((t&31)==0)red[t>>5]=sum;__syncthreads();if(t==0){float a=0;for(int i=0;i<4;i++)a+=red[i];red[4]=a/m;}__syncthreads();for(int j=t;j<m;j+=128)g[r*m+j]=__float2bfloat16_rn((__bfloat162float(z[r*m+j])>0?gain*__bfloat162float(g[r*m+j]):0)-red[4]);}
template<int C>struct Pack;template<>struct Pack<8>{union{uint4 vec;B x[8];};};template<>struct Pack<4>{union{uint2 vec;B x[4];};};
template<int M,int C>__global__ void vec_act(const B*u,B*z,int rows,int batch,float dropout,unsigned seed){int r=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(r>=rows)return;constexpr int R=M/(32*C);Pack<C> v[R];float sum=0;
#pragma unroll
 for(int k=0;k<R;k++){v[k].vec=*reinterpret_cast<const decltype(v[k].vec)*>(u+r*M+k*32*C+lane*C);
#pragma unroll
 for(int j=0;j<C;j++)sum+=__bfloat162float(v[k].x[j]);}
 for(int shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffff,sum,shift);float mean=__shfl_sync(0xffffffff,sum,0)/M;
#pragma unroll
 for(int k=0;k<R;k++){
#pragma unroll
 for(int j=0;j<C;j++){int col=k*32*C+lane*C+j;B rounded=__float2bfloat16_rn(fmaxf(__bfloat162float(v[k].x[j])-mean,0));if(dropout>0){float random=(pb_hash((r%batch)*M+col+seed+(r/batch)*101)+1.f)/4294967296.f;rounded=__float2bfloat16_rn(random<dropout?0:__bfloat162float(rounded)/(1-dropout));}v[k].x[j]=rounded;}*reinterpret_cast<decltype(v[k].vec)*>(z+r*M+k*32*C+lane*C)=v[k].vec;}}
template<int M,int C>__global__ void vec_back(B*g,const B*z,int rows,float gain){int r=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(r>=rows)return;constexpr int R=M/(32*C);float values[R*C];float sum=0;
#pragma unroll
 for(int k=0;k<R;k++){Pack<C> a,b;a.vec=*reinterpret_cast<const decltype(a.vec)*>(g+r*M+k*32*C+lane*C);b.vec=*reinterpret_cast<const decltype(b.vec)*>(z+r*M+k*32*C+lane*C);
#pragma unroll
 for(int j=0;j<C;j++){float v=__bfloat162float(b.x[j])>0?gain*__bfloat162float(a.x[j]):0;values[k*C+j]=v;sum+=v;}}
 for(int shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffff,sum,shift);float mean=__shfl_sync(0xffffffff,sum,0)/M;
#pragma unroll
 for(int k=0;k<R;k++){Pack<C> v;
#pragma unroll
 for(int j=0;j<C;j++)v.x[j]=__float2bfloat16_rn(values[k*C+j]-mean);*reinterpret_cast<decltype(v.vec)*>(g+r*M+k*32*C+lane*C)=v.vec;}}
void act(const B*u,B*z,int rows,int m,int batch,float p,unsigned seed,cudaStream_t s){switch(m){case 128:vec_act<128,4><<<(rows+3)/4,128,0,s>>>(u,z,rows,batch,p,seed);break;case 256:vec_act<256,8><<<(rows+3)/4,128,0,s>>>(u,z,rows,batch,p,seed);break;case 512:vec_act<512,8><<<(rows+3)/4,128,0,s>>>(u,z,rows,batch,p,seed);break;default:pb_act<<<rows,128,0,s>>>(u,z,m,batch,p,seed);}}
void back(B*g,const B*z,int rows,int m,float gain,cudaStream_t s){switch(m){case 128:vec_back<128,4><<<(rows+3)/4,128,0,s>>>(g,z,rows,gain);break;case 256:vec_back<256,8><<<(rows+3)/4,128,0,s>>>(g,z,rows,gain);break;case 512:vec_back<512,8><<<(rows+3)/4,128,0,s>>>(g,z,rows,gain);break;default:pb_back<<<rows,128,0,s>>>(g,z,m,gain);}}
__global__ void pb_error(B*e,const long*y,int n,int off,int total,float mix,unsigned seed,int E,int base,float smoothing){int r=blockIdx.x*128+threadIdx.x;if(r>=n)return;int batch=n/E,local=r%batch;seed+=(r/batch)*101;int partner=pb_hash(local+seed)%total;float a[10],mx=-INFINITY,sum=0;for(int j=0;j<10;j++){a[j]=__bfloat162float(e[r*16+j]);mx=fmaxf(mx,a[j]);}for(int j=0;j<10;j++)sum+=expf(a[j]-mx);for(int j=0;j<10;j++)e[r*16+j]=__float2bfloat16_rn(expf(a[j]-mx)/sum-((1-smoothing)*((1-mix)*(y[(off+local%base)%total]==j)+mix*(y[partner]==j))+smoothing*.1f));for(int j=10;j<16;j++)e[r*16+j]=__float2bfloat16_rn(0);}
__global__ void pb_mix_input(const B*x,B*out,int batch,int d,int off,int total,float mix,unsigned seed){int i=blockIdx.x*256+threadIdx.x;if(i>=batch*d)return;int r=i/d,c=i%d,partner=pb_hash(r+seed)%total;out[i]=__float2bfloat16_rn((1-mix)*__bfloat162float(x[(off+r)*d+c])+mix*__bfloat162float(x[partner*d+c]));}
__global__ void pb_update(B*w,const B*g,B*v,int size,int batch,float lr,float momentum,float wd){int i=blockIdx.x*256+threadIdx.x;if(i>=size)return;float wi=__bfloat162float(w[i]),grad=fminf(PB_CLIP,fmaxf(-PB_CLIP,__bfloat162float(g[i])/batch))+wd*wi;B vel=__float2bfloat16_rn(momentum*__bfloat162float(v[i])+grad);v[i]=vel;w[i]=__float2bfloat16_rn(wi-lr*(grad+momentum*__bfloat162float(vel)));}
// One launch updates all parameter groups; each thread reads/writes two BF16 values.
__global__ void pb_update_all(B*w1,B*w2,B*h,const B*g1,const B*g2,const B*gh,B*v1,B*v2,B*vh,int d,int m,int E,int batch,float lr1,float lr2,float hlr,float momentum,float wd,bool nesterov,float*sw1,float*sw2,float*swh,B*w3,B*w4,const B*g3,const B*g4,B*v3,B*v4,float*sw3,float*sw4,bool collect){
 int pair=blockIdx.x*256+threadIdx.x,i=pair*2,total=E*(m*d+2*m*m+m*16);if(i>=total)return;
 B*w,*v;const B*g;float lr;float*sw;
 if(i<E*m*d){w=w1;g=g1;v=v1;lr=lr1;sw=sw1;}
 else if(i<E*(m*d+m*m)){i-=E*m*d;w=w2;g=g2;v=v2;lr=lr2;sw=sw2;}
 else if(i<E*(m*d+m*m+m*16)){i-=E*(m*d+m*m);w=h;g=gh;v=vh;lr=hlr*256/m;sw=swh;}
 else{i-=E*(m*d+m*m+m*16);w=w3;g=g3;v=v3;lr=lr2;sw=sw3;}
 float2 weights=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w+i));
 float2 grads=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(g+i));
 float2 velocities=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v+i));
 float gx=fminf(PB_CLIP,fmaxf(-PB_CLIP,grads.x/batch))+wd*weights.x,gy=fminf(PB_CLIP,fmaxf(-PB_CLIP,grads.y/batch))+wd*weights.y;
 B vx=__float2bfloat16_rn(momentum*velocities.x+gx),vy=__float2bfloat16_rn(momentum*velocities.y+gy);
 *reinterpret_cast<__nv_bfloat162*>(v+i)=__halves2bfloat162(vx,vy);
 B wx=__float2bfloat16_rn(weights.x-lr*(nesterov?gx+momentum*__bfloat162float(vx):__bfloat162float(vx)));
 B wy=__float2bfloat16_rn(weights.y-lr*(nesterov?gy+momentum*__bfloat162float(vy):__bfloat162float(vy)));
 *reinterpret_cast<__nv_bfloat162*>(w+i)=__halves2bfloat162(wx,wy);if(collect){sw[i]+=__bfloat162float(wx);sw[i+1]+=__bfloat162float(wy);}
}
__global__ void pb_dropout(B*z,int count,float p,unsigned seed){int i=blockIdx.x*256+threadIdx.x;if(i<count){float u=(pb_hash(i+seed)+1.f)/4294967296.f;z[i]=__float2bfloat16_rn(u<p?0:__bfloat162float(z[i])/(1-p));}}
__global__ void pb_noise(const B*x,B*aug,int count,float sd,unsigned seed){int i=blockIdx.x*256+threadIdx.x;if(i<count){float a=(pb_hash(i+seed)+1.f)/4294967296.f,b=(pb_hash(i+seed+123456)+1.f)/4294967296.f;float noise=sqrtf(-2*logf(a))*cosf(6.2831853f*b);aug[i]=__float2bfloat16_rn(__bfloat162float(x[i])+sd*noise);}}
// Preserve the intermediate BF16 rounding while eliminating the mix/noise buffer round trip.
__global__ void pb_mix_noise(const B*x,B*out,int batch,int d,int off,int total,float mix,float sd,unsigned mix_seed,unsigned noise_seed){int i=blockIdx.x*256+threadIdx.x;if(i>=batch*d)return;int r=i/d,c=i%d,partner=pb_hash(r+mix_seed)%total;B mixed=__float2bfloat16_rn((1-mix)*__bfloat162float(x[(off+r)*d+c])+mix*__bfloat162float(x[partner*d+c]));float a=(pb_hash(i+noise_seed)+1.f)/4294967296.f,b=(pb_hash(i+noise_seed+123456)+1.f)/4294967296.f;float noise=sqrtf(-2*logf(a))*cosf(6.2831853f*b);out[i]=__float2bfloat16_rn(__bfloat162float(mixed)+sd*noise);}
__global__ void pb_predict(const B*e,long*out,int n){int r=blockIdx.x*128+threadIdx.x;if(r>=n)return;int best=0;for(int j=1;j<10;j++)if(__bfloat162float(e[r*16+j])>__bfloat162float(e[r*16+best]))best=j;out[r]=best;}
static thread_local cublasHandle_t pb_handle=nullptr;
void prepare_pb_handle(cudaStream_t s){if(!pb_handle&&cublasCreate(&pb_handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BF16 state cublasCreate failed");if(cublasSetStream(pb_handle,s)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BF16 state set stream failed");}
static thread_local cublasLtHandle_t lt_handle=nullptr;
static thread_local void*lt_workspace=nullptr;
constexpr size_t LT_WORKSPACE=4*1024*1024;
struct LtPlan{cublasLtMatmulDesc_t desc;cublasLtMatrixLayout_t a,b,c;cublasLtMatmulAlgo_t algo;};
using LtKey=std::tuple<int,int,int,bool,bool,int,bool,bool>;
static thread_local std::map<LtKey,LtPlan> lt_plans;
static void lt_ok(cublasStatus_t st){if(st!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLASLt tuned operation failed");}
static void mm(const B*a,const B*b,B*out,int m,int n,int k,bool at,bool bt,int E,bool shared_a=false,bool shared_b=false){
 cudaStream_t stream;lt_ok(cublasGetStream(pb_handle,&stream));
 if(!lt_handle){lt_ok(cublasLtCreate(&lt_handle));if(cudaMalloc(&lt_workspace,LT_WORKSPACE)!=cudaSuccess)throw std::runtime_error("LT workspace allocation failed");}
 LtKey key={m,n,k,at,bt,E,shared_a,shared_b};auto found=lt_plans.find(key);
 if(found==lt_plans.end()){
  LtPlan p;lt_ok(cublasLtMatmulDescCreate(&p.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t ta=bt?CUBLAS_OP_N:CUBLAS_OP_T,tb=at?CUBLAS_OP_T:CUBLAS_OP_N;
  lt_ok(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));lt_ok(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  lt_ok(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_16BF,bt?n:k,bt?k:n,bt?n:k));
  lt_ok(cublasLtMatrixLayoutCreate(&p.b,CUDA_R_16BF,at?m:k,at?k:m,at?m:k));
  lt_ok(cublasLtMatrixLayoutCreate(&p.c,CUDA_R_16BF,n,m,n));
  long long sa=shared_b?0:((long long)n*k),sb=shared_a?0:((long long)m*k),sc=(long long)m*n;
  if(E>1){for(auto layout:{p.a,p.b,p.c})lt_ok(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&E,sizeof(E)));lt_ok(cublasLtMatrixLayoutSetAttribute(p.a,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sa,sizeof(sa)));lt_ok(cublasLtMatrixLayoutSetAttribute(p.b,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sb,sizeof(sb)));lt_ok(cublasLtMatrixLayoutSetAttribute(p.c,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sc,sizeof(sc)));}
  cublasLtMatmulPreference_t pref;lt_ok(cublasLtMatmulPreferenceCreate(&pref));size_t bytes=LT_WORKSPACE;lt_ok(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  cublasLtMatmulHeuristicResult_t initial[32];int count=0;lt_ok(cublasLtMatmulAlgoGetHeuristic(lt_handle,p.desc,p.a,p.b,p.c,p.c,pref,32,initial,&count));cublasLtMatmulPreferenceDestroy(pref);
  std::vector<cublasLtMatmulHeuristicResult_t> choices(initial,initial+count);
  cudaStreamCaptureStatus search_capture;cudaStreamIsCapturing(stream,&search_capture);
  if(LT_EXHAUSTIVE&&search_capture==cudaStreamCaptureStatusNone&&m>=128&&m<=4096&&n>=128&&k>=64){
   int ids[256],num_ids=0;lt_ok(cublasLtMatmulAlgoGetIds(lt_handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,256,ids,&num_ids));
   for(int index=0;index<num_ids;index++){
    cublasLtMatmulAlgo_t original_algo;if(cublasLtMatmulAlgoInit(lt_handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,ids[index],&original_algo)!=CUBLAS_STATUS_SUCCESS)continue;
    size_t bytes=0;cublasLtMatmulAlgoCapGetAttribute(&original_algo,CUBLASLT_ALGO_CAP_TILE_IDS,nullptr,0,&bytes);std::vector<int> tiles(bytes/sizeof(int));if(bytes)cublasLtMatmulAlgoCapGetAttribute(&original_algo,CUBLASLT_ALGO_CAP_TILE_IDS,tiles.data(),bytes,&bytes);
    bytes=0;cublasLtMatmulAlgoCapGetAttribute(&original_algo,CUBLASLT_ALGO_CAP_STAGES_IDS,nullptr,0,&bytes);std::vector<int> stages(bytes/sizeof(int));if(bytes)cublasLtMatmulAlgoCapGetAttribute(&original_algo,CUBLASLT_ALGO_CAP_STAGES_IDS,stages.data(),bytes,&bytes);if(stages.empty())stages.push_back(0);
    for(int tile:tiles){if(tile<5||tile>18)continue;for(int stage:stages){for(int split:{1,2,4,8,16}){
     if(split>1&&k<512)continue;auto algo=original_algo;
     if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS)continue;
     if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stage,sizeof(stage))!=CUBLAS_STATUS_SUCCESS)continue;
     if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
     uint32_t reduction=split>1?CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE:CUBLASLT_REDUCTION_SCHEME_NONE;
     if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction))!=CUBLAS_STATUS_SUCCESS)continue;
     cublasLtMatmulHeuristicResult_t check;if(cublasLtMatmulAlgoCheck(lt_handle,p.desc,p.a,p.b,p.c,p.c,&algo,&check)!=CUBLAS_STATUS_SUCCESS||check.state!=CUBLAS_STATUS_SUCCESS||check.workspaceSize>LT_WORKSPACE)continue;
     check.algo=algo;choices.push_back(check);
    }}}
   }
   count=choices.size();
  }

  if(!count)throw std::runtime_error("LT no heuristic algorithms");p.algo=choices[0].algo;
  cudaStreamCaptureStatus capture;cudaStreamIsCapturing(stream,&capture);
  float alpha=1,beta=0,best=1e30f;int winner=0;
  if(capture==cudaStreamCaptureStatusNone){
   cudaStreamSynchronize(stream);cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
   for(int i=0;i<count;i++){
    if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    if(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)throw std::runtime_error("LT tuning capture failed");
    cublasStatus_t st=CUBLAS_STATUS_SUCCESS;for(int repeat=0;repeat<16;repeat++){st=cublasLtMatmul(lt_handle,p.desc,&alpha,b,p.a,a,p.b,&beta,out,p.c,out,p.c,&choices[i].algo,lt_workspace,LT_WORKSPACE,stream);if(st!=CUBLAS_STATUS_SUCCESS)break;}
    cudaError_t ce=cudaStreamEndCapture(stream,&graph);
    if(st!=CUBLAS_STATUS_SUCCESS||ce!=cudaSuccess||!graph){if(graph)cudaGraphDestroy(graph);continue;}
    if(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)!=cudaSuccess){cudaGraphDestroy(graph);continue;}
    cudaGraphLaunch(exec,stream);cudaStreamSynchronize(stream);cudaEventRecord(begin,stream);
    for(int repeat=0;repeat<5;repeat++)cudaGraphLaunch(exec,stream);
    cudaEventRecord(end,stream);cudaEventSynchronize(end);float elapsed;cudaEventElapsedTime(&elapsed,begin,end);
    if(elapsed<best){best=elapsed;winner=i;p.algo=choices[i].algo;}
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
   }
   cudaEventDestroy(begin);cudaEventDestroy(end);
   int id=-1;size_t written;cublasLtMatmulAlgoConfigGetAttribute(&p.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written);
   int tile=-1,split=-1;cublasLtMatmulAlgoConfigGetAttribute(&p.algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written);cublasLtMatmulAlgoConfigGetAttribute(&p.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written);
   fprintf(stderr,"LT tuned m=%d n=%d k=%d at=%d bt=%d E=%d candidates=%d winner=%d id=%d tile=%d split=%d us=%.4f\n",m,n,k,at,bt,E,count,winner,id,tile,split,best*1000/80);
  }
  found=lt_plans.emplace(key,p).first;
 }
 auto&p=found->second;float alpha=1,beta=0;lt_ok(cublasLtMatmul(lt_handle,p.desc,&alpha,b,p.a,a,p.b,&beta,out,p.c,out,p.c,&p.algo,lt_workspace,LT_WORKSPACE,stream));
}

__global__ void ensemble_input(const B*x,B*out,int batch,int d,int off,int total,float mix,float sd,unsigned seed,int E,int base,int original){int i=blockIdx.x*256+threadIdx.x;if(i>=E*batch*d)return;int member=i/(batch*d),local=i%(batch*d),r=local/d,c=local%d;if(c>=original+10){out[i]=__float2bfloat16_rn(0);return;}int noise_index=r*original+c;unsigned ms=seed+member*101+999,ns=seed+member*101+777;int partner=pb_hash(r+ms)%total;B v=x[((off+r%base)%total)*d+c];if(mix>0)v=__float2bfloat16_rn((1-mix)*__bfloat162float(v)+mix*__bfloat162float(x[partner*d+c]));if(sd>0&&c<original){float a=(pb_hash(noise_index+ns)+1.f)/4294967296.f,b=(pb_hash(noise_index+ns+123456)+1.f)/4294967296.f;v=__float2bfloat16_rn(__bfloat162float(v)+sd*sqrtf(-2*logf(a))*cosf(6.2831853f*b));}out[i]=v;}
__global__ void ensemble_predict(const B*logits,long*out,int n,int E){int r=blockIdx.x*128+threadIdx.x;if(r>=n)return;int best=0;float bestv=-INFINITY;for(int c=0;c<10;c++){float sum=0;for(int member=0;member<E;member++)sum+=__bfloat162float(logits[(member*n+r)*16+c]);if(sum>bestv){bestv=sum;best=c;}}out[r]=best;}
__global__ void swa_finalize(B*w1,B*w2,B*h,const float*s1,const float*s2,const float*sh,int d,int m,int E,int count){int i=blockIdx.x*256+threadIdx.x;if(i>=E*(m*d+m*m+m*16))return;B*w;const float*s;if(i<E*m*d){w=w1;s=s1;}else if(i<E*(m*d+m*m)){i-=E*m*d;w=w2;s=s2;}else{i-=E*(m*d+m*m);w=h;s=sh;}w[i]=__float2bfloat16_rn(s[i]/count);}
__global__ void dictionary_targets(const long*y,B*targets,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n*10)targets[i]=__float2bfloat16_rn(y[i/10]==i%10?1.f:0.f);}
__global__ void dictionary_normalize(B*cent,int d,int original){int label=blockIdx.x,lane=threadIdx.x;float sum=0;for(int j=lane;j<original;j+=32){float v=__bfloat162float(cent[label*d+j]);sum+=v*v;}for(int shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffff,sum,shift);float norm=sqrtf(__shfl_sync(0xffffffff,sum,0)+1e-20f);for(int j=lane;j<d;j+=32)cent[label*d+j]=__float2bfloat16_rn(j<original?__bfloat162float(cent[label*d+j])/norm:0);}
__global__ void dictionary_append(B*x,const B*cent,int rows,int d,int original,float scale){int r=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(r>=rows)return;for(int label=0;label<10;label++){float sum=0;for(int j=lane;j<original;j+=32)sum+=__bfloat162float(x[r*d+j])*__bfloat162float(cent[label*d+j]);for(int shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffff,sum,shift);if(lane==0)x[r*d+original+label]=__float2bfloat16_rn(scale*sum);}}
__global__ void dictionary_init(B*w,const B*cent,const B*x,int original,int d,int m,int n,unsigned seed,int kind,float jitter,float gain){int i=blockIdx.x*256+threadIdx.x;if(i>=m*d)return;int row=i/d,col=i%d;if(col>=original){w[i]=__float2bfloat16_rn(0);return;}float a=(pb_hash(row*original+col+seed)+1.f)/4294967296.f,b=(pb_hash(row*original+col+seed+123456)+1.f)/4294967296.f;float random=sqrtf(-2*logf(a))*cosf(6.2831853f*b)/sqrtf(float(original));float data=kind==6?__bfloat162float(cent[(row%10)*d+col]):__bfloat162float(x[(pb_hash(row+seed)%n)*d+col])/sqrtf(float(original));w[i]=__float2bfloat16_rn(gain*(sqrtf(fmaxf(0,1-jitter*jitter))*data+jitter*random));}
__global__ void update_extra(B*w,const B*g,B*v,float*sw,int count,int batch,float lr,float momentum,float wd,bool nesterov,bool collect){int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;float wi=__bfloat162float(w[i]),grad=fminf(PB_CLIP,fmaxf(-PB_CLIP,__bfloat162float(g[i])/batch))+wd*wi;B vel=__float2bfloat16_rn(momentum*__bfloat162float(v[i])+grad);v[i]=vel;B value=__float2bfloat16_rn(wi-lr*(nesterov?grad+momentum*__bfloat162float(vel):__bfloat162float(vel)));w[i]=value;if(collect)sw[i]+=__bfloat162float(value);}
__global__ void finalize_extra(B*w,const float*sw,int count,int num){int i=blockIdx.x*256+threadIdx.x;if(i<count)w[i]=__float2bfloat16_rn(sw[i]/num);}
#include "mlp_rmsnorm_maskregen.cuh"
void input_rms_fused(const B*,const B*,B*,B*,int,int,int,int,float,unsigned,cudaStream_t);
void input_rms_tile32(const B*,const B*,B*,B*,int,int,int,int,float,unsigned,cudaStream_t);
void launch_pb(const float*x,const long*y,const float*q,long*out,B*tx,B*tq,B*w1,B*w2,B*h,B*u1,B*u2,B*z1,B*z2,B*du1,B*du2,B*dw1,B*dw2,B*dh,B*err,B*logits,B*v1,B*v2,B*vh,B*aug,B*targets,B*centroids,float*sw1,float*sw2,float*swh,B*w3,B*u3,B*z3,B*du3,B*dw3,B*v3,float*sw3,B*w4,B*u4,B*z4,B*du4,B*dw4,B*v4,float*sw4,float*stats1,float*stats2,float*stats3,float*stats4,int n,int nq,int d,int original,int m,int E,int batch,int views,int steps,int schedule,float lr1,float lr2,float hlr,float momentum,float wd,float dropout,float noise,float input_scale,float mix,float init_gain,float head_gain,int init_kind,float swa_fraction,float dictionary_scale,float data_noise,float smoothing,unsigned seed,cudaStream_t s){
 cudaMemsetAsync(w4,0,E*m*m*2,s);int swa_count=swa_fraction>0?max(1,int(steps*swa_fraction)):0;bool nesterov=schedule<3;schedule%=3;prepare_pb_handle(s);pb_input<<<n,256,0,s>>>(x,tx,n,d,original,input_scale);pb_input<<<nq,256,0,s>>>(q,tq,nq,d,original,input_scale);if(dictionary_scale>0||init_kind==6){dictionary_targets<<<(n*10+255)/256,256,0,s>>>(y,targets,n);mm(targets,tx,centroids,10,d,n,true,true,1);dictionary_normalize<<<10,32,0,s>>>(centroids,d,original);}else{cudaMemsetAsync(targets,0,n*10*2,s);cudaMemsetAsync(centroids,0,10*d*2,s);}if(dictionary_scale>0){dictionary_append<<<(n+3)/4,128,0,s>>>(tx,centroids,n,d,original,dictionary_scale);dictionary_append<<<(nq+3)/4,128,0,s>>>(tq,centroids,nq,d,original,dictionary_scale);}for(int member=0;member<E;member++){unsigned es=seed+member*101;if(init_kind>=6)dictionary_init<<<(m*d+255)/256,256,0,s>>>(w1+member*m*d,centroids,tx,original,d,m,n,es,init_kind,data_noise,init_gain);else pb_init_weights<<<(m*d+255)/256,256,0,s>>>(w1+member*m*d,d,m,es,init_gain,init_kind,original+(dictionary_scale>0?10:0));pb_init_weights<<<(m*m+255)/256,256,0,s>>>(w2+member*m*m,m,m,es+123123,init_gain,init_kind);pb_init_weights<<<(m*m+255)/256,256,0,s>>>(w3+member*m*m,m,m,es+789789,init_gain,init_kind);pb_init_head<<<(m*16+255)/256,256,0,s>>>(h+member*m*16,m,es+456456,head_gain);}cudaMemsetAsync(v1,0,E*m*d*2,s);cudaMemsetAsync(v2,0,E*m*m*2,s);cudaMemsetAsync(vh,0,E*m*16*2,s);cudaMemsetAsync(v3,0,E*m*m*2,s);cudaMemsetAsync(v4,0,E*m*m*2,s);if(swa_count){cudaMemsetAsync(sw1,0,E*m*d*4,s);cudaMemsetAsync(sw2,0,E*m*m*4,s);cudaMemsetAsync(swh,0,E*m*16*4,s);cudaMemsetAsync(sw3,0,E*m*m*4,s);cudaMemsetAsync(sw4,0,E*m*m*4,s);}
 for(int t=0;t<steps;t++){float decay=schedule==1?(.05f+.95f*.5f*(1+cosf(3.141592653589793f*t/fmaxf(steps-1,1)))):(schedule==2?(t*100>=steps*85?.01f:(t*100>=steps*60?.1f:1.f)):1.f);int off=(t*batch)%n,base=min(batch,n),b=base*views;const B*xb=aug;ensemble_input<<<(E*b*d+255)/256,256,0,s>>>(tx,aug,b,d,off,n,mix,noise,seed+t*100003,E,base,original);
 if(d==64&&m==512&&b%32==0)input_rms_tile32(xb,w1,u1,z1,E,b,d,m,dropout,seed+t*100003+111,s);else if(d==64&&(m==256||m==512)&&b%16==0)input_rms_fused(xb,w1,u1,z1,E,b,d,m,dropout,seed+t*100003+111,s);else{mm(xb,w1,u1,b,m,d,false,false,E);mlp_bn16::act(u1,z1,E*b,m,b,dropout,seed+t*100003+111,stats1,s);}mm(z1,w2,u2,b,m,m,false,false,E);mlp_bn16::act(u2,z2,E*b,m,b,dropout,seed+t*100003+333,stats2,s);mm(z2,w3,u3,b,m,m,false,false,E);mlp_bn16::act(u3,z3,E*b,m,b,dropout,seed+t*100003+555,stats3,s);mm(z3,h,err,b,16,m,false,true,E);pb_error<<<(E*b+127)/128,128,0,s>>>(err,y,E*b,off,n,mix,seed+t*100003+999,E,base,smoothing);
 mm(z3,err,dh,m,16,b,true,true,E);mm(err,h,du3,b,m,16,false,false,E);mlp_bn16::back(du3,z3,u3,E*b,m,b,1/(1-dropout),stats3,s,dropout,seed+t*100003+555);mm(du3,z2,dw3,m,m,b,true,true,E);mm(du3,w3,du2,b,m,m,false,true,E);mlp_bn16::back(du2,z2,u2,E*b,m,b,1/(1-dropout),stats2,s,dropout,seed+t*100003+333);mm(du2,z1,dw2,m,m,b,true,true,E);mm(du2,w2,du1,b,m,m,false,true,E);mlp_bn16::back(du1,z1,u1,E*b,m,b,1/(1-dropout),stats1,s,dropout,seed+t*100003+111);mm(du1,xb,dw1,m,d,b,true,true,E);
 pb_update_all<<<(E*(m*d+2*m*m+m*16)/2+255)/256,256,0,s>>>(w1,w2,h,dw1,dw2,dh,v1,v2,vh,d,m,E,b,decay*lr1,decay*lr2,decay*hlr,momentum,wd,nesterov,sw1,sw2,swh,w3,w4,dw3,dw4,v3,v4,sw3,sw4,swa_count&&t>=steps-swa_count);
 }
 if(swa_count&&steps>0)swa_finalize<<<(E*(m*d+m*m+m*16)+255)/256,256,0,s>>>(w1,w2,h,sw1,sw2,swh,d,m,E,swa_count);
 if(swa_count&&steps>0)finalize_extra<<<(E*m*m+255)/256,256,0,s>>>(w3,sw3,E*m*m,swa_count);
 mm(tq,w1,u1,nq,m,d,false,false,E,true);mlp_bn16::act(u1,z1,E*nq,m,nq,0,0,stats1,s);mm(z1,w2,u2,nq,m,m,false,false,E);mlp_bn16::act(u2,z2,E*nq,m,nq,0,0,stats2,s);mm(z2,w3,u3,nq,m,m,false,false,E);mlp_bn16::act(u3,z3,E*nq,m,nq,0,0,stats3,s);mm(z3,h,logits,nq,16,m,false,true,E);ensemble_predict<<<(nq+127)/128,128,0,s>>>(logits,out,nq,E);
}

void launch_vector_check(const B*u,B*z,B*g,int rows,int m,int batch,float p,unsigned seed,float*stats,cudaStream_t s){mlp_bn16::act(u,z,rows,m,batch,p,seed,stats,s);mlp_bn16::back(g,z,u,rows,m,batch,1/(1-p),stats,s,p,seed);}

void head16_check(const B*z,const B*h,B*logits,B*err,B*dh,B*du,const long*y,int b,int m,int E,int base,int off,int n,float mix,float smoothing,unsigned seed,cudaStream_t s){prepare_pb_handle(s);mm(z,h,logits,b,16,m,false,true,E);cudaMemcpyAsync(err,logits,E*b*16*2,cudaMemcpyDeviceToDevice,s);pb_error<<<(E*b+127)/128,128,0,s>>>(err,y,E*b,off,n,mix,seed,E,base,smoothing);mm(z,err,dh,m,16,b,true,true,E);mm(err,h,du,b,m,16,false,false,E);}
