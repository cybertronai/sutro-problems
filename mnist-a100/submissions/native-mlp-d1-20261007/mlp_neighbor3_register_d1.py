"""Native BF16 MLP with three-neighbor readout."""
CPP=r'''#define P data_ptr
#define EL t::empty_like
#define EE t::empty
#define CC C10_CUDA_CHECK
#include<torch/extension.h>
namespace t=torch;
#include<ATen/cuda/CUDAContext.h>
#include<c10/cuda/CUDAGuard.h>
#include<c10/cuda/CUDAException.h>
#include<cuda_runtime.h>
#include<cuda_bf16.h>
using B=__nv_bfloat16;
void handle(cudaStream_t);
void LP(const float*x,const long*y,const float*q,long*out,B*tx,B*tq,B*w1,B*w2,B*h,B*u1,B*u2,B*z1,B*z2,B*du1,B*du2,B*dw1,B*dw2,B*dh,B*err,B*lg,B*v1,B*v2,B*vh,B*aug,B*tg,B*ct,float*sw1,float*sw2,float*swh,float*hp,int n,int nq,int steps,cudaStream_t s);
#include<map>
#include<sstream>
struct WS{cudaGraphExec_t graph;std::vector<t::Tensor>tensors;};
static thread_local WS*last=nullptr;
static std::map<std::string,WS*>&cache(){static std::map<std::string,WS*>c;return c;}
t::Tensor core(t::Tensor x,t::Tensor y,t::Tensor q){
constexpr int m=256,E=2,bs=1024,views=1,steps=30;
constexpr bool graph=true;
c10::cuda::CUDAGuard device(x.device());int n=x.size(0),nq=q.size(0),od=x.size(1),d=((od+10+63)/64)*64,v=std::max(std::max(n,nq),std::min(bs,n)*views);auto sx=x,sy=y,sq=q;auto s2=at::cuda::getCurrentCUDAStream();
std::ostringstream key;key<<x.get_device()<<":"<<(uintptr_t)(cudaStream_t)s2<<":"<<n<<":"<<nq<<":"<<d;
auto f2=cache().find(key.str());WS*ws=nullptr;
if(!graph||f2==cache().end()){
if(graph){x=EL(sx);y=EL(sy);q=EL(sq);}
auto o0=x.options().dtype(t::kBFloat16);
auto tx=EE({n,d},o0),tq=EE({nq,d},o0),w1=EE({E*m,d},o0),w2=EE({E*m,m},o0),h=EE({E*m,16},o0),u1=EE({E*v,m},o0),u2=EL(u1),z1=EL(u1),z2=EL(u1),du1=EL(u1),du2=EL(u1),dw1=EL(w1),dw2=EL(w2),dh=EL(h),err=EE({E*v,16},o0),lg=EE({E*nq,16},o0),v1=EL(w1),v2=EL(w2),vh=EL(h),aug=EE({E*std::min(bs,n)*views,d},o0),out=EE({nq},y.options());
auto sw1=EL(w1,x.options()),sw2=EL(w2,x.options()),swh=EL(h,x.options());
auto hp=EE({4,E*m,16},x.options());
auto tg=EE({n,10},o0),ct=EE({10,d},o0);
auto cs=graph?at::cuda::getStreamFromPool(false,x.get_device()):s2;c10::cuda::CUDAStreamGuard guard(cs);
auto run=[&](const float*xx,const long*yy,const float*qq,int count){LP(xx,yy,qq,out.P<long>(),(B*)tx.P(),(B*)tq.P(),(B*)w1.P(),(B*)w2.P(),(B*)h.P(),(B*)u1.P(),(B*)u2.P(),(B*)z1.P(),(B*)z2.P(),(B*)du1.P(),(B*)du2.P(),(B*)dw1.P(),(B*)dw2.P(),(B*)dh.P(),(B*)err.P(),(B*)lg.P(),(B*)v1.P(),(B*)v2.P(),(B*)vh.P(),(B*)aug.P(),(B*)tg.P(),(B*)ct.P(),sw1.P<float>(),sw2.P<float>(),swh.P<float>(),hp.P<float>(),n,nq,count,cs);};
if(graph){CC(cudaStreamSynchronize(s2));run(sx.P<float>(),sy.P<long>(),sq.P<float>(),1);CC(cudaStreamSynchronize(cs));CC(cudaStreamBeginCapture(cs,cudaStreamCaptureModeThreadLocal));}
run(x.P<float>(),y.P<long>(),q.P<float>(),steps);C10_CUDA_KERNEL_LAUNCH_CHECK();if(!graph)return out;
cudaGraph_t cg;CC(cudaStreamEndCapture(cs,&cg));ws=new WS;CC(cudaGraphInstantiate(&ws->graph,cg,nullptr,nullptr,0));CC(cudaGraphDestroy(cg));
ws->tensors={x,y,q,tx,tq,w1,w2,h,u1,u2,z1,z2,du1,du2,dw1,dw2,dh,err,lg,v1,v2,vh,aug,sw1,sw2,swh,tg,ct,hp,out};cache()[key.str()]=ws;
}else ws=f2->second;
CC(cudaMemcpyAsync(ws->tensors[0].P(),sx.P(),sx.numel()*4,cudaMemcpyDeviceToDevice,s2));CC(cudaMemcpyAsync(ws->tensors[1].P(),sy.P(),sy.numel()*8,cudaMemcpyDeviceToDevice,s2));CC(cudaMemcpyAsync(ws->tensors[2].P(),sq.P(),sq.numel()*4,cudaMemcpyDeviceToDevice,s2));last=ws;CC(cudaGraphLaunch(ws->graph,s2));return{};
}
void neighbor(const B*,const long*,B*,float*,float*,int,int,int,int,int,float,cudaStream_t);
t::Tensor classify(t::Tensor x,t::Tensor y,t::Tensor q){
auto c2=t::cat({x,q});core(x,y,c2);int N=c2.size(0),n=x.size(0);auto z=last->tensors[11];auto f=EE({N,512},z.options());auto fp=x.options();auto sim=EE({N-n,n},fp),s10=EE({N-n,10},fp);
neighbor((const B*)z.P(),y.P<long>(),(B*)f.P(),sim.P<float>(),s10.P<float>(),N,n,256,2,3,20,at::cuda::getCurrentCUDAStream());return s10.argmax(1);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("classify",&classify);}'''
CUDA=r'''#include<cuda_runtime.h>
#include<cuda_bf16.h>
#include<cublas_v2.h>
#include<math.h>
#include<stdexcept>
#define F __float2bfloat16_rn
#define G __bfloat162float
#define SD __shfl_down_sync
#define SS __shfl_sync
#define RC reinterpret_cast
#define T threadIdx.x
#define X blockIdx.x
#define KG __global__
using B=__nv_bfloat16;using B2=__nv_bfloat162;
#define B2F __bfloat1622float2
#define B2M __halves2bfloat162
__device__ unsigned pb_hash(unsigned a){a^=a>>16;a*=0x7feb352d;a^=a>>15;a*=0x846ca68b;return a^(a>>16);}
KG void pb_input(const float*x,B*z,int n,int d,int od,float fs){int r=X,t=T;__shared__ float red[256];float v=0;for(int j=t;j<od;j+=256)v+=x[r*od+j]*x[r*od+j];red[t]=v;__syncthreads();for(int s=128;s;s>>=1){if(t<s)red[t]+=red[t+s];__syncthreads();}float scale=sqrtf(float(od)/fmaxf(red[0],1e-20f));for(int j=t;j<d;j+=256)z[r*d+j]=F(j<od?x[r*od+j]*scale:0);}
template<int C>struct Pack;template<>struct Pack<8>{union{uint4 vec;B x[8];};};
template<int M,int C>KG void vec_act(const B*u,B*z,int r2,int bs,float dp,unsigned se){int r=X*4+(T>>5),l1=T&31;if(r>=r2)return;constexpr int R=M/(32*C);Pack<C>v[R];float su=0;
#pragma unroll
for(int k=0;k<R;k++){v[k].vec=*RC<const decltype(v[k].vec)*>(u+r*M+k*32*C+l1*C);
#pragma unroll
for(int j=0;j<C;j++)su+=G(v[k].x[j]);}
for(int sh=16;sh;sh>>=1)su+=SD(0xffffffff,su,sh);float mean=SS(0xffffffff,su,0)/M;
#pragma unroll
for(int k=0;k<R;k++){
#pragma unroll
for(int j=0;j<C;j++){int c3=k*32*C+l1*C+j;B r0=F(fmaxf(G(v[k].x[j])-mean,0));if(dp>0){float r1=(pb_hash((r%bs)*M+c3+se+(r/bs)*101)+1.f)/4294967296.f;r0=F(r1<dp?0:G(r0)/(1-dp));}v[k].x[j]=r0;}*RC<decltype(v[k].vec)*>(z+r*M+k*32*C+l1*C)=v[k].vec;}}
template<int M,int C>KG void vec_back(B*g,const B*z,int r2,float ga){int r=X*4+(T>>5),l1=T&31;if(r>=r2)return;constexpr int R=M/(32*C);float v1[R*C];float su=0;
#pragma unroll
for(int k=0;k<R;k++){Pack<C>a,b;a.vec=*RC<const decltype(a.vec)*>(g+r*M+k*32*C+l1*C);b.vec=*RC<const decltype(b.vec)*>(z+r*M+k*32*C+l1*C);
#pragma unroll
for(int j=0;j<C;j++){float v=G(b.x[j])>0?ga*G(a.x[j]):0;v1[k*C+j]=v;su+=v;}}
for(int sh=16;sh;sh>>=1)su+=SD(0xffffffff,su,sh);float mean=SS(0xffffffff,su,0)/M;
#pragma unroll
for(int k=0;k<R;k++){Pack<C>v;
#pragma unroll
for(int j=0;j<C;j++)v.x[j]=F(v1[k*C+j]-mean);*RC<decltype(v.vec)*>(g+r*M+k*32*C+l1*C)=v.vec;}}
void act(const B*u,B*z,int r2,int m,int bs,float p,unsigned se,cudaStream_t s){vec_act<256,8><<<(r2+3)/4,128,0,s>>>(u,z,r2,bs,p,se);}
void back(B*g,const B*z,int r2,int m,float ga,cudaStream_t s){vec_back<256,8><<<(r2+3)/4,128,0,s>>>(g,z,r2,ga);}
KG void upd(B*w1,B*w2,B*h,const B*g1,const B*g2,const B*gh,B*v1,B*v2,B*vh,int d,int m,int E,int bs,float lr1,float lr2,float hlr,float mu,float wd,bool nv,float*sw1,float*sw2,float*swh,bool cl,const float*hp){
int pair=X*256+T,i=pair*2,tt=E*(m*d+m*m+m*16);if(i>=tt)return;
B*w,*v;const B*g;float lr;float*sw;
if(i<E*m*d){w=w1;g=g1;v=v1;lr=lr1;sw=sw1;}
else if(i<E*(m*d+m*m)){i-=E*m*d;w=w2;g=g2;v=v2;lr=lr2;sw=sw2;}
else{i-=E*(m*d+m*m);w=h;g=gh;v=vh;lr=hlr*256/m;sw=swh;}
float2 ww=B2F(*RC<const B2*>(w+i));
float2 gv;
if(g==gh&&m==256){int size=E*m*16;B gx=F(hp[i]+hp[size+i]+hp[2*size+i]+hp[3*size+i]);B gy=F(hp[i+1]+hp[size+i+1]+hp[2*size+i+1]+hp[3*size+i+1]);*RC<B2*>(const_cast<B*>(gh)+i)=B2M(gx,gy);gv=make_float2(G(gx),G(gy));}
else gv=B2F(*RC<const B2*>(g+i));
float2 vel=B2F(*RC<const B2*>(v+i));
float gx=gv.x/bs+wd*ww.x,gy=gv.y/bs+wd*ww.y;
gx=fminf(GC,fmaxf(-GC,gx));gy=fminf(GC,fmaxf(-GC,gy));
B vx=F(mu*vel.x+gx),vy=F(mu*vel.y+gy);
*RC<B2*>(v+i)=B2M(vx,vy);
B wx=F(ww.x-lr*(nv?gx+mu*G(vx):G(vx)));
B wy=F(ww.y-lr*(nv?gy+mu*G(vy):G(vy)));
*RC<B2*>(w+i)=B2M(wx,wy);if(cl){sw[i]+=G(wx);sw[i+1]+=G(wy);}
}
KG void FH(const B*z,const B*h,B*err,B*du,const long*y,int b,int off,int tt,float mix,unsigned se,float ga,int bn){
__shared__ __align__(16)B hh[16*256];
int t=T,mem=blockIdx.y,l1=t&31,warp=t>>5,r4=X*4+warp;
Pack<8>a0,a1,b0,b1;const B*src=h+mem*256*16+t*32;
a0.vec=*RC<const uint4*>(src);a1.vec=*RC<const uint4*>(src+8);
b0.vec=*RC<const uint4*>(src+16);b1.vec=*RC<const uint4*>(src+24);
#pragma unroll
for(int c3=0;c3<8;c3++){
*RC<B2*>(hh+c3*256+t*2)=B2M(a0.x[c3],b0.x[c3]);
*RC<B2*>(hh+(c3+8)*256+t*2)=B2M(a1.x[c3],b1.x[c3]);}
__syncthreads();if(r4>=b)return;
Pack<8>ff;ff.vec=*RC<const uint4*>(z+(mem*b+r4)*256+l1*8);
float error[10],maxv=-INFINITY;
#pragma unroll
for(int c3=0;c3<10;c3++){Pack<8>ww;ww.vec=*RC<const uint4*>(hh+c3*256+l1*8);float dot=0;
#pragma unroll
for(int j=0;j<8;j++)dot=fmaf(G(ff.x[j]),G(ww.x[j]),dot);
for(int sh=16;sh;sh>>=1)dot+=SD(0xffffffff,dot,sh);
float vv=G(F(SS(0xffffffff,dot,0)));error[c3]=vv;maxv=fmaxf(maxv,vv);}
float su=0;
#pragma unroll
for(int c3=0;c3<10;c3++){error[c3]=expf(error[c3]-maxv);su+=error[c3];}
unsigned ms=se+mem*101;int pt=pb_hash(r4+ms)%tt;
#pragma unroll
for(int c3=0;c3<10;c3++){B e=F(error[c3]/su-((1-mix)*(y[(off+r4%bn)%tt]==c3)+mix*(y[pt]==c3)));error[c3]=G(e);if(l1==c3)err[(mem*b+r4)*16+c3]=e;}
if(l1>=10&&l1<16)err[(mem*b+r4)*16+l1]=F(0);
float g0[8]={0},mean=0;
#pragma unroll
for(int c3=0;c3<10;c3++){Pack<8>ww;ww.vec=*RC<const uint4*>(hh+c3*256+l1*8);
#pragma unroll
for(int j=0;j<8;j++)g0[j]=fmaf(error[c3],G(ww.x[j]),g0[j]);}
#pragma unroll
for(int j=0;j<8;j++){g0[j]=G(ff.x[j])>0?ga*G(F(g0[j])):0;mean+=g0[j];}
for(int sh=16;sh;sh>>=1)mean+=SD(0xffffffff,mean,sh);mean=SS(0xffffffff,mean,0)/256;
Pack<8>out;
#pragma unroll
for(int j=0;j<8;j++)out.x[j]=F(g0[j]-mean);
*RC<uint4*>(du+(mem*b+r4)*256+l1*8)=out.vec;
}
static thread_local cublasHandle_t pb_handle=nullptr;
void handle(cudaStream_t s){if(!pb_handle&&cublasCreate(&pb_handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BF16 state cublasCreate failed");if(cublasSetStream(pb_handle,s)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BF16 state set stream failed");}
static void mm(const B*a,const B*b,B*out,int m,int n,int k,bool at,bool bt,int E,bool sa=false,bool sb=false){float alpha=1,beta=0;auto status=cublasGemmStridedBatchedEx(pb_handle,bt?CUBLAS_OP_N:CUBLAS_OP_T,at?CUBLAS_OP_T:CUBLAS_OP_N,n,m,k,&alpha,b,CUDA_R_16BF,bt?n:k,sb?0:((long long)n*k),a,CUDA_R_16BF,at?m:k,sa?0:((long long)m*k),&beta,out,CUDA_R_16BF,n,(long long)m*n,E,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("ensemble BF16 GEMM failed");}
KG void ei(const B*x,B*out,int bs,int d,int off,int tt,float mix,float sd,unsigned se,int E,int bn,int od){int i=X*256+T;if(i>=E*bs*d)return;int mem=i/(bs*d),local=i%(bs*d),r=local/d,c=local%d;if(c>=od+10){out[i]=F(0);return;}int ni=r*od+c;unsigned ms=se+mem*101+999,ns=se+mem*101+777;int pt=pb_hash(r+ms)%tt;B v=x[((off+r%bn)%tt)*d+c];if(mix>0)v=F((1-mix)*G(v)+mix*G(x[pt*d+c]));if(sd>0&&c<od){float a=(pb_hash(ni+ns)+1.f)/4294967296.f,b=(pb_hash(ni+ns+123456)+1.f)/4294967296.f;v=F(G(v)+sd*sqrtf(-2*logf(a))*cosf(6.2831853f*b));}out[i]=v;}
KG void swa_finalize(B*w1,B*w2,B*h,const float*s1,const float*s2,const float*sh,int d,int m,int E,int count){int i=X*256+T;if(i>=E*(m*d+m*m+m*16))return;B*w;const float*s;if(i<E*m*d){w=w1;s=s1;}else if(i<E*(m*d+m*m)){i-=E*m*d;w=w2;s=s2;}else{i-=E*(m*d+m*m);w=h;s=sh;}w[i]=F(s[i]/count);}
KG void dt(const long*y,B*tg,int n){int i=X*256+T;if(i<n*10)tg[i]=F(y[i/10]==i%10?1.f:0.f);}
KG void dnorm(B*cent,int d,int od){int label=X,l1=T;float su=0;for(int j=l1;j<od;j+=32){float v=G(cent[label*d+j]);su+=v*v;}for(int sh=16;sh;sh>>=1)su+=SD(0xffffffff,su,sh);float norm=sqrtf(SS(0xffffffff,su,0)+1e-20f);for(int j=l1;j<d;j+=32)cent[label*d+j]=F(j<od?G(cent[label*d+j])/norm:0);}
KG void da(B*x,const B*cent,int r2,int d,int od,float scale){int r=X*4+(T>>5),l1=T&31;if(r>=r2)return;for(int label=0;label<10;label++){float su=0;for(int j=l1;j<od;j+=32)su+=G(x[r*d+j])*G(cent[label*d+j]);for(int sh=16;sh;sh>>=1)su+=SD(0xffffffff,su,sh);if(l1==0)x[r*d+od+label]=F(scale*su);}}
KG void iw(B*w,int d,int m,unsigned se,float ga,int kind,int od=0){int i=X*256+T;if(i>=m*d)return;float a=(pb_hash(i+se)+1.f)/4294967296.f,b=(pb_hash(i+se+123456)+1.f)/4294967296.f;w[i]=F(ga*sqrtf(-2*logf(a))*cosf(6.2831853f*b)/sqrtf(float(d)));}
KG void ih(B*h,int m,unsigned se,float ga){int i=X*256+T;if(i<m*16)h[i]=F(0);}
KG void di(B*w,const B*cent,const B*x,int od,int d,int m,int n,unsigned se,int kind,float jt,float ga){int i=X*256+T;if(i>=m*d)return;int r4=i/d,c3=i%d;if(c3>=od){w[i]=F(0);return;}float a=(pb_hash(r4*od+c3+se)+1.f)/4294967296.f,b=(pb_hash(r4*od+c3+se+123456)+1.f)/4294967296.f;float r1=sqrtf(-2*logf(a))*cosf(6.2831853f*b)/sqrtf(float(od));float data=kind==6?G(cent[(r4%10)*d+c3]):G(x[(pb_hash(r4+se)%n)*d+c3])/sqrtf(float(od));w[i]=F(ga*(sqrtf(fmaxf(0,1-jt*jt))*data+jt*r1));}
__device__ __forceinline__ int SO(int r4,int c3,int stride){return r4*stride+(1?(c3^((r4&(stride/8-1))*8)):c3);}
union __align__(16)SC{struct{B a[2][16*64],b[2][64*64];}copy;float tile[1024];struct{float residual[10],part[4],mean;}hh;};
template<bool AT,bool BT,int FIX_N=0>__device__ __forceinline__ void CT(const B*a,const B*b,B*out,float*part,int M,int N_input,int K,int mr,int nc,int bg,int ed,SC&sm){
int N=FIX_N?FIX_N:N_input;int t=T,warp=t>>5;
float acc[8]={};int l1=t&31;
auto pf=[&](int st,int p){
for(int i=t;i<2*64;i+=128){int r=AT?i/2:i/(64/8),c=AT?(i%2)*8:(i%(64/8))*8;const B*src=AT?a+(p+r)*M+mr+c:a+(mr+r)*K+p+c;int vd=AT?(p+r<ed?max(0,min(8,M-mr-c))*2:0):(mr+r<M?max(0,min(8,ed-p-c))*2:0);unsigned dst=__cvta_generic_to_shared(sm.copy.a[st]+SO(r,c,AT?16:64));asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(vd));}
{{for(int i=t;i<(1&&FIX_N==16?2:8)*64;i+=128){int packs=1&&FIX_N==16?2:8,r=i/packs,c=(i%packs)*8;int vd=p+r<ed?max(0,min(8,N-nc-c))*2:0;const B*src=b+(p+r)*N+nc+c;unsigned dst=__cvta_generic_to_shared(sm.copy.b[st]+SO(r,c,64));asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(vd));}}}
asm volatile("cp.async.commit_group;");
};
if(bg<ed)pf(0,bg);
for(int p=bg,st=0;p<ed;p+=64,st^=1){bool next=p+64<ed;if(next)pf(st^1,p+64);if(next)asm volatile("cp.async.wait_group 1;");else asm volatile("cp.async.wait_group 0;");__syncthreads();
if(!1||FIX_N!=16||warp==0){
for(int in=0;in<64&&(!0||p+in<ed);in+=16){
unsigned ar[4],br[4];
const B*ap=sm.copy.a[st]+SO(AT?in+l1%16:l1%16,AT?(l1/16)*8:in+(l1/16)*8,AT?16:64);
unsigned aa=__cvta_generic_to_shared(ap);
{asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(ar[0]),"=r"(ar[2]),"=r"(ar[1]),"=r"(ar[3]):"r"(aa));}
#pragma unroll
for(int half=0;half<2;half++){
const B*bp=sm.copy.b[st]+(BT?SO(in+l1%16,warp*16+half*8,64):SO(warp*16+half*8+l1%8,in+((l1/8)%2)*8,64));
unsigned ba=__cvta_generic_to_shared(bp);
asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(br[0]),"=r"(br[1]):"r"(ba));
float*c=acc+half*4;
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
}
}
}
__syncthreads();
}
#pragma unroll
for(int half=0;half<2;half++){
int r4=l1/4,c3=warp*16+half*8+(l1%4)*2;
sm.tile[r4*64+c3]=acc[half*4];sm.tile[r4*64+c3+1]=acc[half*4+1];
sm.tile[(r4+8)*64+c3]=acc[half*4+2];sm.tile[(r4+8)*64+c3+1]=acc[half*4+3];
}
__syncthreads();
for(int i=t;i<1024;i+=128){int r=i/64,c=i%64;if(mr+r<M&&nc+c<N){if(part)part[(mr+r)*N+nc+c]=sm.tile[i];else out[(mr+r)*N+nc+c]=F(sm.tile[i]);}}
__syncthreads();
}
KG __launch_bounds__(128,5)void GP(const B*z,const B*err,float*parts,int m,int bs,int E){
__shared__ SC sm;int job=X,mem=job%E,local=job/E,split=local%4,mr=(local/4)*16,chunk=((bs+63)/64)*16,bg=min(bs,split*chunk),ed=min(bs,bg+chunk);
CT<true,true,16>(z+mem*bs*m,err+mem*bs*16,nullptr,parts+(split*E+mem)*m*16,m,16,bs,mr,0,bg,ed,sm);
}
static void HG(const B*z,const B*err,float*parts,int m,int bs,int E,cudaStream_t s){GP<<<E*((m+15)/16)*4,128,0,s>>>(z,err,parts,m,bs,E);}
void LP(const float*x,const long*y,const float*q,long*out,B*tx,B*tq,B*w1,B*w2,B*h,B*u1,B*u2,B*z1,B*z2,B*du1,B*du2,B*dw1,B*dw2,B*dh,B*err,B*lg,B*v1,B*v2,B*vh,B*aug,B*tg,B*ct,float*sw1,float*sw2,float*swh,float*hp,int n,int nq,int steps,cudaStream_t s){constexpr int d=128,od=60,m=256,E=2,bs=1024,views=1,schedule=0,ik=7;constexpr unsigned se=42;
constexpr float lr1=.4f,lr2=.4f,hlr=.4f,mu=.95f,wd=.001f,dp=.1f,noise=.3f,isc=-1.f,mix=.05f,ig=1.f,hg=0.f,sf=.25f,ds=1.f,dn=.25f;
int swa_count=sf>0?max(1,int(steps*sf)):0;bool nv=true;handle(s);pb_input<<<n,256,0,s>>>(x,tx,n,d,od,isc);pb_input<<<nq,256,0,s>>>(q,tq,nq,d,od,isc);dt<<<(n*10+255)/256,256,0,s>>>(y,tg,n);mm(tg,tx,ct,10,d,n,true,true,1);dnorm<<<10,32,0,s>>>(ct,d,od);da<<<(n+3)/4,128,0,s>>>(tx,ct,n,d,od,ds);da<<<(nq+3)/4,128,0,s>>>(tq,ct,nq,d,od,ds);for(int mem=0;mem<E;mem++){unsigned es=se+mem*101;if(ik>=6)di<<<(m*d+255)/256,256,0,s>>>(w1+mem*m*d,ct,tx,od,d,m,n,es,ik,dn,ig);iw<<<(m*m+255)/256,256,0,s>>>(w2+mem*m*m,m,m,es+123123,ig,ik);ih<<<(m*16+255)/256,256,0,s>>>(h+mem*m*16,m,es+456456,hg);}cudaMemsetAsync(v1,0,E*m*d*2,s);cudaMemsetAsync(v2,0,E*m*m*2,s);cudaMemsetAsync(vh,0,E*m*16*2,s);if(swa_count){cudaMemsetAsync(sw1,0,E*m*d*4,s);cudaMemsetAsync(sw2,0,E*m*m*4,s);cudaMemsetAsync(swh,0,E*m*16*4,s);}
for(int t=0;t<steps;t++){float decay=1.f;int off=(t*bs)%n,bn=min(bs,n),b=bn*views;const B*xb=aug;ei<<<(E*b*d+255)/256,256,0,s>>>(tx,aug,b,d,off,n,mix,noise,se+t*100003,E,bn,od);
mm(xb,w1,u1,b,m,d,false,false,E);act(u1,z1,E*b,m,b,dp,se+t*100003+111,s);mm(z1,w2,u2,b,m,m,false,false,E);act(u2,z2,E*b,m,b,dp,se+t*100003+333,s);FH<<<dim3((b+3)/4,E),128,0,s>>>(z2,h,err,du2,y,b,off,n,mix,se+t*100003+999,1/(1-dp),bn);
HG(z2,err,hp,m,b,E,s);mm(du2,z1,dw2,m,m,b,true,true,E);mm(du2,w2,du1,b,m,m,false,true,E);back(du1,z1,E*b,m,1/(1-dp),s);mm(du1,xb,dw1,m,d,b,true,true,E);
upd<<<(E*(m*d+m*m+m*16)/2+255)/256,256,0,s>>>(w1,w2,h,dw1,dw2,dh,v1,v2,vh,d,m,E,b,decay*lr1,decay*lr2,decay*hlr,mu,wd,nv,sw1,sw2,swh,swa_count&&t>=steps-swa_count,hp);
}
if(swa_count&&steps>0)swa_finalize<<<(E*(m*d+m*m+m*16)+255)/256,256,0,s>>>(w1,w2,h,sw1,sw2,swh,d,m,E,swa_count);
mm(tq,w1,u1,nq,m,d,false,false,E,true);act(u1,z1,E*nq,m,nq,0,0,s);mm(z1,w2,u2,nq,m,m,false,false,E);act(u2,z2,E*nq,m,nq,0,0,s);
}
static void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("neighbor GEMM");}
KG void pack(const B*z,B*f,int N,int h,int E){int r=X,t=T,D=E*h;__shared__ float s8[8];float su=0;for(int j=t;j<D;j+=256){float v=G(z[((j/h)*N+r)*h+j%h]);su+=v*v;}for(int k=16;k;k>>=1)su+=SD(0xffffffff,su,k);if((t&31)==0)s8[t/32]=su;__syncthreads();if(t==0){su=0;for(int j=0;j<8;j++)su+=s8[j];s8[0]=rsqrtf(fmaxf(su,1e-20f));}__syncthreads();for(int j=t;j<D;j+=256)f[r*D+j]=F(G(z[((j/h)*N+r)*h+j%h])*s8[0]);}
KG void votes(const float*s,const long*y,float*out,int n,int k,float t2){
int r=X,t=T;float a=-INFINITY,b=-INFINITY,c=-INFINITY;int ia=-1,ib=-1,ic=-1;
for(int i=t;i<n;i+=128){float v=s[r*n+i];if(v>c){if(v>b){if(v>a){c=b;ic=ib;b=a;ib=ia;a=v;ia=i;}else{c=b;ic=ib;b=v;ib=i;}}else{c=v;ic=i;}}}
__shared__ float c1[128],score[10];__shared__ int ids[128];if(t<10)score[t]=0;__syncthreads();
#pragma unroll
for(int j=0;j<3;j++){
c1[t]=a;ids[t]=ia;__syncthreads();
if(t==0){int w3=0;for(int i=1;i<128;i++)if(c1[i]>c1[w3])w3=i;int id=ids[w3];score[y[id]]+=expf((c1[w3]-1)*t2);ids[0]=w3;}
__syncthreads();if(t==ids[0]){a=b;ia=ib;b=c;ib=ic;c=-INFINITY;ic=-1;}__syncthreads();
}
if(t<10)out[r*10+t]=score[t];
}
void neighbor(const B*z,const long*y,B*f,float*s,float*out,int N,int n,int h,int E,int k,float t2,cudaStream_t s2){
static cublasHandle_t handle=nullptr;if(!handle)cb(cublasCreate(&handle));cb(cublasSetStream(handle,s2));int D=E*h;float one=1,zero=0;pack<<<N,256,0,s2>>>(z,f,N,h,E);
cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,N-n,D,&one,f,CUDA_R_16BF,D,f+n*D,CUDA_R_16BF,D,&zero,s,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));votes<<<N-n,128,0,s2>>>(s,y,out,n,k,t2);
}'''

import os,hashlib
from pathlib import Path
EXT=None
def classify(train_x,train_y,test_x):
 global EXT
 if EXT is None:
  from torch.utils.cpp_extension import load
  os.environ.setdefault("TORCH_CUDA_ARCH_LIST","8.0")
  tag=hashlib.sha256((CPP+CUDA).encode()).hexdigest()[:12]
  path=Path("/tmp")/("sutro-neighbor-"+tag+str(os.getuid()));path.mkdir(exist_ok=True)
  (path/"main.cpp").write_text(CPP);(path/"main.cu").write_text(CUDA)
  EXT=load(name="sutro_neighbor_"+tag,sources=[str(path/"main.cpp"),str(path/"main.cu")],extra_cuda_cflags=["-O3","-lineinfo","-DGC=0.03f"],extra_ldflags=["-lcublas"],verbose=False)
 return EXT.classify(train_x,train_y,test_x)
