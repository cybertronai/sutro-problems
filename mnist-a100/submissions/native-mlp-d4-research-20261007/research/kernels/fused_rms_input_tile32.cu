// Standalone Ampere experiment: true 32x512 output tile, shared B reuse across two M fragments.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
namespace input_tile32_fusion {
using B=__nv_bfloat16;
__device__ unsigned rms_hash(unsigned a){a^=a>>16;a*=0x7feb352d;a^=a>>15;a*=0x846ca68b;return a^(a>>16);}
__device__ __forceinline__ int tile32_swizzle(int row,int col){return row*32+(col^(((row>>1)&3)*8));}
__device__ __forceinline__ void tile32_copy(const B*A,const B*W,B*stage,int row,int K,int kt){for(int i=threadIdx.x;i<544*4;i+=256){int r=i/4,c=(i%4)*8;const B*src=r<32?A+(row+r)*K+kt*32+c:W+(r-32)*K+kt*32+c;unsigned dst=__cvta_generic_to_shared(stage+tile32_swizzle(r,c));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src));}asm volatile("cp.async.commit_group;");}
__global__ void tile32_kernel(const B*A,const B*W,B*U,B*Z,int b,int K,float p,unsigned seed){int member=blockIdx.y,row=blockIdx.x*32,lane=threadIdx.x&31,warp=threadIdx.x>>5;constexpr int Stage=544*32;extern __shared__ __align__(128) unsigned char storage[];B*stages=reinterpret_cast<B*>(storage);float*tile=reinterpret_cast<float*>(storage);A+=member*b*K;W+=member*512*K;U+=member*b*512;Z+=member*b*512;
 // Both 16-row fragments live through the entire K loop. B is loaded/copied once.
 float acc[2][8][4]={};tile32_copy(A,W,stages,row,K,0);asm volatile("cp.async.wait_group 0;");__syncthreads();
 for(int kt=0;kt<K/32;kt++){int current=kt&1;if(kt+1<K/32)tile32_copy(A,W,stages+(1-current)*Stage,row,K,kt+1);
#pragma unroll
  for(int inner=0;inner<32;inner+=16){unsigned ar[2][4];
#pragma unroll
   for(int mr=0;mr<2;mr++){unsigned addr=__cvta_generic_to_shared(stages+current*Stage+tile32_swizzle(mr*16+lane%16,inner+(lane/16)*8));asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(ar[mr][0]),"=r"(ar[mr][1]),"=r"(ar[mr][2]),"=r"(ar[mr][3]):"r"(addr));}
#pragma unroll
   for(int half=0;half<8;half++){unsigned br[2];int nr=warp*64+half*8+lane%8;unsigned addr=__cvta_generic_to_shared(stages+current*Stage+32*32+tile32_swizzle(nr,inner+((lane/8)%2)*8));asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(br[0]),"=r"(br[1]):"r"(addr));
#pragma unroll
    for(int mr=0;mr<2;mr++){float*c=acc[mr][half];asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(ar[mr][0]),"r"(ar[mr][1]),"r"(ar[mr][2]),"r"(ar[mr][3]),"r"(br[0]),"r"(br[1]));}}}
  asm volatile("cp.async.wait_group 0;");__syncthreads();}
 // All async writes have completed. Alias dead staging as a 32x520 FP32 epilogue tile.
#pragma unroll
 for(int mr=0;mr<2;mr++){
#pragma unroll
  for(int half=0;half<8;half++){int rr=mr*16+lane/4,col=warp*64+half*8+(lane%4)*2;*reinterpret_cast<float2*>(tile+rr*520+col)=make_float2(acc[mr][half][0],acc[mr][half][1]);*reinterpret_cast<float2*>(tile+(rr+8)*520+col)=make_float2(acc[mr][half][2],acc[mr][half][3]);}}
 __syncthreads();
 for(int rr=warp;rr<32;rr+=8){float uv[4][4],sq[4]={};
#pragma unroll
  for(int v=0;v<4;v++){
#pragma unroll
   for(int k=0;k<4;k++){int c=lane+v*32+k*128;B rounded=__float2bfloat16_rn(tile[rr*520+c]);U[(row+rr)*512+c]=rounded;uv[v][k]=__bfloat162float(rounded);sq[v]+=uv[v][k]*uv[v][k];}}
  // Current maskregen forward sums its four virtual groups before warp reduction.
  float sum=sq[0]+sq[1]+sq[2]+sq[3];for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);float inv=rsqrtf(__shfl_sync(0xffffffff,sum,0)/512+1e-5f);
#pragma unroll
  for(int v=0;v<4;v++){
#pragma unroll
   for(int k=0;k<4;k++){int c=lane+v*32+k*128;B out=__float2bfloat16_rn(fmaxf(uv[v][k]*inv,0.f));if(p>0){float rnd=(rms_hash((row+rr)*512+c+seed+member*101)+1.f)/4294967296.f;float value=__bfloat162float(out);out=__float2bfloat16_rn(rnd<p?0:(p==.1f?value*1.111111164093017578125f:value/(1-p)));}Z[(row+rr)*512+c]=out;}}}
}
void fused_tile32_launch(const B*A,const B*W,B*U,B*Z,int E,int b,int K,float p,unsigned seed,cudaStream_t s){constexpr int Shared=2*544*32*2;auto status=cudaFuncSetAttribute(tile32_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared);if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));tile32_kernel<<<dim3(b/32,E),256,Shared,s>>>(A,W,U,Z,b,K,p,seed);}

} // namespace input_tile32_fusion
void input_rms_tile32(const __nv_bfloat16*A,const __nv_bfloat16*W,__nv_bfloat16*U,__nv_bfloat16*Z,int E,int b,int K,int N,float p,unsigned seed,cudaStream_t s){if(K!=64||N!=512||b%32)throw std::runtime_error("unsupported input tile32 shape");input_tile32_fusion::fused_tile32_launch(A,W,U,Z,E,b,K,p,seed,s);}
