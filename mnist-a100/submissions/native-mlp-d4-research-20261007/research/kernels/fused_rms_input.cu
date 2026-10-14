// Explicit A100 ldmatrix + mma.sync, K32 double-buffer XOR-swizzled operands.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace input_rms_fusion {
using B=__nv_bfloat16;
__device__ unsigned rms_hash(unsigned a){a^=a>>16;a*=0x7feb352d;a^=a>>15;a*=0x846ca68b;a^=a>>16;return a;}
__device__ __forceinline__ int swizzle(int row,int col){return row*32+(col^(((row>>1)&3)*8));}
template<int N>__device__ void copy_stage(const B*A,const B*W,B*sm,int row,int K,int kt){int t=threadIdx.x;for(int i=t;i<(16+N)*32/8;i+=(N/64)*32){int el=i*8,r=el/32,c=el%32;const B*src=r<16?A+(row+r)*K+kt*32+c:W+(r-16)*K+kt*32+c;unsigned dst=__cvta_generic_to_shared(sm+swizzle(r,c));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src));}asm volatile("cp.async.commit_group;");}
template<int Rows,int N>__global__ void fused(const B*A,const B*W,B*U,B*Z,int b,int K,float p,unsigned seed){int member=blockIdx.y,row=blockIdx.x*Rows,t=threadIdx.x,warp=t/32,lane=t%32;extern __shared__ __align__(128) unsigned char storage[];constexpr int Stage=(16+N)*32;B*stages=reinterpret_cast<B*>(storage);float*tile=reinterpret_cast<float*>(storage);A+=member*b*K;W+=member*N*K;U+=member*b*N;Z+=member*b*N;
 for(int mt=0;mt<Rows;mt+=16){float acc[8][4]={};copy_stage<N>(A,W,stages,row+mt,K,0);asm volatile("cp.async.wait_group 0;");__syncthreads();for(int kt=0;kt<K/32;kt++){int current=kt&1;if(kt+1<K/32)copy_stage<N>(A,W,stages+(1-current)*Stage,row+mt,K,kt+1);
#pragma unroll
 for(int inner=0;inner<32;inner+=16){unsigned ar[4];unsigned aa=__cvta_generic_to_shared(stages+current*Stage+swizzle(lane%16,inner+(lane/16)*8));asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(aa));
#pragma unroll
 for(int half=0;half<8;half++){unsigned br[2];int nr=warp*64+half*8+lane%8;unsigned ba=__cvta_generic_to_shared(stages+current*Stage+16*32+swizzle(nr,inner+((lane/8)%2)*8));asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(br[0]),"=r"(br[1]):"r"(ba));float*c=acc[half];asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));}}
 asm volatile("cp.async.wait_group 0;");__syncthreads();}
#pragma unroll
 for(int half=0;half<8;half++){int rr=lane/4,col=warp*64+half*8+(lane%4)*2;*reinterpret_cast<float2*>(tile+rr*(N+8)+col)=make_float2(acc[half][0],acc[half][1]);*reinterpret_cast<float2*>(tile+(rr+8)*(N+8)+col)=make_float2(acc[half][2],acc[half][3]);}__syncthreads();
 // One warp owns each row. Four independent virtual warp sums reproduce the
 // existing 128-thread RMS reduction order, including BF16 rounding of U.
 for(int rr=warp;rr<16;rr+=N/64){float sums[4]={0,0,0,0};for(int v=0;v<4;v++)for(int c=lane+v*32;c<N;c+=128){B rounded=__float2bfloat16_rn(tile[rr*(N+8)+c]);U[(row+mt+rr)*N+c]=rounded;float x=__bfloat162float(rounded);sums[v]+=x*x;}for(int v=0;v<4;v++)for(int d=16;d;d>>=1)sums[v]+=__shfl_down_sync(0xffffffff,sums[v],d);float sq=__shfl_sync(0xffffffff,sums[0]+sums[1]+sums[2]+sums[3],0);float inv=rsqrtf(sq/N+1e-5f);for(int c=lane;c<N;c+=32){float x=__bfloat162float(__float2bfloat16_rn(tile[rr*(N+8)+c]));B out=__float2bfloat16_rn(fmaxf(x*inv,0.f));if(p>0){float random=(rms_hash((row+mt+rr)*N+c+seed+member*101)+1.f)/4294967296.f;float v=__bfloat162float(out);out=__float2bfloat16_rn(random<p?0:(p==.1f?v*1.111111164093017578125f:v/(1-p)));}Z[(row+mt+rr)*N+c]=out;}}
 __syncthreads(); // U tile is dead; staging can overwrite it for next 16 rows.
 }
}
template<int N>void launch_width(const B*A,const B*W,B*U,B*Z,int E,int b,int K,int rows,float p,unsigned seed,cudaStream_t s){int smem=2*(16+N)*32*2,threads=(N/64)*32;if(rows==16){cudaFuncSetAttribute(fused<16,N>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);fused<16,N><<<dim3(b/16,E),threads,smem,s>>>(A,W,U,Z,b,K,p,seed);}else{cudaFuncSetAttribute(fused<32,N>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);fused<32,N><<<dim3(b/32,E),threads,smem,s>>>(A,W,U,Z,b,K,p,seed);}}
void fused_rms_banks_launch(const B*A,const B*W,B*U,B*Z,int E,int b,int K,int N,int rows,float p,unsigned seed,cudaStream_t s){if(N==128)launch_width<128>(A,W,U,Z,E,b,K,rows,p,seed,s);else if(N==256)launch_width<256>(A,W,U,Z,E,b,K,rows,p,seed,s);else launch_width<512>(A,W,U,Z,E,b,K,rows,p,seed,s);}

}
void input_rms_fused(const __nv_bfloat16*A,const __nv_bfloat16*W,__nv_bfloat16*U,__nv_bfloat16*Z,int E,int b,int K,int N,float p,unsigned seed,cudaStream_t stream){input_rms_fusion::fused_rms_banks_launch(A,W,U,Z,E,b,K,N,16,p,seed,stream);}
