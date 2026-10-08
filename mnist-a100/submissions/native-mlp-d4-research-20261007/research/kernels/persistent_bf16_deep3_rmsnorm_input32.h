#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
void prepare_pb_handle(cudaStream_t);
void launch_pb(const float*x,const long*y,const float*q,long*out,B*tx,B*tq,B*w1,B*w2,B*h,B*u1,B*u2,B*z1,B*z2,B*du1,B*du2,B*dw1,B*dw2,B*dh,B*err,B*logits,B*v1,B*v2,B*vh,B*aug,B*targets,B*centroids,float*sw1,float*sw2,float*swh,B*w3,B*u3,B*z3,B*du3,B*dw3,B*v3,float*sw3,B*w4,B*u4,B*z4,B*du4,B*dw4,B*v4,float*sw4,float*stats1,float*stats2,float*stats3,float*stats4,int n,int nq,int d,int original,int m,int E,int batch,int views,int steps,int schedule,float lr1,float lr2,float hlr,float momentum,float wd,float dropout,float noise,float input_scale,float mix,float init_gain,float head_gain,int init_kind,float swa_fraction,float dictionary_scale,float data_noise,float smoothing,unsigned seed,cudaStream_t s);

void launch_vector_check(const B*u,B*z,B*g,int rows,int m,int batch,float p,unsigned seed,float*stats,cudaStream_t s);

void head16_check(const B*,const B*,B*,B*,B*,B*,const long*,int,int,int,int,int,int,float,float,unsigned,cudaStream_t);
