"""Fused PCANet stage-two convolution, thresholding and nine-block histogram.

One CUDA block handles each (image, first-stage-filter) plane. It stages
inputs and filters in shared memory, uses the archived inline PTX FP32 FMA
and strict-positive threshold, then writes nine 32-bin histograms directly.
The feature normalization remains in model.py. Compiler flags are retained
from the original implementation: -O3 and --use_fast_math.
"""
# The CUDA arithmetic below is retained from the archived implementation.
# Wrapper fixes: launch on PyTorch's current stream, guard the tensor device,
# validate inputs, and surface CUDA launch errors. Compilation remains lazy.
import torch
from torch.utils.cpp_extension import load_inline

SRC = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define H   28
#define KS  7
#define PAD 3
#define SH  (H + 2*PAD)          // 34
#define BLK 14
#define NBS 3                    // blocks per side at stride 7
#define NB  (NBS*NBS)            // 9

// one block per (image, group) plane; blockDim.x = 256
template<int L2>
__global__ void pcanet_fused(const float* __restrict__ M, const float* __restrict__ W,
                             float* __restrict__ HIST, int n, int L1, int stride_blk) {
    const int plane = blockIdx.x;
    const int img   = plane / L1;
    const int grp   = plane % L1;
    const int NBINS = 1 << L2;

    extern __shared__ float sm[];
    float* tile = sm;                          // SH*SH staged input
    float* wts  = tile + SH*SH;                // L2*KS*KS filters for this group
    float* hist = wts + L2*KS*KS;              // NB*NBINS accumulator

    const float* Mp = M + (size_t)img * L1 * H * H + (size_t)grp * H * H;

    for (int i = threadIdx.x; i < SH*SH; i += blockDim.x) {
        int y = i / SH - PAD, x = i % SH - PAD;
        tile[i] = (y >= 0 && y < H && x >= 0 && x < H) ? Mp[y*H + x] : 0.0f;
    }
    for (int i = threadIdx.x; i < L2*KS*KS; i += blockDim.x) wts[i] = W[(size_t)grp*L2*KS*KS + i];
    for (int i = threadIdx.x; i < NB*NBINS; i += blockDim.x) hist[i] = 0.0f;
    __syncthreads();

    for (int p = threadIdx.x; p < H*H; p += blockDim.x) {
        const int py = p / H, px = p % H;
        int code = 0;
        #pragma unroll
        for (int b = 0; b < L2; ++b) {
            float s = 0.0f;
            const float* wb = wts + b*KS*KS;
            #pragma unroll
            for (int ky = 0; ky < KS; ++ky) {
                const float* trow = tile + (py + ky)*SH + px;
                #pragma unroll
                for (int kx = 0; kx < KS; ++kx) {
                    // inline PTX: one fused multiply-add, round-to-nearest, no library call
                    asm("fma.rn.f32 %0, %1, %2, %0;" : "+f"(s) : "f"(trow[kx]), "f"(wb[ky*KS + kx]));
                }
            }
            unsigned pred;
            asm("set.gt.u32.f32 %0, %1, 0f00000000;" : "=r"(pred) : "f"(s));
            code |= (int)(pred & 1u) << b;
        }
        // this pixel lands in every block whose window covers it
        #pragma unroll
        for (int br = 0; br < NBS; ++br) {
            int r0 = br * stride_blk;
            if (py < r0 || py >= r0 + BLK) continue;
            #pragma unroll
            for (int bc = 0; bc < NBS; ++bc) {
                int c0 = bc * stride_blk;
                if (px < c0 || px >= c0 + BLK) continue;
                atomicAdd(&hist[(br*NBS + bc)*NBINS + code], 1.0f);
            }
        }
    }
    __syncthreads();
    float* Hp = HIST + (size_t)plane * NB * NBINS;
    for (int i = threadIdx.x; i < NB*NBINS; i += blockDim.x) Hp[i] = hist[i];
}

torch::Tensor pcanet_hist(torch::Tensor M1, torch::Tensor W, int64_t L1, int64_t L2, int64_t stride_blk) {
    TORCH_CHECK(M1.is_cuda() && W.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(M1.device() == W.device(), "inputs must share a CUDA device");
    TORCH_CHECK(M1.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32,
                "inputs must have dtype float32");
    TORCH_CHECK(M1.is_contiguous() && W.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(L1 > 0 && L2 == 5, "L1 must be positive; only L2=5 is compiled");
    TORCH_CHECK(stride_blk == 7, "the nine-block histogram requires stride 7");
    TORCH_CHECK(M1.dim() == 4 && M1.size(1) == L1 && M1.size(2) == H && M1.size(3) == H,
                "M1 must have shape (n, L1, 28, 28)");
    TORCH_CHECK(W.numel() == L1 * L2 * KS * KS, "unexpected filter element count");
    TORCH_CHECK(M1.size(0) * L1 <= 2147483647, "too many image/filter planes");
    const c10::cuda::CUDAGuard device_guard(M1.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(M1.get_device());
    const int n = M1.size(0);
    const int NBINS = 1 << L2;
    auto out = torch::empty({n, (long)L1, NB, (long)NBINS}, M1.options());
    if (n == 0) return out;
    size_t shmem = (SH*SH + L2*KS*KS + NB*NBINS) * sizeof(float);
    dim3 grid(n * L1), block(256);
    pcanet_fused<5><<<grid, block, shmem, stream.stream()>>>(
        M1.data_ptr<float>(), W.data_ptr<float>(), out.data_ptr<float>(),
        n, (int)L1, (int)stride_blk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

_mod = None


def get():
    global _mod
    if _mod is None:
        decl = 'torch::Tensor pcanet_hist(torch::Tensor M1, torch::Tensor W, int64_t L1, int64_t L2, int64_t stride_blk);'
        _mod = load_inline(name='sutro_original_pcanet_cuda_v1', cpp_sources=decl, cuda_sources=SRC,
                           functions=['pcanet_hist'], verbose=False,
                           extra_cuda_cflags=['-O3', '--use_fast_math'])
    return _mod


def cuda_hist(M1, W2g, L1, L2, stride_blk=7):
    """M1 (n,L1,H,H), filters (L1*L2,1,KS,KS) -> raw block histograms (n, L1, 9, 32)"""
    return get().pcanet_hist(M1.contiguous(), W2g.contiguous().view(-1), L1, L2, stride_blk)
