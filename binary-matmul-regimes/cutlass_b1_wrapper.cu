// SPDX-License-Identifier: BSD-3-Clause
// Minimal C ABI around NVIDIA CUTLASS 3.8.0 B1 AND+POPC GEMM kernels.
//
// Logical operation:
//   D[m,n] = sum_k (A[m,k] & B[k,n])
// A is packed row-major, B is packed column-major, and D is S32 column-major.
// Each uint1b_t stores one bit; CUTLASS packs consecutive bits into bytes.

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using ElementInput = cutlass::uint1b_t;
using ElementOutput = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using Accumulator = int32_t;
using Compute = int32_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    Accumulator,
    Compute>;

template <typename ThreadblockShape, typename WarpShape, int Stages>
using B1Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    LayoutA,
    ElementInput,
    LayoutB,
    ElementOutput,
    LayoutC,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    128,
    128,
    false,
    cutlass::arch::OpAndPopc>;

using Gemm128x256K1024 = B1Gemm<
    cutlass::gemm::GemmShape<128, 256, 1024>,
    cutlass::gemm::GemmShape<64, 64, 1024>,
    3>;
using Gemm256x128K1024 = B1Gemm<
    cutlass::gemm::GemmShape<256, 128, 1024>,
    cutlass::gemm::GemmShape<64, 64, 1024>,
    3>;
using Gemm128x128K1024 = B1Gemm<
    cutlass::gemm::GemmShape<128, 128, 1024>,
    cutlass::gemm::GemmShape<64, 64, 1024>,
    4>;
using Gemm64x128K1024 = B1Gemm<
    cutlass::gemm::GemmShape<64, 128, 1024>,
    cutlass::gemm::GemmShape<32, 64, 1024>,
    3>;
using Gemm128x64K1024 = B1Gemm<
    cutlass::gemm::GemmShape<128, 64, 1024>,
    cutlass::gemm::GemmShape<64, 32, 1024>,
    3>;
using Gemm64x64K1024 = B1Gemm<
    cutlass::gemm::GemmShape<64, 64, 1024>,
    cutlass::gemm::GemmShape<32, 32, 1024>,
    5>;

using Gemm128x256K512 = B1Gemm<
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    3>;
using Gemm256x128K512 = B1Gemm<
    cutlass::gemm::GemmShape<256, 128, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    3>;
using Gemm64x256K512 = B1Gemm<
    cutlass::gemm::GemmShape<64, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    4>;
using Gemm256x64K512 = B1Gemm<
    cutlass::gemm::GemmShape<256, 64, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    4>;
using Gemm128x128K512 = B1Gemm<
    cutlass::gemm::GemmShape<128, 128, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    5>;
using Gemm64x128K512 = B1Gemm<
    cutlass::gemm::GemmShape<64, 128, 512>,
    cutlass::gemm::GemmShape<32, 64, 512>,
    6>;
using Gemm128x64K512 = B1Gemm<
    cutlass::gemm::GemmShape<128, 64, 512>,
    cutlass::gemm::GemmShape<64, 32, 512>,
    6>;
using Gemm64x64K512 = B1Gemm<
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<32, 32, 512>,
    10>;

template <typename Gemm>
int launch_gemm(
    void const *a,
    void const *b,
    void *d,
    int m,
    int n,
    int k,
    void *stream_ptr) {
  // Leading dimensions are expressed in logical elements (bits for A/B).
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<ElementInput const *>(a), k},
      {reinterpret_cast<ElementInput const *>(b), k},
      {reinterpret_cast<ElementOutput const *>(d), m},
      {reinterpret_cast<ElementOutput *>(d), m},
      {1, 0});

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, reinterpret_cast<cudaStream_t>(stream_ptr));
  return static_cast<int>(status);
}

#define EXPORT_GEMM(name, type)                                                \
  extern "C" int name(                                                        \
      void const *a, void const *b, void *d, int m, int n, int k,             \
      void *stream_ptr) {                                                      \
    return launch_gemm<type>(a, b, d, m, n, k, stream_ptr);                   \
  }

EXPORT_GEMM(cutlass_b1_128x256_k1024, Gemm128x256K1024)
EXPORT_GEMM(cutlass_b1_256x128_k1024, Gemm256x128K1024)
EXPORT_GEMM(cutlass_b1_128x128_k1024, Gemm128x128K1024)
EXPORT_GEMM(cutlass_b1_64x128_k1024, Gemm64x128K1024)
EXPORT_GEMM(cutlass_b1_128x64_k1024, Gemm128x64K1024)
EXPORT_GEMM(cutlass_b1_64x64_k1024, Gemm64x64K1024)
EXPORT_GEMM(cutlass_b1_128x256_k512, Gemm128x256K512)
EXPORT_GEMM(cutlass_b1_256x128_k512, Gemm256x128K512)
EXPORT_GEMM(cutlass_b1_64x256_k512, Gemm64x256K512)
EXPORT_GEMM(cutlass_b1_256x64_k512, Gemm256x64K512)
EXPORT_GEMM(cutlass_b1_128x128_k512, Gemm128x128K512)
EXPORT_GEMM(cutlass_b1_64x128_k512, Gemm64x128K512)
EXPORT_GEMM(cutlass_b1_128x64_k512, Gemm128x64K512)
EXPORT_GEMM(cutlass_b1_64x64_k512, Gemm64x64K512)

extern "C" char const *cutlass_b1_build_description() {
  return "CUTLASS 3.8.0; SM80; mma.sync m16n8k256; B1 AND+POPC -> S32";
}
