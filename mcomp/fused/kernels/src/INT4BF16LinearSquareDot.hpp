/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Hopper GEMM example with different data types using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example shows how to perform INT4 x FP8 GEMM and scale up the INT4 weight during dequantization. It uses a look-up table to avoid the multiplications
    between INT4 and FP8. To trigger this method, use cutlass::Array<ElementScale, 8> as the scale type in the collective's arguments.
    
    However, this algorithm requires changes to the encoding of INT4 weights and scale factors. These changes must happen before launching the GEMM. See the helper functions
    `unify_quant_encoding`, `initialize_packed_scale` in the header `fp8_packed_scale.hpp` for details.

    In a nutshell, the positive values of INT4 weights need to be encoded in the same way as negative values except for the sign bit. For each scale factor,
    8 negative results (-8 x scale, -7 x scale, ... -1 x scale) are packed together, forming a cutlass::Array<ElementScale, 8> value.

    The narrower type always passes through the register file. Therefore, in cases where the narrower type is operand B, the collective will implicitly swap 
    A and B in the main loop. However, as a result of this collective performing implicit swaps, it does not support TMA epilogues. Consequently, it is essential to consider this when constructing the epilogue, 
    as illustrated in this example.

    Note that in this example, we explicitly swap A and B in order to use TMA epilogues. We do this since TMA epilogues are more performant on problem sizes of interest.

    As an additional optimization, we can reorder the narrow data type tensor such that elements read into register file by the same thread are contiguous in global and shared memory.
    This promotes vectorization of shared memory loads and removes additional instructions on the critical path. For example, when MMA is performed in FP8 data type, each thread reads
    4 groups of 4 elements that are logically contiguous in the same row (refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n32-a for thread-value layout).
    If the narrow type is INT4 and tensor is major in K dim, only 16 bits can be read at a time, leading to extra load instructions and suboptimal utilization of shared memory throughput.
    If we reorder the data offline to place all 16 elements read by a thread contiguously in memory, a single 64-bit load is sufficient. This reordering is often feasible when the quantized
    tensor is static (e.g. weight tensor of a NN layer at inference time). This example demonstrates how such a reordering can be performed and communicated to the kernel when the options.shuffle is set to true.

    It is expected that the scale's K dimension be scale_k = ceil_div(problem_k, group_size). 
    
    Scales are always expected to be MN major. This means the fastest changing dimension must be M if A is scaled or N if B is scaled.
    
    If A is being scaled, the scales must have shape [M, scale_k],  while if B is scaled, it must have shape [N, scale_k].

    The implementation only supports "group-wise" scales. However, we can make it work for per-column scales by setting the group's size
    equal to the gemm problem K.

    Limitations:
      1) Only supports INT4 x { FP8, INT8, UINT8 }. The scales must be the same as mma Type. Scale with zero-point mode is not supported.
      2) The INT4 weights and scale factors have additional encoding requirements.
      3) The scales must be MN major. That means if A is scaled, it must be column major, but if B is scaled it must be row major.
      4) The scales must have the same layout and groupsize.
      5) The groupsize must be greater or equal to the tile shape k.
      6) Currently, TMA epilogues cannot be used when the narrow type is the B operand. This limitation arises because the implementation always swaps the 
         operands to ensure that the narrow type passes through the register file, and TMA epilogues do not currently support implicit swap + transpose operations. 
         We plan to address this limitation in the future. However, we address this in the example by explicitly swapping and transposing the operands.
    
    Optimizing suggestions:
      1) Use a small tile size, since the register pressure for this GEMM (and RS GEMM in general) is high (it uses a lot of register space).

    Examples:
      
      Runs the mixed input batched gemm (with batch size 2), converting B to the type of A (mode 0)
      $ ./examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm --m=2048 --n=2048 --k=2048 --l=2 --mode=0

      Runs the mixed input gemm, and applies a scaling factor to B before mma (mode 1). Applies a vector of scales to the entire
      matrix (group size is the same as the gemm k dimension).
      $ ./examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm --m=4096 --n=5120 --k=8192 --g=8192 --mode=1
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/mixed_dtype_utils.hpp"

#include "cutlass/bfloat16.h"

#include "mixed_dtype_utils.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

#include "../common/helper.h"


using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace cutlass {

template <int N>
struct plus<Array<bfloat16_t, N>> {
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(Array<bfloat16_t, N> const & lhs, 
        Array<bfloat16_t, N> const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        __nv_bfloat162 const *lhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&lhs);
        __nv_bfloat162 const *rhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&rhs);
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hadd2(lhs_ptr[i], rhs_ptr[i]);
        }
        
        if constexpr (N % 2) {
            __nv_bfloat16 const *a_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&lhs);
            __nv_bfloat16 const *b_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&rhs);
            __nv_bfloat16 d_residual
                = __hadd(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);
            
            result[N - 1] = reinterpret_cast<bfloat16_t const &>(d_residual);
        }
        
#else
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = lhs[i] + rhs[i];
        }
#endif
        
        return result;
    }
        
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(bfloat16_t const & lhs,
        Array<bfloat16_t, N> const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        __nv_bfloat162 lhs_pair
            = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16 const &>(lhs));
        __nv_bfloat162 const *rhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&rhs);
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hadd2(lhs_pair, rhs_ptr[i]);
        }
        
        if constexpr (N % 2) {
            __nv_bfloat16 const *b_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&rhs);
            __nv_bfloat16 d_residual
                = __hadd(reinterpret_cast<__nv_bfloat16 const &>(lhs), 
                b_residual_ptr[N - 1]);
            
            result[N - 1] = reinterpret_cast<bfloat16_t const &>(d_residual);
        }
        
#else
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
              result[i] = lhs + rhs[i];
        }
#endif
        
        return result;
    }
        
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(Array<bfloat16_t, N> const & lhs, 
        bfloat16_t const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        __nv_bfloat162 const *lhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&lhs);
        __nv_bfloat162 rhs_pair
            = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16 const &>(rhs));
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hadd2(lhs_ptr[i], rhs_pair);
        }
        
        if constexpr (N % 2) {
            __nv_bfloat16 const *a_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&lhs);
            __nv_bfloat16 d_residual = __hadd(a_residual_ptr[N - 1], 
                    reinterpret_cast<__nv_bfloat16 const &>(rhs));
            
            result[N - 1] = reinterpret_cast<bfloat16_t const &>(d_residual);
        }
        
#else
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = lhs[i] + rhs;
        }
#endif
        
        return result;
    }
};

template <int N>
struct multiplies<Array<bfloat16_t, N>> {
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(Array<bfloat16_t, N> const & lhs, 
        Array<bfloat16_t, N> const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        __nv_bfloat162 const *lhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&lhs);
        __nv_bfloat162 const *rhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&rhs);
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hmul2(lhs_ptr[i], rhs_ptr[i]);
        }
    
        if constexpr (N % 2) {
            __nv_bfloat16 const *a_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&lhs);
            __nv_bfloat16 const *b_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&rhs);
            __nv_bfloat16 d_residual = __hmul(a_residual_ptr[N - 1],
                b_residual_ptr[N - 1]);
    
            result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
        }
    
#else
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = lhs[i] * rhs[i];
        }
#endif
    
        return result;
    }
    
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(bfloat16_t const & lhs,
        Array<bfloat16_t, N> const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        _nv_bfloat162 lhs_pair
            = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16 const &>(lhs));
        __nv_bfloat162 const *rhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&rhs);
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hmul2(lhs_pair, rhs_ptr[i]);
        }
    
        if constexpr (N % 2) {
        __nv_bfloat16 const *b_residual_ptr
            = reinterpret_cast<__nv_bfloat16 const *>(&rhs);
    
        __nv_bfloat16 d_residual = __hmul(
            reinterpret_cast<__nv_bfloat16 const &>(lhs),
            b_residual_ptr[N - 1]);
    
        result[N - 1] = reinterpret_cast<bfloat16_t const &>(d_residual);
    }
    
#else
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = lhs * rhs[i];
        }
#endif
    
        return result;
    }
    
    CUTLASS_HOST_DEVICE
    Array<bfloat16_t, N> operator()(Array<bfloat16_t, N> const & lhs, 
        bfloat16_t const &rhs) const {
        Array<bfloat16_t, N> result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    
        __nv_bfloat162 *result_ptr = reinterpret_cast<__nv_bfloat162 *>(&result);
        __nv_bfloat162 const *lhs_ptr
            = reinterpret_cast<__nv_bfloat162 const *>(&lhs);
        __nv_bfloat162 rhs_pair
            = __bfloat162bfloat162(reinterpret_cast<__nv_bfloat16 const &>(rhs));
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_ptr[i] = __hmul2(lhs_ptr[i], rhs_pair);
        }
    
        if constexpr (N % 2) {
            __nv_bfloat16 const *a_residual_ptr
                = reinterpret_cast<__nv_bfloat16 const *>(&lhs);
    
            __nv_bfloat16 d_residual = __hmul(
                a_residual_ptr[N - 1],
                reinterpret_cast<__half const &>(rhs));
    
            result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
        }
    
#else
    
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = lhs[i] * rhs;
        }
#endif
    
        return result;
    }
};

template <>
struct atomic_add<bfloat16_t>
{
  CUTLASS_DEVICE
  void operator()(bfloat16_t *ptr, const bfloat16_t &data)
  {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nv_bfloat16 *const atom_ptr = reinterpret_cast<__nv_bfloat16 *>(ptr);
    const __nv_bfloat16 atom_data = reinterpret_cast<const __nv_bfloat16 &>(data);
    atomicAdd(atom_ptr, atom_data);
#else
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(data);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

template<>
struct atomic_add<__nv_bfloat162>
{
    CUTLASS_DEVICE
    void operator()(__nv_bfloat162 *ptr, const __nv_bfloat162 &data)
    {
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__)  && (__CUDA_ARCH__ < 900))
        CUTLASS_UNUSED(ptr);
        CUTLASS_UNUSED(data);
        CUTLASS_NOT_IMPLEMENTED();
#else
        // Vector-2 atomic reduction requires .target sm_60 or higher
        uint32_t word = reinterpret_cast<const uint32_t&>(data);
        asm volatile ("red.gpu.global.add.noftz.bf16x2 [%0], %1;\n" : : "l"(ptr), "r"(word));
#endif // (__CUDA_ARCH__ >= 600)
    }
};

template <
  int BlockThreads,
  typename ArrayT>
struct BlockStripedReduce<BlockThreads, ArrayT, bfloat16_t> :
  BlockStriped<
    BlockThreads,
    ArrayT,
    __nv_bfloat162>
{
  static_assert(BlockStripedReduce::kStripes % 2 == 0, "Array of bfloat16 must be even number in length");

  /// Reduce
  CUTLASS_DEVICE
  static void reduce(ArrayT *ptr, const ArrayT &data, int thread_idx)
  {
    cutlass::atomic_add<__nv_bfloat162> reduce;
    __nv_bfloat162 *access_output = reinterpret_cast<__nv_bfloat162*>(ptr);
    const __nv_bfloat162 *access_data = reinterpret_cast<const __nv_bfloat162*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < BlockStripedReduce::kStripes; ++i)
    {
      reduce(access_output + (BlockThreads * i) + thread_idx, access_data[i]);
    }
  }
};
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM A input type
using MmaType = cutlass::bfloat16_t;
// GEMM B input type
using QuantType = cutlass::int4b_t;

// A matrix configuration

// Element type for A matrix operand
using ElementA = MmaType;
// Layout type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;
// Layout type for A^T matrix operand
using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
// Memory access alignment of A matrix in units of elements
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;
// Stride type for A matrix operand
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;

// B matrix configuration

// Element type for B matrix operand
using ElementB = QuantType;
// Layout type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor;
// Memory access alignment of B matrix in units of elements
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;
// Layout type for B^T matrix operand
using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
// Stride type for B matrix operand
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in contiguous locations in global memory.
// It specifies the reordering within a single warp's fragment
//using ValueShuffle = Layout<_1>;                          // no value reordering
using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>; // order [0,2,4,6,1,3,5,7]
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;

// Reordered Atomic Layout for MMA computation
using LayoutAtomQuant
    = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, 
    ValueShuffle>());
  // Reordered B Layout for MMA computation
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, 
    Layout<Shape<int,int,int>, StrideB>{}));

// C matrix configuration

// Element type for C matrix operands: unused
using ElementC = cutlass::bfloat16_t;
// Layout type for C matrix operands
using LayoutC = cutlass::layout::RowMajor;
// Memory access alignment of C matrix in units of elements
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
// Layout type for C^T matrix operand
using LayoutC_T = typename cutlass::layout::LayoutTranspose<LayoutC>::type;

// D matrix configuration
// Element type for D matrix operands: unused
using ElementD = cutlass::bfloat16_t;
// Layout type for D matrix operands
using LayoutD = cutlass::layout::RowMajor;
// Memory access alignment of D matrix in units of elements
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
// Layout type for C^T matrix operand
using LayoutD_T = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// A scale vector configuration

// Element type for A scale operand
using ElementAScale = cutlass::bfloat16_t;
// Memory access alignment of A scale in units of elements
constexpr int AlignmentAScale = 128 / sizeof_bits_v<ElementAScale>;
// Stride type for A scale operand
using StrideAScale = Stride<_0,_1,_0>;
using StrideAux = Stride<_0,_1,_0>;

// B scale vector configuration

// Element type for B scale operand
using ElementBScale = MmaType;
// Layout type for B scale operand
using LayoutBScale = cutlass::layout::RowMajor;

// Core kernel configurations

// Element type for internal accumulation
using ElementAccumulator  = float;
// Element type for epilogue computation
using ElementCompute      = cutlass::bfloat16_t;

// Tag indicating the minimum SM that supports the intended feature
using ArchTag = cutlass::arch::Sm90;
// Operator class tag
using OpClass = cutlass::arch::OpClassTensorOp;
// Kernel to launch based on the default setting in the Collective Builder 
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// Epilogue to launch based on the default setting in the Collective Builder 
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;
// Epilogue subtile shape
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

// Rounding Style on Epilogue Computation
static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

template<class TileShape>
using HiddenState = cutlass::epilogue::fusion::Sm90EVT< // ascale * acc + C
    cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, 
        ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementAScale, 
        ElementCompute, StrideAScale, AlignmentAScale>,// ascale
    cutlass::epilogue::fusion::Sm90AccFetch, // acc,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementCompute> // C
>;

template<class TileShape>
using DotProduct = cutlass::epilogue::fusion::Sm90EVT<//sigma((ascale*acc)^2>
    cutlass::epilogue::fusion::Sm90RowReduction<cutlass::plus, cutlass::plus, 
        cutlass::atomic_add, 0, TileShape, ElementCompute, ElementCompute, 
        RoundStyle>,
    cutlass::epilogue::fusion::Sm90EVT<// (ascale * acc) ^ 2
        cutlass::epilogue::fusion::Sm90Compute<cutlass::magnitude_squared, 
            ElementCompute, ElementCompute, RoundStyle>,
        cutlass::epilogue::fusion::Sm90SplitTreeFetch
    >
>;

template<class TileShape>
using FusionOps = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
    HiddenState<TileShape>,
    DotProduct<TileShape>
>;

// Collective Epilogue
template<class TileShape, class ClusterShape>
using CollectiveEpilogue = typename cutlass::epilogue::collective
  ::CollectiveBuilder<ArchTag, OpClass, TileShape, ClusterShape, 
  EpilogueTileType, ElementAccumulator, ElementCompute, ElementC, LayoutC_T, 
  AlignmentC, ElementD, LayoutD_T, AlignmentD, EpilogueSchedule, 
  FusionOps<TileShape>>::CollectiveOp;

// Collective Mainloop
template<class TileShape, class ClusterShape>
using CollectiveMainloop = typename cutlass::gemm::collective
    ::CollectiveBuilder<ArchTag, OpClass, cute::tuple<ElementB, 
    ElementBScale>, LayoutB_Reordered, AlignmentB, ElementA, 
    LayoutA_T, AlignmentA, ElementAccumulator, TileShape, ClusterShape, 
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(
    typename CollectiveEpilogue<TileShape, ClusterShape>::SharedStorage))>, 
    KernelSchedule>::CollectiveOp;

// Kernel tile schecduler of kernel
using TileScheduler = cutlass::gemm::PersistentScheduler;

// GEMM kernel
template<class TileShape, class ClusterShape>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop<TileShape, ClusterShape>,
    CollectiveEpilogue<TileShape, ClusterShape>, TileScheduler>;
// Kernel Parameters
using Params = cutlass::gemm::kernel::detail
  ::PersistentTileSchedulerSm90::Params;
// Kernel Rasterization option
using RasterOrderOptions = Params::RasterOrderOptions;

// Device level kernel
template<class TileShape, class ClusterShape>
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel<TileShape, 
    ClusterShape>>;

// Stride Type of C
using StrideC = typename GemmKernel<Shape<_64, _256, _64>,
    Shape<_2,_1,_1>>::StrideC;
// Stride Type of D
using StrideD = typename GemmKernel<Shape<_64, _256, _64>,
    Shape<_2,_1,_1>>::StrideD;
// Stride Type of B Scale
using StrideBScale = typename CollectiveMainloop<Shape<_64, _256, _64>, Shape<_2,_1,_1>>::StrideScale;

#endif


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
// K tile size
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

struct INT4BF16SquareDot {
    // Threadblock-level tile size
    using TileShape = Shape<_64, _256, cute::Int<TileShapeK>>;
    // Shape of the threadblocks in a cluster
    using ClusterShape = Shape<_2,_1,_1>;

    Gemm<TileShape, ClusterShape> gemm;

    cutlass::KernelHardwareInfo hw_info;
    using Args = typename Gemm<TileShape, ClusterShape>::GemmKernel::Arguments;

    int m;
    int n;
    int k;
    int scale_k;
    int l;
    int g;

    cutlass::DeviceAllocation<ElementB> block_B;
    cutlass::DeviceAllocation<ElementBScale> block_Bscale;

    cutlass::DeviceAllocation<ElementD> block_D;
    LayoutB_Reordered layout_B_reordered;

    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;
    StrideAScale stride_Ascale;
    StrideBScale stride_Bscale;
    StrideAux stride_aux;

    bool initialized;

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace;

    INT4BF16SquareDot(void *weights_, void *weight_scales_,
        int input_dim, int output_dim, int group_size) : n(output_dim),
        k(input_dim), scale_k(ceil_div(input_dim, group_size)), g(group_size),
        block_B((ElementB *)weights_, n * k),
        block_Bscale((ElementBScale *)weight_scales_,
        n * scale_k) {
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo
            ::query_device_multiprocessor_count(hw_info.device_id);
    
        auto shape_B = cute::make_shape(n, k, 1);
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
        auto layout_B = make_layout(shape_B, stride_B);
    
        stride_Bscale = cutlass::make_cute_packed_stride(StrideBScale{}, 
            cute::make_shape(n, scale_k, 1));
            
        layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
        cutlass::reorder_tensor(block_B.get(), layout_B, layout_B_reordered);

        initialized = false;
        l = -1;
        m = -1;
    }

    ~INT4BF16SquareDot() {
        block_B.release();
        block_Bscale.release();
    }

    void *forward(void *inputs_, void *input_scales_, void *residuals_,
        void *outputs_, void *square_sum, int batch_size, int seq_len, 
        cudaStream_t stream) {
        l = batch_size;
        m = seq_len;

        stride_A = cutlass::make_cute_packed_stride(StrideA{}, 
            cute::make_shape(m * l, k, 1));
        // Reverse stride here due to swap and transpose
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, 
            cute::make_shape(n, m * l, 1));
        // Reverse stride here due to swap and transpose
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, 
            cute::make_shape(n, m * l, 1));

        Args arguments{
            cutlass::gemm::GemmUniversalMode::kBatched,
            {n, m * l, k, 1},
            {block_B.get(), layout_B_reordered, (ElementA *)inputs_, stride_A, 
              block_Bscale.get(), stride_Bscale, g},
            {
                {// fusion_args
                    {
                        {(ElementAScale *)input_scales_, ElementAScale(0.0f), stride_Ascale}, // A scale
                        {}, // acc
                        {}, // C
                        {} // homogeneous_multiply_add
                    },
                    {
                        {
                            {}, // hidden_state
                            {} // magnitude squared
                        }, // hidden_state^2
                        {(ElementCompute *)square_sum, ElementCompute(0.0f)} // add reduction
                    }
                },
                (ElementC *)residuals_, stride_C, (ElementD *)outputs_, stride_D
            },
            hw_info
        };
        
        auto &tile_scheduler = arguments.scheduler;
        tile_scheduler.raster_order = RasterOrderOptions::Heuristic;
        tile_scheduler.max_swizzle_size = 8;

        if(!initialized) {
            // Using the arguments, query for extra workspace required for 
            // matrix multiplication computation
            size_t workspace_size = Gemm<TileShape, ClusterShape>
                ::GemmKernel::get_workspace_size(arguments);

            // Allocate workspace memory
            workspace.reset(workspace_size);

            // Check if the problem size is supported or not
            CUTLASS_CHECK(gemm.can_implement(arguments));

            // Initialize CUTLASS kernel with arguments and workspace pointer
            CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));

            initialized = true;
        }
        else
            CUTLASS_CHECK(gemm.update(arguments, workspace.get()));

        CUTLASS_CHECK(gemm.run(stream));
    }
};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
