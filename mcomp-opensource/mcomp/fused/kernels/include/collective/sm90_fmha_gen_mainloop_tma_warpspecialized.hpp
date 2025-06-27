/***************************************************************************************************
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include <cuda_fp16.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"

#include "algorithm/fmha_copy.hpp"
#include "algorithm/fmha_dropout.hpp"
#include "algorithm/fmha_scale.hpp"
#include "algorithm/fmha_softmax.hpp"
#include "collective/fmha_mask.hpp"
#include "collective/fmha_varlen.hpp"
#include "collective/fmha_mixed_input_utils.hpp"

namespace cutlass::fmha {

using namespace cute;


namespace collective {

/**
 * @brief Sm90 FMHA Forward Mainloop TMA Warp Specialized
 * 
 * @tparam TileShape_HGKVD Tile shape type (CTA_HG, CTA_KV, CTA_D)
 * @tparam ElementQ_ Type of element of Q to compute
 * @tparam ElementK_ Type of element of K to compute
 * @tparam ElementV_ Type of element of V to compute
 */
template <class TileShape_HGKVD_, class ClusterShape_, class ElementQ_,
    class ElementK_, class ElementV_>
struct Sm90FmhaGenMainloopTmaWarpspecialized {
private:
using Utils = detail::MixedInputUtils<Sm90FmhaGenMainloopTmaWarpspecialized>;
public:
    // type of element of Q to compute
    using ElementQ = cutlass::float_e4m3_t;
    // Q Global memory Tag: row major
    using GmemLayoutQTag = cutlass::layout::RowMajor;
    // number of elements of Qin alignment of 16 bytes
    static const int AlignmentQ = 128 / sizeof_bits_v<ElementQ>;
    // type of element of K to compute
    using ElementK = cutlass::int4b_t;
    // K Global memory Tag: column major
    using GmemLayoutKTag = cutlass::layout::ColumnMajor;
    // number of elements of K in alignment of 16 bytes
    static const int AlignmentK = 128 / sizeof_bits_v<ElementK>;
    // type of element of P to compute
    using ElementP = cutlass::float_e4m3_t;
    // type of element of V to compute
    using ElementV = cutlass::int4b_t;
    using ElementScale = cutlass::Array<float_e4m3_t, 8>;
    // V Global memory Tag: row major
    using GmemLayoutVTag = cutlass::layout::RowMajor;
    // number of elements of K in alignment of 16 bytes
    static const int AlignmentV = 128 / sizeof_bits_v<ElementV>;
    // type of element to accumulate QK^T
    using ElementAccumulatorS = cutlass::half_t;
    // type of element to accumulate PV
    using ElementAccumulatorO = float;
    // tile shape type
    using TileShape = TileShape_HGKVD_;
    // thread block custer shape
    using ClusterShape = ClusterShape_;
    // two stage implmenentation
    using StageCount = cutlass::gemm::collective::StageCount<2>;
    // Pingpong Scheduling
    using KernelScheduleType = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

    // type of division inside CTA
    using AtomLayoutQKVD = Shape<_1, _2, _1>;
    // type of tile shape of QK compute: (HG, KV / 2, D)
    using TileShapeQK = decltype(shape_div(TileShape{}, AtomLayoutQKVD{}));
    // type of tile shape of QK compute: (HG, D, KV / 2)
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));
    // type of Tiled MMA QK
    using TiledMmaQK = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
        ElementQ, ElementQ, ElementAccumulatorS, TileShapeQK, 
        cute::GMMA::Major::K, cute::GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}));
    // type of Tiled MMA PV
    using TiledMmaPV = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
        ElementP, ElementP, ElementAccumulatorO, TileShapePV, 
        cute::GMMA::Major::K, cute::GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}));
    
    // Q Global memory tiled copy type
    using GmemTiledCopyQ = cute::SM90_TMA_LOAD_MULTICAST;
    // K Global memory tiled copy type
    using GmemTiledCopyK = cute::SM90_TMA_LOAD;
    // V Global memory tiled copy type
    using GmemTiledCopyV = cute::SM90_TMA_LOAD;
    // Scale K Global memory tiled copy type
    using GmemTiledCopyScaleK = cute::SM90_TMA_LOAD;
    // Scale K Global memory tiled copy type
    using GmemTiledCopyScaleV = cute::SM90_TMA_LOAD;
    // Shared memory K Copy Atom
    using SmemCopyAtomK = Copy_Atom<cute::AutoVectorizingCopy, ElementK>;
    // Shared memory K Copy Atom
    using GmmaSmemCopyAtomK = Copy_Atom<cute::AutoVectorizingCopy, ElementP>;

    // Tile along modes in a way that maximizes the TMA box size.

    // atomic shared memory layout for Q
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail
        ::ss_smem_selector<GMMA::Major::K, ElementQ, decltype(cute::get<0>(
        TileShapeQK{})), decltype(cute::get<2>(TileShapeQK{}))>());
    // atomic shared memory layout for K
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail
        ::ss_smem_selector<GMMA::Major::K, ElementK, decltype(cute::get<1>(
        TileShapeQK{})), decltype(cute::get<2>(TileShapeQK{}))>());
    // atomic shared memory layout for V
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail
        ::rs_smem_selector<GMMA::Major::MN, ElementV, decltype(cute::get<1>(TileShapePV{})), decltype(cute::get<2>(TileShapePV{}))>());
    // atomic shared memory layout for K
    using GmmaSmemLayoutAtomK = decltype(cutlass::gemm::collective::detail
        ::ss_smem_selector<GMMA::Major::K, ElementP, decltype(cute::get<1>(
        TileShapeQK{})), decltype(cute::get<2>(TileShapeQK{}))>());
    // atomic shared memory layout for VT
    using GmmaSmemLayoutAtomV = decltype(transform::collective::detail::gmma_smem_transpose_or_passthrough<
        true, SmemLayoutAtomV, ElementP>());
    // atomic shared memory layout for scale of K
    using SmemLayoutAtomScaleK = Layout<Shape<decltype(cute::shape<0>(
        SmemLayoutAtomK{})), cute::Int<1>>>;
    // atomic shared memory layout for scale of V
    using SmemLayoutAtomScaleV = Layout<Shape<cute::Int<1>, decltype(
        cute::shape<0>(SmemLayoutAtomV{}))>>;
    // tile shape of scale (64, 1)
    using ScaleTileShapeK = decltype(make_shape(shape<0>(TileShape{}),
        shape<1>(SmemLayoutAtomScaleK{})));
    using ScaleTileShapeV = decltype(make_shape(shape<1>(
        SmemLayoutAtomScaleK{}), shape<0>(TileShape{})));

    // stride type of Q: (D, 1, (H_G*D, H_G*H_R*D))
    using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;
    // stride type of cache K: (D, 1, (KV*D, P*KV*D, H_R*P*KV*D))
    using StrideCacheK = cute::tuple<int, _1, cute::tuple<int, int, int>>;
    // stride type of cache V: (D, 1, (KV*D, P*KV*D, H_R*P*KV*D))
    using StrideCacheV = StrideK;
    // These are always MN major (1, 0, (KV, P*KV, H_R*P*KV))
    using StrideScale = cute::tuple<cute::Int<1>, int64_t,
        cute::tuple<int, int, int>>;

    // TMA Element for subbyte arrays
    using TmaElementKV = int8_t;
    // in case we have array. translating to uint to satisfy tma descriptor's 
    // specialization
    using TmaElementScale = uint_bit_t<sizeof_bits_v<ElementScale>>;

    // number of shared memory stages for Q load
    static constexpr int StageCountQ = 1;
    // number of shared memory stages for K load
    static constexpr int StageCountKV = 2;

    // Pipeline states for Q load
    using PipelineStateQ = cutlass::PipelineState<StageCountQ>;
    // Pipeline states for K, V load
    using PipelineStateKV = cutlass::PipelineState<StageCountKV>;
    
    // type of shared memory stages for K V load
    using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;

    // type of shared memory layout of Q
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{},
        make_shape(shape<0>(TileShapeQK{}), shape<2>(TileShapeQK{}), 
        Int<StageCountQ>{}), Step<_1,_2,_3>{}));
    // type of shared memory layout of K
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        make_shape(shape<1>(TileShapeQK{}), shape<2>(TileShapeQK{}), 
        Int<StageCountKV>{}), Step<_1,_2,_3>{}));
    // type of shared memory layout of V
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
        make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{}),  
        Int<StageCountKV>{}), Step<_2,_1,_3>{}));
    // SmemLayoutK for GMMA is different from SmemLayoutK for TMA
    using GmmaSmemLayoutK = decltype(tile_to_shape(GmmaSmemLayoutAtomK{},
        make_shape(shape<1>(TileShapeQK{}), shape<2>(TileShapeQK{}),  
        Int<StageCountKV>{}), Step<_1,_2,_3>{}));
    // SmemLayoutV for GMMA is different from SmemLayoutV for TMA
    using GmmaSmemLayoutV = decltype(tile_to_shape(GmmaSmemLayoutAtomV{},
        make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{}),  
        Int<StageCountKV>{}), Step<_1,_2,_3>{}));
    // It is assumed that the scales and zero-points share the same smem layout
    // type of shared memory layout of scale K
    using SmemLayoutScaleK = decltype(tile_to_shape(SmemLayoutAtomScaleK{}, 
        make_shape(shape<0>(ScaleTileShapeK{}), shape<1>(ScaleTileShapeK{}), 
        Int<StageCountKV>{}), Step<_1,_2,_3>{}));
    // type of shared memory layout of scale V
    using SmemLayoutScaleV = decltype(tile_to_shape(SmemLayoutAtomScaleV{}, 
        make_shape(shape<0>(ScaleTileShapeV{}), shape<1>(ScaleTileShapeV{}), 
        Int<StageCountKV>{}), Step<_2,_1,_3>{}));

    static constexpr size_t SmemAlignmentQ = cutlass::detail
        ::alignment_for_swizzle(SmemLayoutQ{}); 

    static constexpr size_t SmemAlignmentK = cutlass::detail
        ::alignment_for_swizzle(SmemLayoutK{});

    static constexpr size_t SmemAlignmentV = cutlass::detail
        ::alignment_for_swizzle(SmemLayoutV{});
      
    // Just pick the max alignment of Q and K since it is required to be at 
    // least 128B
    static constexpr size_t SmemAlignmentScaleK = cute::max(SmemAlignmentQ, 
        SmemAlignmentK);
    // Just pick the max alignment of Q and V since it is required to be at 
    // least 128B
    static constexpr size_t SmemAlignmentScaleV = cute::max(SmemAlignmentQ, 
        SmemAlignmentV);
    
    // V matrix transpose operation
    using TransposeOperandV = decltype(cute::AsyncTranspositionINT4OperandB(0, 
        0, TiledMmaPV{}, SmemLayoutV{}, SmemLayoutAtomV{}, ElementV{},
        ElementP{})); 

    // from load to mma warp, protects q in smem
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<StageCountQ>;
    // from load to mma warp, protects k in smem
    using MainloopPipelineK = cutlass::PipelineTmaAsync<StageCountKV>;
    // from load to mma warp, protects v in smem
    using MainloopPipelineV = cutlass::PipelineTmaAsync<StageCountKV>;
    // from load to mma warp, protects scale k in smem
    using MainloopPipelineSK = cutlass::PipelineTmaAsync<StageCountKV>;
    // from load to mma warp, protects scale v in smem
    using MainloopPipelineSV = cutlass::PipelineTmaAsync<StageCountKV>;
    
    /**
     * @brief Shared Memory Storage
     */
    struct SharedStorage {
        static constexpr int scale_elements_k
            = Utils::elements_per_smem_scale_k();
        static constexpr int scale_elements_v
            = Utils::elements_per_smem_scale_v();
        /**
         * @brief Tensor storage inside shared memory
         */
        struct TensorStorage : cute::aligned_struct<128, _0> {
            // shared memory allocation of Q
            cute::array_aligned<ElementQ, cute::cosize_v<SmemLayoutQ>> smem_q;
            // shared memory allocation of K
            cute::array_aligned<ElementK, cute::cosize_v<SmemLayoutK>> smem_k;
            // shared memory allocation of dequantized K
            cute::array_aligned<ElementQ, cute::cosize_v<GmmaSmemLayoutK>> 
                smem_kd;
            // shared memory allocation of V
            cute::array_aligned<ElementV, cute::cosize_v<SmemLayoutV>> smem_v;
            // shared memory allocation of dequantized V
            cute::array_aligned<ElementQ, cute::cosize_v<GmmaSmemLayoutV>> 
                smem_vt;
            // shared memory allocation of Scale K
            cute::ArrayEngine<ElementScale, scale_elements_k> smem_scale_k;
            // shared memory allocation of Scale V
            cute::ArrayEngine<ElementScale, scale_elements_v> smem_scale_v;
        } tensors;
        // Storage of Load Q Pipeline type
        using PipelineQStorage = typename MainloopPipelineQ::SharedStorage;
        // Storage of Load K Pipeline type
        using PipelineKStorage = typename MainloopPipelineK::SharedStorage;
        // Storage of Load V Pipeline type
        using PipelineVStorage = typename MainloopPipelineV::SharedStorage;
        // Storage of Load Scale K Pipeline type
        using PipelineSKStorage = typename MainloopPipelineSK::SharedStorage;
        // Storage of Load Scale V Pipeline type
        using PipelineSVStorage = typename MainloopPipelineSV::SharedStorage;
        // Storage of Load Q Pipeline
        PipelineQStorage pipeline_q;
        // Storage of Load K Pipeline
        PipelineKStorage pipeline_k;
        // Storage of Load V Pipeline
        PipelineVStorage pipeline_v;
        // Storage of Load Scale K Pipeline
        PipelineSKStorage pipeline_scale_k;
        // Storage of Load Scale V Pipeline
        PipelineSVStorage pipeline_scale_v;
    };
    // Tensor storage inside shared memory
    using TensorStorage = typename SharedStorage::TensorStorage;
    // Storage of Load Q Pipeline type
    using PipelineQStorage = typename SharedStorage::PipelineQStorage;
    // Storage of Load K Pipeline type
    using PipelineKStorage = typename SharedStorage::PipelineKStorage;
    // Storage of Load V Pipeline type
    using PipelineVStorage = typename SharedStorage::PipelineVStorage;
    // Storage of Load Scale K Pipeline type
    using PipelineSKStorage = typename SharedStorage::PipelineSKStorage;
    // Storage of Load Scale V Pipeline type
    using PipelineSVStorage = typename SharedStorage::PipelineSVStorage;

    using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<2, 2>;

    /**
     * @brief Host side kernel arguments
     */
    struct Arguments {
        // pointer of Q
        const ElementQ* ptr_Q;
        // stride of Q
        StrideQ dQ;
        // pointer of K
        ElementQ* ptr_cache_K;
        // stride of K
        StrideCacheK dCacheK;
        // pointer of V
        ElementQ* ptr_cache_V;
        // stride of V
        StrideCacheV dCacheV;
        // pointer of scale K
        ElementScale const* ptr_SK;
        // stride of scale K
        StrideScale dSK;
        // pointer of scale V
        ElementScale const* ptr_SV;
        // stride of scale V
        StrideScale dSV;
        
        // if zero, defaults to 1/sqrt(D)
        double scale_softmax = 0.0;

        // scaling factors to dequantize QKV

        // scaling factors to dequantize Q
        double scale_q = 1.0;
        // scaling factors to dequantize K
        double scale_k = 1.0;
        // scaling factors to dequantize V
        double scale_v = 1.0;

        // scaling factor to quantize P
        double scale_p = 448.0;

        // seed for pseudo-random generator
        const unsigned long long seed = 0;
        // offset for pseudo-random generator
        const unsigned long long offset = 0;
        // dropout probability
        double drop_p = 0.0;

        // scaling factor to quantize O
        double inv_scale_o = 1.0;
    };

    static constexpr int nrows = size<0>(TileShapeQK{}) / 64;

    /**
     * @brief Device side kernel params
     */
    struct Params {
    public:
        // Assumption: StrideQ is congruent with Problem_QD
        using LayoutQ = decltype(detail::get_gmem_layout(repeat_like(StrideQ{}, 
            int32_t(0)), StrideQ{}));
        // Assumption: StrideK is congruent with Problem_KVD
        using LayoutK = decltype(detail::get_gmem_layout(repeat_like(
            StrideCacheK{}, int32_t(0)), StrideCacheK{}));
        // Assumption: StrideV is congruent with Problem_KVD
        using LayoutV = decltype(detail::get_gmem_layout(repeat_like(
            select<1,0,2>(StrideCacheV{}), int32_t(0)),
            select<1,0,2>(StrideCacheV{})));

        // Assumption: StrideQ is congruent with Problem_QD
        using TMA_Q = decltype(make_tma_copy_A_sm90(GmemTiledCopyQ{},
            make_tensor(detail::get_logical_ptr(static_cast<ElementQ const *>(
            nullptr)), LayoutQ{}), SmemLayoutQ{}(_,_,cute::Int<0>{}), 
            TileShapeQK{}, ClusterShape{}));
        // Assumption: StrideK is congruent with Problem_KVD
        using TMA_K = decltype(make_tma_copy_B_sm90<TmaElementKV>(
            GmemTiledCopyK{}, make_tensor(detail::get_logical_ptr(
            static_cast<ElementK const *>(nullptr)), LayoutK{}),
            SmemLayoutK{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{}));
        // Assumption: StrideV is congruent with Problem_KVD
        using TMA_V = decltype(make_tma_copy_B_sm90<TmaElementKV>(
            GmemTiledCopyV{}, make_tensor(detail::get_logical_ptr(
            static_cast<ElementV const *>(nullptr)), LayoutV{}),
            SmemLayoutV{}(_,_,cute::Int<0>{}), TileShapePV{}, ClusterShape{}));

        // mcast along N mode for this M load, if any.
        // Scale is ALWAYS loaded with A for RF kernel
        using TMA_Scale_K = decltype(make_tma_copy<TmaElementScale>(
            GmemTiledCopyScaleK{}, make_tensor(detail::get_logical_ptr(
            static_cast<ElementScale const*>(nullptr)), repeat_like(
            StrideScale{}, int32_t(0)), StrideScale{}), SmemLayoutScaleK{}(_,_,
            cute::Int<0>{}), ScaleTileShapeK{}, _1{}));
        // mcast along N mode for this M load, if any.
        // Scale is ALWAYS loaded with A for RF kernel
        using TMA_Scale_V = decltype(make_tma_copy<TmaElementScale>(
            GmemTiledCopyScaleV{}, make_tensor(detail::get_logical_ptr(
            static_cast<ElementScale const*>(nullptr)), repeat_like(
            StrideScale{}, int32_t(0)), StrideScale{}), SmemLayoutScaleV{}(_,_,
            cute::Int<0>{}), ScaleTileShapeV{}, _1{}));

        // TMA Q load descriptor
        TMA_Q tma_load_q;
        // TMA K load descriptor
        TMA_K tma_load_k;
        // TMA V load descriptor
        TMA_V tma_load_v;
        // TMA Scale K load descriptor
        TMA_Scale_K tma_load_scale_k;
        // TMA Scale V load descriptor
        TMA_Scale_V tma_load_scale_v;
        // scale structure device params
        typename cute::Scale::Params scale;
        // softmax structure device params
        SoftmaxParams softmax;
        // dropout structure device params
        typename cute::Dropout::Params dropout;
        // total TMA trnasaction bytes
        uint32_t tma_transaction_bytes = TmaTransactionBytes;
        // Q TMA trnasaction bytes
        uint32_t tma_transaction_bytes_q = TmaTransactionBytesQD;
        // K TMA trnasaction bytes
        uint32_t tma_transaction_bytes_k = TmaTransactionBytesKD;
        // V TMA trnasaction bytes
        uint32_t tma_transaction_bytes_v = TmaTransactionBytesVD;
        // Q stride
        StrideQ dQ;
        // Cache K stride
        StrideCacheK dCacheK;
        // Cache V stride
        StrideCacheV dCacheV;
    };

    /**
     * @brief Check the Problem Shape Alignment
     * 
     * @tparam ProblemShape type of problem shape
     * @param problem_shape problem shape (1, KV, D, ((H_G, H_R), B))
     * @param args host side kernel arguments
     * @return true Alignment meet
     * @return false Alignment does not meet
     */
    template<class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape,
        [[maybe_unused]] Arguments const& args) {
        constexpr int tma_alignment_bits = 128;
        auto [Q,KV,D,HB] = problem_shape;
        auto [H, B] = HB;
        auto [HG, HR] = H;

        // check Q alignement
        constexpr int min_tma_aligned_elements_Q = tma_alignment_bits
            / cutlass::sizeof_bits<ElementQ>::value;
        bool check_aligned_Q = cutlass::detail::check_alignment<
            min_tma_aligned_elements_Q>(detail::get_gmem_layout(
            cute::make_shape(HG,D,cute::make_shape(HR, B)), args.dQ));

        // check K alignment
        constexpr int min_tma_aligned_elements_K = tma_alignment_bits
            / cutlass::sizeof_bits<ElementK>::value;
        bool check_aligned_K = cutlass::detail::check_alignment<
            min_tma_aligned_elements_K>(detail::get_gmem_layout(
            cute::make_shape(KV,D,cute::make_shape(HR, B)), args.dCacheK));

        // check V alignment
        constexpr int min_tma_aligned_elements_V = tma_alignment_bits
            / cutlass::sizeof_bits<ElementV>::value;
        bool check_aligned_V = cutlass::detail::check_alignment<
            min_tma_aligned_elements_V>(detail::get_gmem_layout(
            cute::make_shape(KV,D,cute::make_shape(HR, B)), args.dCacheV));
        
        bool check_aligned_SK = true;
        bool check_aligned_SV = true;
        bool check_mode_args = true;

        const int scale_kv = KV;
        const int scale_d = 1;
        constexpr int min_tma_aligned_elements_scale = tma_alignment_bits
            / cutlass::sizeof_bits<ElementScale>::value;
        // check scale K alignment
        check_aligned_SK = cutlass::detail::check_alignment<
            min_tma_aligned_elements_scale>(cute::make_shape(KV,_1{},
            cute::make_shape(HR, B)), args.dSK);
        // check scale V alignment
        check_aligned_SV = cutlass::detail::check_alignment<
            min_tma_aligned_elements_scale>(cute::make_shape(KV,_1{},
            cute::make_shape(HR, B)), args.dSV);
        check_mode_args = check_mode_args && (args.ptr_SK != nullptr);
        check_mode_args = check_mode_args && (args.ptr_SV != nullptr);

        if (!check_mode_args) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Invalid arguments for the "
                "selected conversion mode.\n");
        }
        if (!check_aligned_Q) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor A meet the minimum "
                "alignment requirements for TMA.\n");
        }
        if (!check_aligned_K) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor B meet the minimum "
                "alignment requirements for TMA.\n");
        }
        if (!check_aligned_V) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor B meet the minimum "
                "alignment requirements for TMA.\n");
        }
        if (!check_aligned_SK) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor SK (scale k) meet the "
                "minimum alignment requirements for TMA.\n");
        }
        if (!check_aligned_SV) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor SV (scale v) meet the "
                "minimum alignment requirements for TMA.\n");
        }
        return check_mode_args && check_aligned_Q && check_aligned_K
            && check_aligned_V && check_aligned_SK && check_aligned_SV;
    }

    /**
     * @brief Map user facing arguments to device facing params
     * 
     * @tparam ProblemShape Problem shape type
     * @param problem_shape problem shape
     * @param args kernel arguments
     * @param workspace workspace
     * @return Params kernel arguments
     */
    template<class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const& problem_shape,
        Arguments const& args, void* workspace) {
        auto [HG,KV,D,HB] = problem_shape;
        auto [HR, B] = HB;

        auto ptr_Q = args.ptr_Q;
        auto ptr_K = args.ptr_cache_K;
        auto ptr_V = args.ptr_cache_V;
        auto ptr_SK = args.ptr_SK;
        auto ptr_SV = args.ptr_SV;
        auto dQ = args.dQ;
        auto dCacheK = args.dCacheK;
        auto dCacheV = args.dCacheV;

        Tensor tensor_q = make_tensor(detail::get_logical_ptr(ptr_Q), 
            detail::get_gmem_layout(make_layout(make_shape(HG,D,HB), dQ)));
        Tensor tensor_k = make_tensor(detail::get_logical_ptr(ptr_K), 
            detail::get_gmem_layout(make_layout(make_shape(KV,D,HB),
            dCacheK)));
        Tensor tensor_v = make_tensor(detail::get_logical_ptr(ptr_V), 
            detail::get_gmem_layout(make_layout(make_shape(D,KV,HB),
            select<1,0,2>(dCacheV))));

        auto tma_load_q = make_tma_copy_A_sm90(GmemTiledCopyQ{}, tensor_q, 
            SmemLayoutQ{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{});
        // mcast along M mode for this N load, if any
        auto tma_load_k = make_tma_copy_B_sm90<TmaElementKV>(GmemTiledCopyK{}, 
            tensor_k, SmemLayoutK{}(_,_,cute::Int<0>{}), TileShapeQK{}, 
            ClusterShape{});
        // mcast along M mode for this N load, if any
        auto tma_load_v = make_tma_copy_B_sm90<TmaElementKV>(GmemTiledCopyV{}, 
            tensor_v, SmemLayoutV{}(_,_,cute::Int<0>{}), TileShapePV{}, 
            ClusterShape{});
        
        typename Params::TMA_Scale_K tma_load_scale_k{};
        typename Params::TMA_Scale_V tma_load_scale_v{};

        ElementScale const* ptr_SK = args.ptr_SK;
        StrideScale dSK = args.dSK;
        ElementScale const* ptr_SV = args.ptr_SV;
        StrideScale dSV = args.dSV;
        Tensor tensor_scale_k = make_tensor(detail::get_logical_ptr(ptr_SK), 
            make_layout(make_shape(KV,_1{},L), dSK));
        Tensor tensor_scale_v = make_tensor(detail::get_logical_ptr(ptr_SV), 
            make_layout(make_shape(_1{},KV,L), select<1,0,2>(dSV)));
        tma_load_scale_k = make_tma_copy<TmaElementScale>(GmemTiledCopyScaleK{},
            tensor_scale_k, SmemLayoutScaleK{}(_,_,cute::Int<0>{}),
            ScaleTileShapeK{}, _1{});
        tma_load_scale_v = make_tma_copy<TmaElementScale>(GmemTiledCopyScaleV{},
            tensor_scale_v, SmemLayoutScaleV{}(_,_,cute::Int<0>{}),
            ScaleTileShapeV{}, _1{});

        typename Scale::Arguments scale = {args.scale_softmax, args.scale_q, 
            args.scale_k};
        SoftmaxArguments softmax = {args.drop_p, args.scale_p,
            args.scale_v, args.inv_scale_o};
        typename Dropout::Arguments dropout = {args.seed, args.offset,
            args.drop_p};

        uint32_t transaction_bytes_q = TmaTransactionBytesQD;
        uint32_t transaction_bytes_k = TmaTransactionBytesKD;
        uint32_t transaction_bytes_v = TmaTransactionBytesVD;
        uint32_t transaction_bytes = TmaTransactionBytes;

        return Params{tma_load_q, tma_load_k, tma_load_v, tma_load_scale_k, 
            tma_load_scale_v, Scale::to_underlying_arguments(problem_shape, 
            scale, workspace), to_underlying_softmax_arguments(problem_shape, 
            softmax, workspace), Dropout::to_underlying_arguments(
            problem_shape, dropout, workspace), transaction_bytes, 
            transaction_bytes_q, transaction_bytes_k, transaction_bytes_v, dQ, 
            dCacheK, dCacheV};
    }

    // size of transaction bytes to load Q
    static constexpr int TmaTransactionBytesQD = cutlass::bits_to_bytes(
        size<0>(SmemLayoutQ{}) * size<1>(SmemLayoutQ{})
        * static_cast<uint32_t>(sizeof_bits<ElementQ>::value));
    // size of transaction bytes to load KV
    static constexpr int TmaTransactionBytesKD = cutlass::bits_to_bytes(
        size<0>(SmemLayoutK{}) * size<1>(SmemLayoutK{})
        * static_cast<uint32_t>(sizeof_bits<ElementK>::value));
    // unused
    // size of transaction bytes to load KV
    static constexpr int TmaTransactionBytesVD = cutlass::bits_to_bytes(
        size<0>(SmemLayoutV{}) * size<1>(SmemLayoutV{})
        * static_cast<uint32_t>(sizeof_bits<ElementV>::value));
    // unused
    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesQD
        + TmaTransactionBytesKD + TmaTransactionBytesVD;

    /**
     * @brief Issue Tma Descriptor Prefetch -- ideally from a single thread for 
     * best performance
     * 
     * @param params kernel parameters
     */
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_scale_k
            .get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_scale_v
            .get_tma_descriptor());
    }

    /**
     * @brief Get global tensors to process and trip count
     * 
     * @tparam BlkCoord Coord<int, _0, Coord<int, int>>
     * @tparam ProblemShape Coord<int, int, int, Coord<Coord<int, int>, int>>
     * @tparam ParamsProblemShape Original Problem shape, could be variable len
     * @param blk_coord_in block coordinate (bidq, _0, (bidh, bidb))
     * @param problem_shape problem shape (q, kv, d, ((hg, hr), b))
     * @return cute::tuple<TensorQ, TensorK, TensorV, int> const& global tensor 
     * of Q, K, V, and masked tile count
     */
    template<class BlkCoord, class ProblemShape_QKVDHGHRB,
        class ParamsProblemShape_QKVDHGHRB>
    CUTLASS_DEVICE auto load_init(BlkCoord const& blk_coord_in,
        ProblemShape_QKVDHGHRB const& problem_shape,
        Params const& mainloop_params) const {
        auto [Q,KV,D,HB] = problem_shape;
        auto [H, B] = HB;
        auto [HG, HR] = H;
        auto L = make_shape(HR, B);
        // Q matrix block coordinate
        BlkCoord blk_coord_q = blk_coord_in;
        // K, V matrix block coordinate
        BlkCoord blk_coord_kv = blk_coord_in;
        // number of horizontal trips for compute
        int mask_tile_count = Mask::get_trip_count(blk_coord_in, TileShape{}, 
            ClusterShape{}, problem_shape);
    
        using X = Underscore;
    
        // this one is only executed by one thread, no need to elect_one
    
        // TMA requires special handling of strides to deal with coord codomain 
        // mapping
        // Represent the full tensors -- get these from TMA

        // (grouped_heads, head_dim, (head_kv, batch_size))
        Tensor mQ_hdl = mainloop_params.tma_load_q.get_tma_tensor(shape(
            detail::get_gmem_layout(make_shape(HG,D,L), mainloop_params.dQ)));
        // (seq_kv, head_dim, (head_kv, batch_size))
        Tensor mK_kdl = mainloop_params.tma_load_k.get_tma_tensor(shape(
            detail::get_gmem_layout(make_shape(KV,D,L),
            mainloop_params.dCacheK)));
        // (head_dim, seq_kv, (head_kv, batch_size))
        Tensor mV_dkl = mainloop_params.tma_load_v.get_tma_tensor(
            select<1,0,2>(shape(detail::get_gmem_layout(make_shape(KV,D,L),
            mainloop_params.dCacheV))));
    
        // (BLK_H,BLK_D,h,d,l)
        Tensor gQ_hdl = local_tile(mQ_hdl, TileShapeQK{}, make_coord(_, _, _), 
            Step<_1, X, _1>{});
        // (BLK_KV,BLK_D,kv,d,l)
        Tensor gK_kdl = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), 
            Step<X, _1, _1>{});
        // (BLK_D,BLK_KV,d,kv,l)
        Tensor gV_dkl = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), 
            Step<X, _1, _1>{});

        // (m,1,l)
        Tensor mSK_k1l = mainloop_params.tma_load_scale_k.get_tma_tensor(
            make_shape(KV,_1{},L));
        // (m,1,l)
        Tensor mSV_1kl = mainloop_params.tma_load_scale_v.get_tma_tensor(
            select<1,0,2>(make_shape(KV,_1{},L)));
        // (BLK_KV,BLK_Scale_K,kv,1,l)
        Tensor gSK_k1l = local_tile(mSK_k1l, ScaleTileShapeK{},
            make_coord(_,_));
        // (BLK_Scale_V,BLK_KV,1,kv,l)
        Tensor gSV_1kl = local_tile(mSV_1kl, ScaleTileShapeV{},
            make_coord(_,_));
        int seqlen_cache_kv = get<1>(problem_shape)
            - ((mainloop_params.ptr_new_k != nullptr) ? 1 : 0);
            
        return cute::make_tuple(gQ_hdl, gK_kdl, gV_dkl, gSK_k1l, gSV_1kl, 
            blk_coord_q, blk_coord_kv, mask_tile_count, seqlen_cache_kv);
    }

    /**
     * @brief Perform a collective-scoped matrix multiply-accumulate
     * Producer Perspective
     * 
     * @tparam TensorQ (BLK_Q, BLK_D, q_off_0, 0, (0, q_offs_2_1))
     * @tparam TensorK (BLK_K, BLK_D, kv_off_0, 0, (0, kv_offs_2_1))
     * @tparam TensorV (BLK_D, BLK_V, 0, kv_off_0, (0, kv_offs_2_1))
     * @tparam BlkCoord Coord<int, _0, Coord<int, int>>
     * @tparam Ts scale tensor type
     * @param mainloop_params kernel parameters
     * @param pipeline_q Q load pipeline
     * @param pipeline_k K load pipeline
     * @param pipeline_v V load pipeline
     * @param pipeline_scale_k Scale K load pipeline
     * @param pipeline_scale_v Scale V load pipeline
     * @param pipeline_q_producer_state Q load pipeline state
     * @param pipeline_k_producer_state K load pipeline state
     * @param pipeline_v_producer_state V load pipeline state
     * @param pipeline_scale_k_producer_state K load pipeline state
     * @param pipeline_scale_v_producer_state V load pipeline state
     * @param load_inputs global tensor Q, K, V, and masked tile count
     * @param blk_coord_q Q block coordinate (bidq, 0, (bidh, bidb))
     * @param blk_coord_kv KV block coordinate (bidq, 0, (bidh, bidb))
     * @param masked_tile_count number of horizontal trips
     * @param block_rank_in_cluster thread block index in a cluster
     * @param shared_tensors tensor in shared memory
     * @param load_scale_inputs load scale tensors
     */
    template<class TensorQ, class TensorK, class TensorV, class BlkCoord, 
        class... Ts>
    CUTLASS_DEVICE void load(Params const& mainloop_params,
        MainloopPipelineQ pipeline_q, MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v, MainloopPipelineSK pipeline_scale_k, 
        MainloopPipelineSV pipeline_scale_v,
        PipelineStateQ& pipeline_q_producer_state,
        PipelineStateKV& pipeline_k_producer_state,
        PipelineStateKV& pipeline_v_producer_state,
        PipelineStateKV& pipeline_scale_k_producer_state,
        PipelineStateKV& pipeline_scale_v_producer_state,
        cute::tuple<TensorQ, TensorK, TensorV> const& load_inputs,
        BlkCoord const& blk_coord_q, BlkCoord const& blk_coord_kv,
        int mask_tile_count, uint32_t block_rank_in_cluster,
        TensorStorage& shared_tensors,
        cute::tuple<Ts...> const& load_scale_inputs, int seqlen_cache_kv) {
        
        // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
        int warp_idx = canonical_warp_idx_sync();
        [[maybe_unused]] int warp_group_thread_idx = threadIdx.x % 128;

        // pick one thread in warpgroup for TMA load
        int lane_predicate = cute::elect_one_sync() && warp_idx == 0;

        // shared memory tensor Q (BLK_Q,BLK_D,PIPE)
        Tensor sQ = make_tensor(make_smem_ptr(
            shared_tensors.smem_q.data()), SmemLayoutQ{});
        // shared memory tensor K (BLK_KV,BLK_D,PIPE)
        Tensor sK_ = make_tensor(make_smem_ptr(
            shared_tensors.smem_k.data()), SmemLayoutK{});
        // shared memory tensor K (BLK_KV,BLK_D,PIPE)
        Tensor sK = as_position_independent_swizzle_tensor(sK_);
        // shared memory tensor V (BLK_D,BLK_KV,PIPE)
        Tensor sV_ = make_tensor(make_smem_ptr(shared_tensors.smem_v.data()), 
            SmemLayoutV{});
        // shared memory tensor V (BLK_D,BLK_KV,PIPE)
        Tensor sV = as_position_independent_swizzle_tensor(sV_);

        // number of thread blocks in Q dimension
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        // local threadblock id {blockIdx.x, blockIdx.y}
        uint2 cluster_local_block_id = {block_rank_in_cluster % 
            cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        // compute gQ, sQ
        // compute gK, sK
        // compute gV, sV
        
        // Q tensor global memory with q, d, l dim (BLK_Q,BLK_D,q,d,l)
        Tensor gQ_hdl = get<0>(load_inputs);
        // K tensor global memory with kv, d, l dim (BLK_KV,BLK_D,kv,d,l)
        Tensor gK_kdl = get<1>(load_inputs);
        // V tensor global memory with d, kv, l dim(BLK_D,BLK_KV,d,k,l)
        Tensor gV_dkl = get<2>(load_inputs);

        // block of q global tensor for TMA load
        auto block_tma_q = mainloop_params.tma_load_q.get_slice(
            cluster_local_block_id.y);
        // block of k global tensor for TMA load, multicasted
        auto block_tma_k = mainloop_params.tma_load_k.get_slice(
            cluster_local_block_id.x);
        // block of v global tensor for TMA load, multicasted
        auto block_tma_v = mainloop_params.tma_load_v.get_slice(
            cluster_local_block_id.x);

        // Partition the inputs based on the current block coordinates.
        // global memory tensor Q (BLK_H,BLK_D,h)
        Tensor gQ = gQ_hdl(_, _, _, _0{}, get<2>(blk_coord_q));
        // global memory tensor K (BLK_KV,BLK_D,kv)
        Tensor gK = gK_kdl(_, _, _, _0{}, get<2>(blk_coord_kv));
        // global memory tensor V (BLK_D,BLK_KV,kv)
        Tensor gV = gV_dkl(_, _, _0{}, _, get<2>(blk_coord_kv));
        
        // Applies the mapping from block_tma_a

        // TMA Load global memory tensor Q (TMA,TMA_Q,q)
        Tensor tQgQ = block_tma_q.partition_S(gQ)(_, _, _0{}, _);
        // TMA Store shared memory tensor Q (TMA,TMA_Q,PIPE)
        Tensor tQsQ = block_tma_q.partition_D(sQ)(_, _, _0{}, _);
        // TMA Load global memory tensor K (TMA,TMA_KV,kv)
        Tensor tKgK = block_tma_k.partition_S(gK)(_, _, _0{}, _);
        // TMA Store shared memory tensor K (TMA,TMA_KV,PIPE)
        Tensor tKsK = block_tma_k.partition_D(sK)(_, _, _0{}, _);
        // TMA Load global memory tensor V (TMA,TMA_KV,kv)
        Tensor tVgV = block_tma_v.partition_S(gV)(_, _0{}, _, _);
        // TMA Store shared memory tensor V (TMA,TMA_KV,PIPE)
        Tensor tVsV = block_tma_v.partition_D(sV_)(_, _0{}, _, _);

        // TMA multicast mask
        uint16_t mcast_mask_q = 0;
        // (m,n) -> block_id
        auto block_layout = Layout<ClusterShape>{}; 
        for (int n = 0; n < size<1>(block_layout); ++n) {
            mcast_mask_q |= (uint16_t(1) << block_layout(
                cluster_local_block_id.x, n, Int<0>{}));
        }

        auto extra_input_partitions = Utils::partition_extra_tma_inputs(
            mainloop_params, load_inputs, shared_tensors, 
            cluster_local_block_id, get<0>(blk_coord_kv), get<1>(blk_coord_kv), 
            get<2>(blk_coord_kv));

        // blk_coord in decomposed in terms of TileShape, not TileShapeQK
        // As such, it needs to be transformed as
        // (a,b,c): a -> a (Q)
        //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)
        
        // first Q index: a -> 2*a (Q0)
        int q_index = get<0>(blk_coord_q);
        // K index
        int k_index = 2 * get<0>(blk_coord_kv);
        // V index
        int v_index = 2 * get<0>(blk_coord_kv);
        int tile_count = mask_tile_count;

        int full_tiles_cache = seqlen_cache_kv / get<1>(TileShapeQK{});

            // Q0, K0, Q1, K1, V0, K2, V1, ... K(n-1), V(n-2), V(n-1)
            // two pipes: Q and KV
            // from Memory (prod) to TensorCore (cons)
        if(lane_predicate) {
            // Q0
            // LOCK state for _writing_
            pipeline_q.producer_acquire(pipeline_q_producer_state);
            // Copy gmem to smem
            auto tma_barrier_q = pipeline_q.producer_get_barrier(
                pipeline_q_producer_state);
            copy(mainloop_params.tma_load_q.with(*tma_barrier_q, mcast_mask_q, 
                cute::TMA::CacheHintSm90::EVICT_FIRST), tQgQ(_,_,q_index), 
                tQsQ(_,_,pipeline_q_producer_state.index()));

            auto tSgSK = get<0>(extra_input_partitions);
            auto tSsSK = get<1>(extra_input_partitions);
            auto tOgSV = get<2>(extra_input_partitions);
            auto tOsSV = get<3>(extra_input_partitions);

            CUTLASS_PRAGMA_NO_UNROLL
            for (; tile_count > 0; tile_count -= 1) {
                // K(i)
                // LOCK state for _writing_
                pipeline_k.producer_acquire(pipeline_k_producer_state);
                // Copy gmem to smem
                auto tma_barrier_k = pipeline_k.producer_get_barrier(
                    pipeline_k_producer_state);
                copy(mainloop_params.tma_load_k.with(*tma_barrier_k, 0, 
                    cute::TMA::CacheHintSm90::EVICT_LAST), tKgK(_,_,k_index), 
                    tKsK(_,_,pipeline_k_producer_state.index()));
                    // Advance pipeline_k_producer_state
                ++pipeline_k_producer_state;

                pipeline_scale_k.producer_acquire(
                    pipeline_scale_k_producer_state);
                auto tma_barrier_scale_k = pipeline_k.producer_get_barrier(
                    pipeline_scale_k_producer_state);
                copy(mainloop_params.tma_load_scale_k.with(
                    *tma_barrier_scale_k, 0), tSgSK(_,_,_,k_index),
                    tSsSK(_,_,_,pipeline_scale_k_producer_state.index()));
                ++pipeline_scale_k_producer_state;
                k_index++;

                // K(i + 1)
                // LOCK state for _writing_
                pipeline_k.producer_acquire(pipeline_k_producer_state);
                // Copy gmem to smem
                auto tma_barrier_k = pipeline_k.producer_get_barrier(
                    pipeline_k_producer_state);
                copy(mainloop_params.tma_load_k.with(*tma_barrier_k, 0, 
                    cute::TMA::CacheHintSm90::EVICT_LAST), tKgK(_,_,k_index), 
                    tKsK(_,_,pipeline_k_producer_state.index()));
                    // Advance pipeline_k_producer_state
                ++pipeline_k_producer_state;

                pipeline_scale_k.producer_acquire(
                    pipeline_scale_k_producer_state);
                auto tma_barrier_scale_k = pipeline_k.producer_get_barrier(
                    pipeline_scale_k_producer_state);
                copy(mainloop_params.tma_load_scale_k.with(
                    *tma_barrier_scale_k, 0), tSgSK(_,_,_,k_index),
                    tSsSK(_,_,_,pipeline_scale_k_producer_state.index()));
                ++pipeline_scale_k_producer_state;
                k_index++;

                // V(i)
                // LOCK state for _writing_
                pipeline_v.producer_acquire(pipeline_v_producer_state);
                // Copy gmem to smem
                auto tma_barrier_v = pipeline_v.producer_get_barrier(
                    pipeline_v_producer_state);
                copy(mainloop_params.tma_load_v.with(*tma_barrier_v, 0, 
                    cute::TMA::CacheHintSm90::EVICT_LAST), tVgV(_,_,v_index), 
                    tVsV(_,_,pipeline_v_producer_state.index()));
                ++pipeline_v_producer_state;

                pipeline_scale_v.producer_acquire(
                    pipeline_scale_v_producer_state);
                auto tma_barrier_scale_v = pipeline_v.producer_get_barrier(
                    pipeline_scale_v_producer_state);
                copy(mainloop_params.tma_load_scale_v.with(
                    *tma_barrier_scale_v, 0), tOgSV(_,_,_,v_index),
                    tOsSV(_,_,_,pipeline_scale_v_producer_state.index()));
                ++pipeline_scale_v_producer_state;
                v_index++;

                // V(i + 1)
                // LOCK state for _writing_
                pipeline_v.producer_acquire(pipeline_v_producer_state);
                // Copy gmem to smem
                auto tma_barrier_v = pipeline_v.producer_get_barrier(
                    pipeline_v_producer_state);
                copy(mainloop_params.tma_load_v.with(*tma_barrier_v, 0, 
                    cute::TMA::CacheHintSm90::EVICT_LAST), tVgV(_,_,v_index), 
                    tVsV(_,_,pipeline_v_producer_state.index()));
                ++pipeline_v_producer_state;

                pipeline_scale_v.producer_acquire(
                    pipeline_scale_v_producer_state);
                auto tma_barrier_scale_v = pipeline_v.producer_get_barrier(
                    pipeline_scale_v_producer_state);
                copy(mainloop_params.tma_load_scale_v.with(
                    *tma_barrier_scale_v, 0), tOgSV(_,_,_,v_index),
                    tOsSV(_,_,_,pipeline_scale_v_producer_state.index()));
                ++pipeline_scale_v_producer_state;
                v_index++;
            }
        }
    }

    /**
     * @brief Perform a Producer Epilogue to prevent early exit of blocks in a 
     * Cluster
     * 
     * @param pipeline_q Q load pipeline
     * @param pipeline_k K load pipeline
     * @param pipeline_v V load pipeline
     * @param pipeline_scale_K Scale V load pipeline
     * @param pipeline_scale_v Scale V load pipeline
     * @param pipeline_q_producer_state Q load pipeline state
     * @param pipeline_k_producer_state K load pipeline state
     * @param pipeline_v_producer_state V load pipeline state
     * @param pipeline_scale_k_producer_state Scale K load pipeline state
     * @param pipeline_scale_v_producer_state Scale V load pipeline state
     */
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q, MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v, MainloopPipelineSK pipeline_scale_k, 
        MainloopPipelineSV pipeline_scale_v,
        PipelineStateQ& pipeline_q_producer_state,
        PipelineStateKV& pipeline_k_producer_state,
        PipelineStateKV& pipeline_v_producer_state,
        PipelineStateKV& pipeline_scale_k_producer_state,
        PipelineStateKV& pipeline_scale_v_producer_state) {
        int warp_idx = canonical_warp_idx_sync();

        // pick one thread in warpgroup for TMA load
        int lane_predicate = cute::elect_one_sync() && warp_idx == 0;

        // Issue the epilogue waits
        if (lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all
            * Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was
            * still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(pipeline_q_producer_state);
            pipeline_k.producer_tail(pipeline_k_producer_state);
            pipeline_v.producer_tail(pipeline_v_producer_state);
            pipeline_scale_k.producer_tail(pipeline_scale_k_producer_state);
            pipeline_scale_v.producer_tail(pipeline_scale_v_producer_state);
        }
    }

    /**
     * @brief Perform a collective-scoped matrix multiply-accumulate
     * Consumer Perspective
     * 
     * @tparam BlkCoord Type of block coordinates
     * @tparam ProblemShape Type of problem shape
     * @tparam FrgTensorO Fragment Tensor O 
     * @tparam FrgTensorL Fragment Tensor L
     * @param pipeline_q Q load pipeline
     * @param pipeline_k K load pipeline
     * @param pipeline_v V load pipeline
     * @param pipeline_sk Scale K load pipeline
     * @param pipeline_sv Scale V load pipeline
     * @param pipeline_q_consumer_state Q consume pipeline state
     * @param pipeline_k_consumer_state K consume pipeline state
     * @param pipeline_v_consumer_state V consume pipeline state
     * @param pipeline_sk_consumer_state Scale K consume pipeline state
     * @param pipeline_sv_consumer_state Scale V consume pipeline state
     * @param tOrO Tensor O ((2, 2, D/8), MMA_Q, MMA_D):((1, 2, 4), 0, 0)
     * @param mask_tile_count number of trips
     * @param thread_idx thread index
     * @param shared_tensors shared memory tensors
     * @param mainloop_params params kernel parameters
     * @param blk_coord block coordinate (q, 0, (bidh, bidb))
     * @param problem_shape problem shape (q, kv, d, ((hg, hr), b))
     */
    template<class BlkCoord, class ProblemShape, class FrgTensorO,
        class Softmax>
    CUTLASS_DEVICE void mma(MainloopPipelineQ pipeline_q,
        MainloopPipelineK pipeline_k, MainloopPipelineSK pipeline_sk,
        MainloopPipelineV pipeline_v, MainloopPipelineSK pipeline_sv,
        PipelineStateQ& pipeline_q_consumer_state,
        PipelineStateKV& pipeline_k_consumer_state,
        PipelineStateKV& pipeline_v_consumer_state,
        PipelineStateKV& pipeline_sk_consumer_state,
        PipelineStateKV& pipeline_sv_consumer_state,
        FrgTensorO& tOrO, Softmax& softmax, int mask_tile_count,
        int thread_idx, TensorStorage& shared_tensors,
        Params const &mainloop_params, BlkCoord const& blk_coord,
        ProblemShape const& problem_shape,
        MathWarpGroupOrderBarrier& math_wg_order_barrier) {
        static_assert(is_rmem<FrgTensorO>::value,
            "O tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutQ{}) == 3,
            "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutK{}) == 3,
            "Smem layout must be rank 3.");
        static_assert(cute::rank(GmmaSmemLayoutV{}) == 3,
            "Smem layout must be rank 3.");
        // warp index
        int warp_idx = canonical_warp_idx_sync();
        [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

        // shared memory tensor Q(BLK_Q, BLK_D, PIPE)
        Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.data()), 
            SmemLayoutQ{});
        // shared memory tensor K (BLK_KV, BLK_D, PIPE)
        Tensor sK_ = make_tensor(make_smem_ptr(shared_tensors.smem_k.data()), 
            SmemLayoutK{});
        // shared memory tensor K (BLK_KV, BLK_D, PIPE)
        Tensor sK = as_position_independent_swizzle_tensor(sK_);
        // shared memory tensor V (BLK_D, BLK_KV, PIPE)
        Tensor sV_ = make_tensor(make_smem_ptr(shared_tensors.smem_v.data()), 
            SmemLayoutV{});
        // shared memory tensor V (BLK_D, BLK_KV, PIPE)
        Tensor sV = as_position_independent_swizzle_tensor(sV_);
        // shared memory tensor K for GMMA ops (BLK_D,BLK_KV,PIPE)
        Tensor gmma_sK = make_tensor(make_smem_ptr(
            shared_tensors.smem_kd.data()), GmmaSmemLayoutK{});
        // shared memory tensor V for GMMA ops (BLK_D,BLK_KV,PIPE)
        Tensor gmma_sV = make_tensor(make_smem_ptr(
            shared_tensors.smem_vt.data()), GmmaSmemLayoutV{});

        //
        // Define S, O accumulators and Q/K/V partitioning
        //
        static_assert(stride<0>(typename TiledMmaQK::ALayout{}) == 0
            and stride<0>(typename TiledMmaQK::BLayout{}) == 0
            and size<0>(typename TiledMmaQK::ALayout{})
            == NumThreadsPerWarpGroup 
            and size<0>(typename TiledMmaQK::BLayout{})
            == NumThreadsPerWarpGroup, 
            "Stride of the first mode must be 0 and the size of the mode must "
            "be NumThreadsPerWarpGroup");
        static_assert(stride<0>(typename TiledMmaPV::BLayout{}) == 0
            and size<0>(typename TiledMmaPV::BLayout{})
            == NumThreadsPerWarpGroup, 
            "Stride of the first mode must be 0 and the size of the mode must "
            "be NumThreadsPerWarpGroup");
        
        constexpr int MmaWarpGroups = size(TiledMmaQK{})
            / NumThreadsPerWarpGroup;
        // warpgroup thread layout: (1):(128)
        Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, 
            Int<NumThreadsPerWarpGroup>{});
        // 1 or 2 for consumers
        int warp_group_idx = __shfl_sync(0xFFFFFFFF,
            thread_idx / NumThreadsPerWarpGroup, 0);

        // Tiled MMA for QK^T: (MMA_ATOM, (1, 1, 1), (_, _, _))
        TiledMmaQK tiled_mma_qk;
        // Tiled MMA for PV: (MMA_ATOM, (1, 1, 1), (_, _, _))
        TiledMmaPV tiled_mma_pv;

        // MMA Thread Slice QK^T
        auto mma_thread_slice_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        // MMA Thread Slice PV
        auto mma_thread_slice_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        // thread partitioned shared tensor K (MMA, MMA_KV, MMA_D, PIPE)
        Tensor tSsK = mma_thread_slice_qk.partition_B(sK);
        // thread partitioned shared tensor K (MMA, MMA_KV, MMA_D, PIPE)
        Tensor gmma_tSsK = mma_thread_slice_qk.partition_B(gmma_sK);
        // thread partitioned shared tensor V (MMA, MMA_D, MMA_KV, PIPE)
        Tensor tOsV = mma_thread_slice_pv.partition_B(gmma_sV);

        // MMA warpgroup slice of QK^T
        auto mma_warpgroup_slice_qk = tiled_mma_qk.get_slice(
            warp_group_thread_layout(warp_group_idx));
        // MMA warpgroup slice of of MMA PV
        auto mma_warpgroup_slice_pv = tiled_mma_pv.get_slice(
            warp_group_thread_layout(warp_group_idx));

        // Allocate "fragments/descriptors"
        // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tSrK_mma = mma_thread_slice_qk.partition_fragment_B(sK(_,_,
            Int<0>{}));
        // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tSrK_load = make_fragment_like<ElementK>(tSrK_mma);

        // thread partitioned register tensor S (MMA, MMA_Q, MMA_KV)
        Tensor tSrS = partition_fragment_C(tiled_mma_qk,
            select<0, 1>(TileShapeQK{}));
        // thread partitioned register tensor V (MMA, MMA_D, MMA_KV, PIPE)
        Tensor tOrV = mma_thread_slice_pv.make_fragment_B(tOsV);

        // thread partitioned shared tensor Q (MMA, MMA_Q, MMA_D, PIPE)
        Tensor tSsQ = mma_thread_slice_qk.partition_A(sQ);
        // thread partitioned register tensor Q (MMA, MMA_Q, MMA_D, PIPE)
        Tensor tSrQ = mma_thread_slice_qk.make_fragment_A(tSsQ);

        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrS)); // Q
        CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<2>(tSrS)); // KV
        CUTE_STATIC_ASSERT_V(size<2>(tSsQ) == size<2>(tSsK)); // D
        CUTE_STATIC_ASSERT_V(Int<StageCountQ>{} == size<2>(sQ)); // PIPE
        CUTE_STATIC_ASSERT_V(Int<StageCountKV>{} == size<2>(sK)); // PIPE
        CUTE_STATIC_ASSERT_V(size<1>(tOsV) == size<2>(tOrO)); // D
        CUTE_STATIC_ASSERT_V(Int<StageCountKV>{} == size<2>(gmma_sV)); // PIPE

        // rmem layout is
        // S0 S1`O0 O1
        // sequential in memory, where S overlaps with P and V
        // Allocate the the accumulators for the (M,N) blk_shape

        curandStatePhilox4_32_10_t state;
        Dropout{}.init_rand(state, mainloop_params.dropout);

        int tile_count = mask_tile_count;
        
        //
        // Copy Atom K retiling
        //
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomK{}, 
            tiled_mma_qk);
        auto smem_tiled_copy_gmma_K = make_tiled_copy_B(GmmaSmemCopyAtomK{}, 
            tiled_mma_qk);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(
            warp_group_thread_idx);
        auto smem_thr_copy_gmma_K = smem_tiled_copy_gmma_K.get_thread_slice(
            warp_group_thread_idx);
        // (CPY,CPY_KV,CPY_D)
        Tensor tSrK_copy_view  = smem_thr_copy_K.retile_D(tSrK_load);

        // Partition of thread -> shared and thread -> RF
        auto partitioned_extra_info_k = Utils::partition_extra_mma_info_k(
            mma_thread_slice_qk, shared_tensors);
        auto copy_partitions_extra_info_k = Utils::retile_extra_mma_info_k(
            tiled_mma_qk, partitioned_extra_info_k, warp_group_thread_idx);

        constexpr int D_BLOCK_MAX = size<2>(tSrQ);
        constexpr int D_WAIT_MAX = cute::min(D_BLOCK_MAX - 1, 7);
        constexpr int KV_BLOCK_MAX = size<2>(tOrP);
        constexpr int KV_WAIT_MAX = cute::min(KV_BLOCK_MAX - 1, 7);

        // WAIT on smem_pipe_read until its data are available (phase bit flips 
        // from rdPhaseBit value)
        auto barrier_token_q = pipeline_q.consumer_try_wait(
            pipeline_q_consumer_state);
        pipeline_q.consumer_wait(pipeline_q_consumer_state, barrier_token_q);

        TransposeOperandV transpose = cute::AsyncTranspositionOperandB(
            warp_idx, warp_group_thread_idx, TiledMmaPV{}, SmemLayoutV{},
            SmemLayoutAtomV{}, ElementV{}, ElementP{});

        auto partitioned_extra_info_v = Utils::partition_extra_mma_info_v(
            mma_thread_slice_pv, shared_tensors);
        
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

        // loop:
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_count > 0; tile_count -= 1) {
            // gemm Q * K0 -> S
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

            warpgroup_fence_operand(tSrS);

            // WAIT on smem_pipe_read until its data are available (phase bit 
            // flips from rdPhaseBit value)
            auto barrier_token_k = pipeline_k.consumer_try_wait(
                pipeline_k_consumer_state);
            pipeline_k.consumer_wait(pipeline_k_consumer_state, 
                barrier_token_k);
            auto barrier_token_sk = pipeline_sk.consumer_try_wait(
                pipeline_sk_consumer_state);
            pipeline_sk.consumer_wait(pipeline_sk_consumer_state, 
                barrier_token_sk);
            // wait for K0
            int k_index = pipeline_k_consumer_state.index();

            // copy smem->rmem for A operand
            Utils::copy_tensors_KVD(smem_tiled_copy_K, tSsK, tSrK_copy_view, 
                partitioned_extra_info_k, copy_partitions_extra_info_k, 0, 
                k_index);
            if(D_BLOCK_MAX > 1) { // prefetch next block
                Utils::copy_tensors_KVD(smem_tiled_copy_K, tSsK, 
                    tSrK_copy_view, partitioned_extra_info_k, 
                    copy_partitions_extra_info_k, 1, k_index);
            }
            Utils::dequantize_K_dblock(tSrK_load, tSrK_mma, 
                partitioned_extra_info_k, 0);
            copy(smem_tiled_copy_gmma_K, tSrK_mma, gmma_tSsK(_,_,_0{},k_index));
            cutlass::arch::fence_view_async_shared();

            // Order two Math WG's MMA one after the other, helps hide Epilogue
            math_wg_order_barrier.wait();

            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int d_block = 0; d_block < D_BLOCK_MAX; ++d_block) {
                warpgroup_arrive();
                // (V,H) x (V,KV) => (V,H,KV)
                cute::gemm(tiled_mma_qk, tSrQ(_,_,d_block,_0{}), gmma_tSsK(_,_,
                    d_block,k_index), tSrS);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                warpgroup_commit_batch();

                if (d_block < D_BLOCK_MAX - 2) { // prefetch next block
                    Utils::copy_tensors_MK(smem_tiled_copy_K, tSsK, 
                        tSrK_copy_view, partitioned_extra_info_k, 
                        copy_partitions_extra_info_k, d_block + 2, k_index);
                }
                if (d_block < D_BLOCK_MAX - 1) {
                    Utils::dequantize_K_kblock(tSrK_load, tSrK_mma, 
                        partitioned_extra_info_k, d_block + 1);
                    copy(smem_tiled_copy_gmma_K, tSrK_mma, gmma_tSsK(_,_,
                        d_block + 1,k_index));
                    cutlass::arch::fence_view_async_shared();
                }
            }

            // Cue for next Math WG's MMA to start
            math_wg_order_barrier.arrive();

            warpgroup_wait<0>();
            warpgroup_fence_operand(tSrS);

            // release K0
            pipeline_k.consumer_release(pipeline_k_consumer_state);
            pipeline_sk.consumer_release(pipeline_sk_consumer_state);

            // compute softmax
            Scale::scale(tSrS, mainloop_params.scale);
            softmax.softmax_preexp<true>(tSrS, mainloop_params.softmax);
            softmax.softmax_exp(tSrS, mainloop_params.softmax);
            Dropout{}.apply_dropout(tSrS, state, mainloop_params.dropout);
            // thread partitioned register tensor P (MMA, MMA_Q, MMA_KV)
            Tensor tOrP_A = make_tensor(tSrS.data(), convert_layout_C_to_A(
                tSrS.layout()));
            // slice out staging
            Tensor tOrP = make_tensor_like<ElementP>(tOrP_A);
            convert(tOrP_A, tOrP);
            permute_A(tOrP);

            warpgroup_fence_operand(tOrP);
            warpgroup_fence_operand(tOrO);

            // wait for Vi
            // WAIT on smem_pipe_read until its data are available (phase 
            // bit flips from rdPhaseBit value)
            auto barrier_token_v = pipeline_v.consumer_try_wait(
                pipeline_v_consumer_state);
            pipeline_v.consumer_wait(pipeline_v_consumer_state, 
                barrier_token_v);
            int v_index = pipeline_v_consumer_state.index();
            auto barrier_token_sv = pipeline_sv.consumer_try_wait(
                pipeline_sv_consumer_state);
            pipeline_sv.consumer_wait(pipeline_sv_consumer_state, 
                barrier_token_sv);

            transpose(sV, gmma_sV, pipeline_v_producer_state.index(), 
                partitioned_extra_info_v);

            // Order two Math WG's MMA one after the other, helps hide 
            // Epilogue
            math_wg_order_barrier.wait();

            warpgroup_arrive();

            // gemm P * Vi -> O
            // Unroll the KV mode manually to set scale to 1
            CUTLASS_PRAGMA_UNROLL
            for (int kv_block = 0; kv_block < size<2>(tOrP); ++kv_block) {
                // (V,Q,KV) x (V,D,KV) => (V,Q,D)
                cute::gemm(tiled_mma_pv, tOrP(_,_,kv_block),
                    tOrV(_,_,kv_block,v_index), tOrO);
                tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
            }

            warpgroup_commit_batch();

            // Cue for next Math WG's MMA to start
            math_wg_order_barrier.arrive();

            warpgroup_wait<0>();
            warpgroup_fence_operand(tOrO);

            // release V(i-1)
            pipeline_v.consumer_release(pipeline_v_consumer_state);
            pipeline_sv.consumer_release(pipeline_sv_consumer_state);
        }

        // release Q
        pipeline_q.consumer_release(pipeline_q_consumer_state);
    }
};

}  // namespace cutlass::fmha::collective

} // namespace cutlass::fmha