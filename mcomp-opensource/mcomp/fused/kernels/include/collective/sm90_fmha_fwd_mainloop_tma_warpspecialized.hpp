/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace cutlass::fmha {

using namespace cute;


namespace collective {

/**
 * @brief Sm90 FMHA Forward Mainloop TMA Warp Specialized
 * 
 * @tparam TileShape_QKVD Tile shape type (CTA_Q, CTA_KV, CTA_D)
 * @tparam ElementQ_ Type of element of Q to compute
 * @tparam ElementK_ Type of element of K to compute
 * @tparam ElementV_ Type of element of V to compute
 */
template <class TileShape_QKVD_, class ClusterShape_, class ElementQ_,
    class ElementK_, class ElementV_>
struct Sm90FmhaFwdMainloopTmaWarpspecialized {
    // type of element of Q to compute
    using ElementQ = cutlass::float_e4m3_t;
    // Q Global memory Tag: row major
    using GmemLayoutQTag = cutlass::layout::RowMajor;
    // number of elements of Qin alignment of 16 bytes
    static const int AlignmentQ = 128 / sizeof_bits_v<ElementQ>;
    // type of element of K to compute
    using ElementK = cutlass::float_e4m3_t;
    // K Global memory Tag: column major
    using GmemLayoutKTag = cutlass::layout::ColumnMajor;
    // number of elements of K in alignment of 16 bytes
    static const int AlignmentK = 128 / sizeof_bits_v<ElementK>;
    // type of element of P to compute
    using ElementP = cutlass::float_e4m3_t;
    // type of element of V to compute
    using ElementV = cutlass::float_e4m3_t;
    // V Global memory Tag: row major
    using GmemLayoutVTag = cutlass::layout::RowMajor;
    // number of elements of K in alignment of 16 bytes
    static const int AlignmentV = 128 / sizeof_bits_v<ElementV>;
    // type of element to accumulate QK^T
    using ElementAccumulatorS = cutlass::half_t;
    // type of element to accumulate PV
    using ElementAccumulatorO = float;
    // tile shape type
    using TileShape = TileShape_QKVD_;
    // thread block custer shape
    using ClusterShape = ClusterShape_;
    // two stage implmenentation
    using StageCount = cutlass::gemm::collective::StageCount<2>;
    // Pingpong Scheduling
    using KernelScheduleType = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

    // type of division inside CTA
    using AtomLayoutQKVD = Shape<_2, _1, _1>;
    // type of tile shape of QK compute: (Q / 2, KV, D)
    using TileShapeQK = decltype(shape_div(TileShape{}, AtomLayoutQKVD{}));
    // type of tile shape of QK compute: (Q / 2, D, KV)
    using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));
    // type of Tiled MMA QK
    using TiledMmaQK = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
        ElementQ, ElementK, ElementAccumulatorS, TileShapeQK, 
        cute::GMMA::Major::K, cute::GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}));
    // type of Tiled MMA PV
    using TiledMmaPV = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
        ElementP, ElementV, ElementAccumulatorO, TileShapePV, 
        cute::GMMA::Major::K, cute::GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}));
    
    // Q Global memory tiled copy type
    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    // K Global memory tiled copy type
    using GmemTiledCopyK = cute::SM90_TMA_LOAD_MULTICAST;
    // V Global memory tiled copy type
    using GmemTiledCopyV = cute::SM90_TMA_LOAD_MULTICAST;

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
        ::ss_smem_selector<GMMA::Major::MN, ElementV, decltype(cute::get<1>(
        TileShapePV{})), decltype(cute::get<2>(TileShapePV{}))>());
    // atomic shared memory layout for VT
    using GmmaSmemLayoutAtomV = decltype(cutlass::gemm::collective::detail
        ::ss_smem_selector<GMMA::Major::K, ElementV, decltype(cute::get<1>(
        TileShapePV{})), decltype(cute::get<2>(TileShapePV{}))>());

    // stride type of Q: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, 
        int>>;
    // stride type of K: (H_R* D, 1, ((0, D), H_R*D*KV))
    using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, 
        int>>;
    // stride type of V: (H_R* D, 1, ((0, D), H_R*D*KV))
    using StrideV = StrideK;
    // masking type
    using Mask = CausalMask;
    // dropout type
    using Dropout = NoDropout;
    

    // number of shared memory stages for Q load
    static constexpr int StageCountQ = 2;
    // number of shared memory stages for K load
    static constexpr int StageCountKV = 3;

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
    // SmemLayoutV for GMMA is different from SmemLayoutV for TMA
    using GmmaSmemLayoutV = decltype(tile_to_shape(GmmaSmemLayoutAtomV{},
        make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{}),  
        Int<StageCountKV>{}), Step<_1,_2,_3>{}));
    
    using TransposeOperandV = decltype(cute::AsyncTranspositionOperandB(0, 0, 
        TiledMmaPV{}, SmemLayoutV{}, SmemLayoutAtomV{}, ElementV{})); 

    // from load to mma warp, protects q in smem
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<StageCountQ>;
    // from load to mma warp, protects k in smem
    using MainloopPipelineK = cutlass::PipelineTmaAsync<StageCountKV>;
    // Inside load warp, protects v in smem
    using MainloopPipelineV = cutlass::PipelineTmaAsync<StageCountKV>;
    // Inside load warp, protects v^T in smem
    using MainloopPipelineVT = cutlass::PipelineAsync<StageCountKV>;
    
    /**
     * @brief Shared Memory Storage
     */
    struct SharedStorage {
        /**
         * @brief Tensor storage inside shared memory
         */
        struct TensorStorage : cute::aligned_struct<128, _0> {
            // shared memory allocation of Q
            cute::array_aligned<ElementQ, cute::cosize_v<SmemLayoutQ>> smem_q;
            // shared memory allocation of K
            cute::array_aligned<ElementK, cute::cosize_v<SmemLayoutK>> smem_k;
            // shared memory allocation of V
            cute::array_aligned<ElementV, cute::cosize_v<SmemLayoutV>> smem_v;
            // shared memory allocation of V
            cute::array_aligned<ElementV, cute::cosize_v<GmmaSmemLayoutV>> 
                smem_vt;
        } tensors;
        // Storage of Load Q Pipeline type
        using PipelineQStorage = typename MainloopPipelineQ::SharedStorage;
        // Storage of Load K Pipeline type
        using PipelineKStorage = typename MainloopPipelineK::SharedStorage;
        // Storage of Load V Pipeline type
        using PipelineVStorage = typename MainloopPipelineV::SharedStorage;
        // Storage of Load V Pipeline type
        using PipelineVTStorage = typename MainloopPipelineVT::SharedStorage;
        // Storage of Load Q Pipeline
        PipelineQStorage pipeline_q;
        // Storage of Load K Pipeline
        PipelineKStorage pipeline_k;
        // Storage of Load V Pipeline
        PipelineVStorage pipeline_v;
        // Storage of Load V^T Pipeline
        PipelineVTStorage pipeline_vt;
    };
    // Tensor storage inside shared memory
    using TensorStorage = typename SharedStorage::TensorStorage;
    // Storage of Load Q Pipeline type
    using PipelineQStorage = typename SharedStorage::PipelineQStorage;
    // Storage of Load K Pipeline type
    using PipelineKStorage = typename SharedStorage::PipelineKStorage;
    // Storage of Load V Pipeline type
    using PipelineVStorage = typename SharedStorage::PipelineVStorage;
    // Storage of V^T Transpose Pipeline type
    using PipelineVTStorage = typename SharedStorage::PipelineVTStorage;

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
        const ElementK* ptr_K;
        // stride of K
        StrideK dK;
        // pointer of V
        const ElementV* ptr_V;
        // stride of V
        StrideV dV;
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
        // Assumption: StrideQ is congruent with Problem_QD
        using TMA_Q = decltype(make_tma_copy_A_sm90(GmemTiledCopyQ{},
            make_tensor(static_cast<ElementQ const*>(nullptr),
            repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
            SmemLayoutQ{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{}));
        // Assumption: StrideK is congruent with Problem_KVD
        using TMA_K = decltype(make_tma_copy_B_sm90(GmemTiledCopyK{},
            make_tensor(static_cast<ElementK const*>(nullptr),
            repeat_like(StrideK{}, int32_t(0)), StrideK{}),
            SmemLayoutK{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{}));
        // Assumption: StrideV is congruent with Problem_KVD
        using TMA_V = decltype(make_tma_copy_B_sm90(GmemTiledCopyV{},
            make_tensor(static_cast<ElementV const*>(nullptr),
            repeat_like(select<1,0,2>(StrideV{}), int32_t(0)),
            select<1,0,2>(StrideV{})), SmemLayoutV{}(_,_,cute::Int<0>{}), 
            TileShapePV{}, ClusterShape{}));
        // TMA Q load descriptor
        TMA_Q tma_load_q;
        // TMA K load descriptor
        TMA_K tma_load_k;
        // TMA V load descriptor
        TMA_V tma_load_v;
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
    };

    /**
     * @brief Check the Problem Shape Alignment
     * 
     * @tparam ProblemShape type of problem shape
     * @param problem_shape problem shape (Q, KV, D, ((H_G, H_R), B))
     * @param args host side kernel arguments
     * @return true Alignment meet
     * @return false Alignment does not meet
     */
    template<class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape,
        [[maybe_unused]] Arguments const& args) {
        constexpr int tma_alignment_bits = 128;
        auto [Q,KV,D,HB] = problem_shape;

        bool implementable = true;
        constexpr int min_tma_aligned_elements_Q = tma_alignment_bits
            / cutlass::sizeof_bits<ElementQ>::value;
        implementable = implementable && cutlass::detail::check_alignment<
            min_tma_aligned_elements_Q>(cute::make_shape(Q, D, HB), StrideQ{});
        constexpr int min_tma_aligned_elements_K = tma_alignment_bits
            / cutlass::sizeof_bits<ElementK>::value;
        implementable = implementable && cutlass::detail::check_alignment<
            min_tma_aligned_elements_K>(cute::make_shape(KV, D, HB), StrideK{});
        constexpr int min_tma_aligned_elements_V = tma_alignment_bits
            / cutlass::sizeof_bits<ElementV>::value;
        implementable = implementable && cutlass::detail::check_alignment<
            min_tma_aligned_elements_V>(cute::make_shape(KV, D, HB), StrideV{});

        if (!implementable) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the "
                "minimum alignment requirements for TMA.\n");
        }
        return implementable;
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
        auto ptr_Q = args.ptr_Q;
        auto ptr_K = args.ptr_K;
        auto ptr_V = args.ptr_V;
        auto dQ = args.dQ;
        auto dK = args.dK;
        auto dV = args.dV;
        auto problem_shape_qk = problem_shape;
    
        if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
            auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
            if (cumulative_length_q != nullptr) {
                int max_length_q = get<0>(problem_shape).max_length;
                // for variable sequence length, the batch is in units of 
                // row_stride
                get<2,1>(dQ) = get<0>(dQ);
                get<3,1>(problem_shape_qk) = std::max(get<3,1>(
                    problem_shape_qk), max_length_q * (1 + get<3,1>(
                    problem_shape)));
                // offset ptr by the amount we add back in later
                ptr_Q -= max_length_q * get<0>(dQ);
            }
        }
    
        if constexpr (is_variable_length_v<tuple_element_t<1, ProblemShape>>) {
            auto cumulative_length_kv = get<1>(problem_shape).cumulative_length;
            if (cumulative_length_kv != nullptr) {
                int max_length_kv = get<1>(problem_shape).max_length;
                // for variable sequence length, the batch is in units of 
                // row_stride
                get<2,1>(dK) = get<0>(dK);
                get<2,1>(dV) = get<0>(dV);
                get<3,1>(problem_shape_qk) = std::max(get<3,1>(
                    problem_shape_qk), max_length_kv * (1 + get<3,1>(
                    problem_shape)));
                // offset ptr by the amount we add back in later
                ptr_K -= max_length_kv * get<0>(dK);
                ptr_V -= max_length_kv * get<0>(dV);
            }
        }

        auto [Q,KV,D,HB] = problem_shape_qk;

        Tensor tensor_q = make_tensor(ptr_Q, make_layout(make_shape(Q,D,HB), 
            args.dQ));
        Tensor tensor_k = make_tensor(ptr_K, make_layout(make_shape(KV,D,HB), 
            args.dK));
        Tensor tensor_v = make_tensor(ptr_V, make_layout(make_shape(D,KV,HB), 
            select<1,0,2>(args.dV)));

        auto tma_load_q = make_tma_copy_A_sm90(GmemTiledCopyQ{}, tensor_q, 
            SmemLayoutQ{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{});
        auto tma_load_k = make_tma_copy_B_sm90(GmemTiledCopyK{}, tensor_k, 
            SmemLayoutK{}(_,_,cute::Int<0>{}), TileShapeQK{}, ClusterShape{});
        auto tma_load_v = make_tma_copy_B_sm90(
            GmemTiledCopyV{}, tensor_v, SmemLayoutV{}(_,_,cute::Int<0>{}),
            TileShapePV{}, ClusterShape{});

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

        return Params{tma_load_q, tma_load_k, tma_load_v,
            Scale::to_underlying_arguments(problem_shape, scale, workspace),
            to_underlying_softmax_arguments(problem_shape, softmax, 
                workspace),
            Dropout::to_underlying_arguments(problem_shape, dropout, workspace),
            transaction_bytes, transaction_bytes_q, transaction_bytes_k, 
            transaction_bytes_v};
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
    }

    /**
     * @brief Get global tensors to process and trip count
     * 
     * @tparam BlkCoord Coord<int, _0, Coord<int, int>>
     * @tparam ProblemShape Coord<int, int, int, Coord<Coord<int, int>, int>>
     * @tparam ParamsProblemShape Original Problem shape, could be variable len
     * @param blk_coord_in block coordinate (bidq, _0, (bidh, bidb))
     * @param problem_shape problem shape (q, kv, d, ((hg, hr), b))
     * @param params_problem_shape kernel parameter problem shape 
     * @return cute::tuple<TensorQ, TensorK, TensorV, int> const& global tensor 
     * of Q, K, V, and masked tile count
     */
    template<class BlkCoord, class ProblemShape_QKVDHGHRB,
        class ParamsProblemShape_QKVDHGHRB>
    CUTLASS_DEVICE auto load_init(BlkCoord const& blk_coord_in,
        ProblemShape_QKVDHGHRB const& problem_shape,
        ParamsProblemShape_QKVDHGHRB const& params_problem_shape,
        Params const& mainloop_params) const {
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

        // (seq_q, head_dim, ((grouped_heads, head_kv), batch_size))
        Tensor mQ_qdl_p = mainloop_params.tma_load_q.get_tma_tensor(
            select<0,2,3>(problem_shape));
        // (seq_kv, head_dim, ((grouped_heads, head_kv), batch_size))
        Tensor mK_kdl_p = mainloop_params.tma_load_k.get_tma_tensor(
            select<1,2,3>(problem_shape));
        // (head_dim, seq_kv, ((grouped_heads, head_kv), batch_size))
        Tensor mV_dkl_p = mainloop_params.tma_load_v.get_tma_tensor(
            select<2,1,3>(problem_shape));
    
        // Q matrix offset at Q dimension
        int q_offs_0 = 0;
        // Q matrix offset at batch index
        int q_offs_2_1 = 0;
        // K, V matrix offset at K, V dimension
        int kv_offs_0 = 0;
        // K, V matrix offset at batch index
        int kv_offs_2_1 = 0;
    
        if constexpr (is_variable_length_v<tuple_element_t<0, 
            ParamsProblemShape_QKVDHGHRB>>) {
            auto cumulative_length_q
                = get<0>(params_problem_shape).cumulative_length;
            if (cumulative_length_q != nullptr) {
                int max_length_q = get<0>(params_problem_shape).max_length;
                q_offs_0 = max_length_q - get<0>(problem_shape);
                q_offs_2_1 = cumulative_length_q[get<2,1>(blk_coord_q)]
                    + get<0>(problem_shape);
                get<2,1>(blk_coord_q) = 0;
            }
        }
        if constexpr (is_variable_length_v<tuple_element_t<1, 
            ParamsProblemShape_QKVDHGHRB>>) {
            auto cumulative_length
                = get<1>(params_problem_shape).cumulative_length;
            if (cumulative_length != nullptr) {
                int max_length = get<1>(params_problem_shape).max_length;
                kv_offs_0 = max_length - get<1>(problem_shape);
                kv_offs_2_1 = cumulative_length[get<2,1>(blk_coord_kv)]
                    + get<1>(problem_shape);
                get<2,1>(blk_coord_kv) = 0;
            }
        }
    
        // TMA requires special handling of strides to deal with coord codomain 
        // mapping. Represent the full tensors -- get these from TMA

        // (q=q_offs_0,d=0,q_offs_2_1)
        Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{},
            make_coord(_0{}, q_offs_2_1)), mQ_qdl_p);
        // (kv=kv_offs_0,d=0,kv_offs_2_1)
        Tensor mK_kdl = domain_offset(make_coord(kv_offs_0, _0{},
            make_coord(_0{}, kv_offs_2_1)), mK_kdl_p);
        // (d=0,kv=kv_offs_0,kv_offs_2_1)
        Tensor mV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, 
            make_coord(_0{}, kv_offs_2_1)), mV_dkl_p);
    
        // (BLK_Q,BLK_D,q,d,l)
        Tensor gQ_qdl = local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), 
            Step<_1, X, _1>{});
        // (BLK_KV,BLK_D,kv,d,l)
        Tensor gK_kdl = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), 
            Step<X, _1, _1>{});
        // (BLK_D,BLK_KV,d,kv,l)
        Tensor gV_dkl = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), 
            Step<X, _1, _1>{});
            
        return cute::make_tuple(gQ_qdl, gK_kdl, gV_dkl, blk_coord_q, 
            blk_coord_kv, mask_tile_count);
    }

    /**
     * @brief Perform a collective-scoped matrix multiply-accumulate
     * Producer Perspective
     * 
     * @tparam TensorQ (BLK_Q, BLK_D, q_off_0, 0, (0, q_offs_2_1))
     * @tparam TensorK (BLK_K, BLK_D, kv_off_0, 0, (0, kv_offs_2_1))
     * @tparam TensorV (BLK_D, BLK_V, 0, kv_off_0, (0, kv_offs_2_1))
     * @tparam BlkCoord Coord<int, _0, Coord<int, int>>
     * @param mainloop_params kernel parameters
     * @param pipeline_q Q load pipeline
     * @param pipeline_k K load pipeline
     * @param pipeline_v V load pipeline
     * @param pipeline_vt V^T convert pipeline
     * @param pipeline_q_producer_state Q load pipeline state
     * @param pipeline_k_producer_state K load pipeline state
     * @param pipeline_v_producer_state V load producer pipeline state
     * @param load_inputs global tensor Q, K, V, and masked tile count
     * @param blk_coord_q Q block coordinate (bidq, 0, (bidh, bidb))
     * @param blk_coord_kv KV block coordinate (bidq, 0, (bidh, bidb))
     * @param masked_tile_count number of horizontal trips
     * @param block_rank_in_cluster thread block index in a cluster
     * @param shared_tensors tensor in shared memory
     */
    template<class TensorQ, class TensorK, class TensorV, class BlkCoord>
    CUTLASS_DEVICE void load(Params const& mainloop_params,
        MainloopPipelineQ pipeline_q, MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v, MainloopPipelineVT pipeline_vt,
        PipelineStateQ& pipeline_q_producer_state,
        PipelineStateKV& pipeline_k_producer_state,
        PipelineStateKV& pipeline_v_producer_state,
        cute::tuple<TensorQ, TensorK, TensorV> const& load_inputs,
        BlkCoord const& blk_coord_q, BlkCoord const& blk_coord_kv,
        int mask_tile_count, uint32_t block_rank_in_cluster,
        TensorStorage& shared_tensors) {
        
        // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
        int warp_idx = canonical_warp_idx_sync();
        [[maybe_unused]] int warp_group_thread_idx = threadIdx.x % 128;

        // pick one thread in warpgroup for TMA load
        int lane_predicate = cute::elect_one_sync() && warp_idx == 0;

        // compute gQ, sQ
        // compute gK, sK
        // compute gV, sV
        
        // Q tensor global memory with q, d, l dim (BLK_Q,BLK_D,q,d,l)
        Tensor gQ_qdl = get<0>(load_inputs);
        // K tensor global memory with kv, d, l dim (BLK_KV,BLK_D,kv,d,l)
        Tensor gK_kdl = get<1>(load_inputs);
        // V tensor global memory with d, kv, l dim(BLK_D,BLK_KV,d,k,l)
        Tensor gV_dkl = get<2>(load_inputs);
            
        // shared memory tensor Q (BLK_Q,BLK_D,PIPE)
        Tensor sQ = make_tensor(make_smem_ptr(
            shared_tensors.smem_q.data()), SmemLayoutQ{});
        // shared memory tensor K (BLK_KV,BLK_D,PIPE)
        Tensor sK = make_tensor(make_smem_ptr(
            shared_tensors.smem_k.data()), SmemLayoutK{});
        // shared memory tensor V (BLK_D,BLK_KV,PIPE)
        Tensor sV_ = make_tensor(make_smem_ptr(shared_tensors.smem_v.data()), 
            SmemLayoutV{});
        // shared memory tensor V (BLK_D,BLK_KV,PIPE)
        Tensor sV = as_position_independent_swizzle_tensor(sV_);
        // shared memory tensor V for GMMA ops (BLK_D,BLK_KV,PIPE)
        Tensor gmma_sV_ = make_tensor(make_smem_ptr(
            shared_tensors.smem_vt.data()), GmmaSmemLayoutV{});
        // shared memory tensor V for GMMA ops (BLK_D,BLK_KV,PIPE)
        Tensor gmma_sV = as_position_independent_swizzle_tensor(gmma_sV_);

        // number of thread blocks in Q dimension
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        // local threadblock id {blockIdx.x, blockIdx.y}
        uint2 cluster_local_block_id = {block_rank_in_cluster % 
            cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
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
        // global memory tensor Q (BLK_Q,BLK_D,q)
        Tensor gQ = gQ_qdl(_, _, _, _0{}, get<2>(blk_coord_q));
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
        uint16_t mcast_mask_kv = 0;
        // (m,n) -> block_id
        auto block_layout = Layout<ClusterShape>{}; 
        for (int m = 0; m < size<0>(block_layout); ++m) {
            mcast_mask_kv |= (uint16_t(1) << block_layout(m,
                cluster_local_block_id.y, Int<0>{}));
        }

        // blk_coord in decomposed in terms of TileShape, not TileShapeQK
        // As such, it needs to be transformed as
        // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
        //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)
        
        // first Q index: a -> 2*a (Q0)
        int q_index = 2 * get<0>(blk_coord_q);
        // K index
        int k_index = 0;
        // V index
        int v_index = 0;
        int tile_count = mask_tile_count;
        // does require two stage load pipeline?
        bool two_stage = tile_count > 1;

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
            copy(mainloop_params.tma_load_q.with(*tma_barrier_q, 0, 
                cute::TMA::CacheHintSm90::EVICT_FIRST), tQgQ(_,_,q_index), 
                tQsQ(_,_,pipeline_q_producer_state.index()));
            q_index++;
            ++pipeline_q_producer_state;
            // K0
            // LOCK state for _writing_
            pipeline_k.producer_acquire(pipeline_k_producer_state);
            // Copy gmem to smem
            auto tma_barrier_k = pipeline_k.producer_get_barrier(
                pipeline_k_producer_state);
            copy(mainloop_params.tma_load_k.with(*tma_barrier_k, mcast_mask_kv, 
                cute::TMA::CacheHintSm90::EVICT_LAST), tKgK(_,_,k_index),
                tKsK(_,_,pipeline_k_producer_state.index()));
            // Advance pipeline_k_producer_state
            ++pipeline_k_producer_state;
            k_index++;
            // Q1
            // LOCK state for _writing_
            pipeline_q.producer_acquire(pipeline_q_producer_state);
            // Copy gmem to smem
            tma_barrier_q = pipeline_q.producer_get_barrier(
                pipeline_q_producer_state);
            copy(mainloop_params.tma_load_q.with(*tma_barrier_q, 0, 
                cute::TMA::CacheHintSm90::EVICT_FIRST), tQgQ(_,_,q_index), 
                tQsQ(_,_,pipeline_q_producer_state.index()));
            ++pipeline_q_producer_state;
        }

        tile_count -= 1;

        if(two_stage) {
            if(lane_predicate) {
                // K1
                // LOCK state for _writing_
                pipeline_k.producer_acquire(pipeline_k_producer_state);
                // Copy gmem to smem
                auto tma_barrier_k = pipeline_k.producer_get_barrier(
                    pipeline_k_producer_state);
                copy(mainloop_params.tma_load_k.with(*tma_barrier_k, 
                    mcast_mask_kv, cute::TMA::CacheHintSm90::EVICT_LAST),
                    tKgK(_,_,k_index), tKsK(_,_,
                        pipeline_k_producer_state.index()));
                // Advance pipeline_k_producer_state
                ++pipeline_k_producer_state;
                k_index++;
            }

            tile_count -= 1;
        }

        PipelineStateKV pipeline_v_consumer_state(
            pipeline_v_producer_state.index(),
            pipeline_v_producer_state.phase() ^ 1,
            pipeline_v_producer_state.count());
        
        TransposeOperandV transpose = cute::AsyncTranspositionOperandB(warp_idx, warp_group_thread_idx, TiledMmaPV{}, SmemLayoutV{},
            SmemLayoutAtomV{}, ElementV{});

        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_count > 0; tile_count -= 1) {
            if(lane_predicate) {
                // V(i-1)
                // LOCK state for _writing_
                pipeline_v.producer_acquire(pipeline_v_producer_state);
                // Copy gmem to smem
                auto tma_barrier_v = pipeline_v.producer_get_barrier(
                    pipeline_v_producer_state);
                copy(mainloop_params.tma_load_v.with(*tma_barrier_v, 
                    mcast_mask_kv, cute::TMA::CacheHintSm90::EVICT_LAST),
                    tVgV(_,_,v_index), tVsV(_,_,
                    pipeline_v_producer_state.index()));
                v_index++;
            }

            pipeline_vt.producer_acquire(pipeline_v_producer_state);

            auto barrier_token_v = pipeline_v.consumer_try_wait(
                pipeline_v_consumer_state);
            pipeline_v.consumer_wait(pipeline_v_consumer_state, 
                barrier_token_v);

            transpose(sV, gmma_sV, pipeline_v_producer_state.index());

            pipeline_v.consumer_release(pipeline_v_consumer_state);
            ++pipeline_v_consumer_state;
            pipeline_vt.producer_commit(pipeline_v_producer_state);
            ++pipeline_v_producer_state;

            // K(i+1)
            // LOCK state for _writing_
            if(lane_predicate) {
                pipeline_k.producer_acquire(pipeline_k_producer_state);
                // Copy gmem to smem
                auto tma_barrier_k = pipeline_k.producer_get_barrier(
                    pipeline_k_producer_state);
                copy(mainloop_params.tma_load_k.with(*tma_barrier_k, 
                    mcast_mask_kv, cute::TMA::CacheHintSm90::EVICT_LAST),
                    tKgK(_,_,k_index), tKsK(_,_,
                    pipeline_k_producer_state.index()));
                // Advance pipeline_k_producer_state
                ++pipeline_k_producer_state;
                k_index++;
            }
        }

        if(two_stage) {
            if(lane_predicate) {
                // V(n-2)
                // LOCK state for _writing_
                pipeline_v.producer_acquire(pipeline_v_producer_state);
                // Copy gmem to smem
                auto tma_barrier_v = pipeline_v.producer_get_barrier(
                    pipeline_v_producer_state);
                copy(mainloop_params.tma_load_v.with(*tma_barrier_v, 
                    mcast_mask_kv, cute::TMA::CacheHintSm90::EVICT_LAST),
                    tVgV(_,_,v_index), tVsV(_,_,
                    pipeline_v_producer_state.index()));
                v_index++;
            }

            pipeline_vt.producer_acquire(pipeline_v_producer_state);

            auto barrier_token = pipeline_v.consumer_try_wait(
                pipeline_v_consumer_state);
            pipeline_v.consumer_wait(pipeline_v_consumer_state, 
                barrier_token);

            transpose(sV, gmma_sV, pipeline_v_producer_state.index());

            pipeline_v.consumer_release(pipeline_v_consumer_state);
            ++pipeline_v_consumer_state;
            pipeline_vt.producer_commit(pipeline_v_producer_state);
            ++pipeline_v_producer_state;
        }

        if(lane_predicate) {
            // V(n-1)
            // LOCK state for _writing_
            pipeline_v.producer_acquire(pipeline_v_producer_state);
            // Copy gmem to smem
            auto tma_barrier_v = pipeline_v.producer_get_barrier(
                pipeline_v_producer_state);
            copy(mainloop_params.tma_load_v.with(*tma_barrier_v, 
                mcast_mask_kv, cute::TMA::CacheHintSm90::EVICT_LAST),
                tVgV(_,_,v_index), tVsV(_,_,pipeline_v_producer_state.index()));
            v_index++;
        }

        pipeline_vt.producer_acquire(pipeline_v_producer_state);

        auto barrier_token = pipeline_v.consumer_try_wait(
            pipeline_v_consumer_state);
        pipeline_v.consumer_wait(pipeline_v_consumer_state, 
            barrier_token);

        transpose(sV, gmma_sV, pipeline_v_producer_state.index());

        pipeline_v.consumer_release(pipeline_v_consumer_state);
        ++pipeline_v_consumer_state;
        pipeline_vt.producer_commit(pipeline_v_producer_state);
        ++pipeline_v_producer_state;
    }

    /**
     * @brief Perform a Producer Epilogue to prevent early exit of blocks in a 
     * Cluster
     * 
     * @param pipeline_q Q load pipeline
     * @param pipeline_k K load pipeline
     * @param pipeline_v V load pipeline
     * @param pipeline_vt V^T convert pipeline
     * @param pipeline_q_producer_state Q load pipeline state
     * @param pipeline_k_producer_state K load pipeline state
     * @param pipeline_v_producer_state V load pipeline state
     * @param pipeline_vt_producer_state V^T convert pipeline state
     */
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q, MainloopPipelineK pipeline_k,
        MainloopPipelineVT pipeline_vt,
        PipelineStateQ& pipeline_q_producer_state,
        PipelineStateKV& pipeline_k_producer_state,
        PipelineStateKV& pipeline_v_producer_state) {
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
        }
        pipeline_vt.producer_tail(pipeline_v_producer_state);
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
     * @param pipeline_q_consumer_state Q consume pipeline state
     * @param pipeline_k_consumer_state K consume pipeline state
     * @param pipeline_vt_consumer_state V^T consume pipeline state
     * @param tOrO Tensor O ((2, 2, D/8), MMA_Q, MMA_D):((1, 2, 4), 0, 0)
     * @param tLrL Tensor L (TILE_Q):(1)
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
        MainloopPipelineK pipeline_k, MainloopPipelineVT pipeline_vt,
        PipelineStateQ& pipeline_q_consumer_state,
        PipelineStateKV& pipeline_k_consumer_state,
        PipelineStateKV& pipeline_vt_consumer_state,
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
        Tensor sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.data()), 
            SmemLayoutK{});
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

        // warpgroup thread layout of MMA QK^T
        auto thread_mma_qk = tiled_mma_qk.get_slice(
            warp_group_thread_layout(warp_group_idx));
        // warpgroup thread layout of MMA PV
        auto thread_mma_pv = tiled_mma_pv.get_slice(
            warp_group_thread_layout(warp_group_idx));

        // thread partitioned shared tensor Q (MMA, MMA_Q, MMA_D, PIPE)
        Tensor tSsQ = thread_mma_qk.partition_A(sQ);
        // thread partitioned shared tensor K (MMA, MMA_KV, MMA_D, PIPE)
        Tensor tSsK = thread_mma_qk.partition_B(sK);
        // thread partitioned shared tensor V (MMA, MMA_D, MMA_KV, PIPE)
        Tensor tOsV = thread_mma_pv.partition_B(gmma_sV);

        // Allocate "fragments/descriptors"

        // thread partitioned register tensor Q (MMA, MMA_Q, MMA_D, PIPE)
        Tensor tSrQ = thread_mma_qk.make_fragment_A(tSsQ);
        // thread partitioned register tensor K (MMA, MMA_KV, MMA_D, PIPE)
        Tensor tSrK = thread_mma_qk.make_fragment_B(tSsK);
        // thread partitioned register tensor S (MMA, MMA_Q, MMA_KV)
        Tensor tSrS = partition_fragment_C(tiled_mma_qk,
            select<0, 1>(TileShapeQK{}));
        // thread partitioned register tensor V (MMA, MMA_D, MMA_KV, PIPE)
        Tensor tOrV = thread_mma_pv.make_fragment_B(tOsV);

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

        // identity tensor for position
        Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
        // offset for masking
        auto logical_offset = make_coord(get<0>(blk_coord)
            * get<0>(TileShapeQK{}), 0);
        Tensor cS = domain_offset(logical_offset, cS_base);
        auto thread_mma_cs = thread_mma_qk.get_slice(warp_group_thread_idx);
        Tensor tScS = thread_mma_cs.partition_C(cS);
        
        // Copied tensor S->V for Softmax computation
        Tensor tVrV = make_tensor<ElementAccumulatorS>(tSrS.layout());

        // thread partitioned register tensor P (MMA, MMA_Q, MMA_KV)
        Tensor tOrP_A = make_tensor(tVrV.data(), convert_layout_C_to_A(
            tVrV.layout()));
        Tensor tOrP = make_tensor_like<ElementP>(tOrP_A);  // slice out staging

        CUTE_STATIC_ASSERT_V(size<1>(tOrP) == size<1>(tOrO)); // Q

        curandStatePhilox4_32_10_t state;
        Dropout{}.init_rand(state, mainloop_params.dropout);

        // shared memory K index
        int k_index = 0;
        // shared memory V index
        int v_index = 0;

        int tile_count = mask_tile_count;
        bool two_stage = tile_count > 1;

        constexpr int D_WAIT_MAX = cute::min(size<2>(tSrQ) - 1, 7);
        constexpr int KV_WAIT_MAX = cute::min(size<2>(tOrP) - 1, 7);

        warpgroup_fence_operand(tVrV);

        // WAIT on smem_pipe_read until its data are available (phase bit flips 
        // from rdPhaseBit value)
        auto barrier_token_q = pipeline_q.consumer_try_wait(
            pipeline_q_consumer_state);
        pipeline_q.consumer_wait(pipeline_q_consumer_state, barrier_token_q);
        int q_index = pipeline_q_consumer_state.index();

        // WAIT on smem_pipe_read until its data are available (phase bit flips 
        // from rdPhaseBit value)
        auto barrier_token_k = pipeline_k.consumer_try_wait(
            pipeline_k_consumer_state);
        pipeline_k.consumer_wait(pipeline_k_consumer_state, barrier_token_k);
        // wait for K0
        k_index = pipeline_k_consumer_state.index();

        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();

        warpgroup_arrive();

        // gemm Q * K0 -> S
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int d_block = 0; d_block < size<2>(tSrQ); ++d_block) {
            // (V,Q,D) x (V,KV,D) => (V,Q,KV)
            cute::gemm(tiled_mma_qk, tSrQ(_,_,d_block, q_index),
                tSrK(_,_,d_block,k_index), tVrV);
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();

        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();

        warpgroup_wait<0>();
        warpgroup_fence_operand(tVrV);

        // release K0
        pipeline_k.consumer_release(pipeline_k_consumer_state);
        ++pipeline_k_consumer_state;

        tile_count -= 1;

        if(two_stage) {
            math_wg_order_barrier.advance();

            warpgroup_fence_operand(tSrS);

            auto barrier_token_k = pipeline_k.consumer_try_wait(
                pipeline_k_consumer_state);
            pipeline_k.consumer_wait(pipeline_k_consumer_state, 
                barrier_token_k);
            k_index = pipeline_k_consumer_state.index();

            // Order two Math WG's MMA one after the other, helps hide Epilogue
            math_wg_order_barrier.wait();

            // wait for K1
            // WAIT on smem_pipe_read until its data are available (phase bit 
            // flips from rdPhaseBit value)
            warpgroup_arrive();

            // gemm Q * K1 -> S
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
            // Unroll the D mode manually to set scale to 1
            CUTLASS_PRAGMA_UNROLL
            for (int d_block = 0; d_block < size<2>(tSrQ); ++d_block) {
                // (V,Q,D) x (V,KV,D) => (V,Q,KV)
                cute::gemm(tiled_mma_qk, tSrQ(_,_,d_block, q_index),
                    tSrK(_,_,d_block,k_index), tSrS);
                tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            }

            warpgroup_commit_batch();

            // Cue for next Math WG's MMA to start
            math_wg_order_barrier.arrive();
        }

        // compute softmax
        Scale::scale(tVrV, mainloop_params.scale);
        Mask::apply_mask(tVrV, tScS, problem_shape);
        softmax.softmax_preexp<true>(tVrV, mainloop_params.softmax);
        softmax.softmax_exp(tVrV, mainloop_params.softmax);
        Dropout{}.apply_dropout(tVrV, state, mainloop_params.dropout);
        convert(tOrP_A, tOrP);
        permute_A(tOrP);

        if(two_stage) {
            warpgroup_wait<0>();
            warpgroup_fence_operand(tSrS);
            // release K0
            pipeline_k.consumer_release(pipeline_k_consumer_state);
            ++pipeline_k_consumer_state;
            tile_count -= 1;

            copy(tSrS, tVrV);
            tScS.data() = tScS.data() + E<1>{} * get<1>(TileShapeQK{});
            warpgroup_fence_operand(tVrV);
        }

        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

        // loop:
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_count > 0; tile_count -= 1) {
            warpgroup_fence_operand(tOrP);
            warpgroup_fence_operand(tOrO);
            warpgroup_fence_operand(tSrS);

            math_wg_order_barrier.advance();

            // wait for Vi
            // WAIT on smem_pipe_read until its data are available (phase 
            // bit flips from rdPhaseBit value)
            auto barrier_token_v = pipeline_vt.consumer_try_wait(
                pipeline_vt_consumer_state);
            pipeline_vt.consumer_wait(pipeline_vt_consumer_state, 
                barrier_token_v);
            v_index = pipeline_vt_consumer_state.index();

            // wait for Ki
            // WAIT on smem_pipe_read until its data are available (phase 
            // bit flips from rdPhaseBit value)
            auto barrier_token_k = pipeline_k.consumer_try_wait(
                pipeline_k_consumer_state);
            pipeline_k.consumer_wait(pipeline_k_consumer_state, 
                barrier_token_k);
            k_index = pipeline_k_consumer_state.index();

            // Order two Math WG's MMA one after the other, helps hide 
            // Epilogue
            math_wg_order_barrier.wait();

            // gemm Q * Ki -> S
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

            warpgroup_arrive();

            // gemm P * Vi -> O
            // Unroll the KV mode manually to set scale to 1
            CUTLASS_PRAGMA_UNROLL
            for (int kv_block = 0; kv_block < size<2>(tOrP); ++kv_block) {
                // (V,Q,KV) x (V,D,KV) => (V,Q,D)
                cute::gemm(tiled_mma_pv, tOrP(_,_,kv_block),
                    tOrV(_,_,kv_block, v_index), tOrO);
                tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
            }

            // Unroll the D mode manually to set scale to 1
            CUTLASS_PRAGMA_UNROLL
            for (int d_block = 0; d_block < size<2>(tSrQ); ++d_block) {
                // (V,Q,D) x (V,KV,D) => (V,Q,KV)
                cute::gemm(tiled_mma_qk, tSrQ(_,_,d_block, q_index),
                    tSrK(_,_,d_block,k_index), tSrS);
                tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            }

            warpgroup_commit_batch();

            // Cue for next Math WG's MMA to start
            math_wg_order_barrier.arrive();

            warpgroup_fence_operand(tVrV);

            // compute softmax
            Scale::scale(tVrV, mainloop_params.scale);
            Mask::apply_mask(tVrV, tScS, problem_shape);
            softmax.softmax_preexp<false>(tVrV, mainloop_params.softmax);
            softmax.softmax_exp(tVrV, mainloop_params.softmax);
            Dropout{}.apply_dropout(tVrV, state,
                mainloop_params.dropout);

            warpgroup_wait<D_WAIT_MAX>();

            warpgroup_fence_operand(tOrP);
            warpgroup_fence_operand(tOrO);

            // release V(i-1)
            pipeline_vt.consumer_release(pipeline_vt_consumer_state);
            ++pipeline_vt_consumer_state;

            convert(tOrP_A, tOrP);
            permute_A(tOrP);
            softmax.correction_rescale(tOrO);
            tScS.data() = tScS.data() + E<1>{} * get<1>(TileShapeQK{});

            warpgroup_wait<0>();
            warpgroup_fence_operand(tSrS);

            // release V(i-1)
            pipeline_k.consumer_release(pipeline_k_consumer_state);
            ++pipeline_k_consumer_state;

            copy(tSrS, tVrV);
            warpgroup_fence_operand(tVrV);
        }

        // release Q
        pipeline_q.consumer_release(pipeline_q_consumer_state);

        if(two_stage) {
            warpgroup_fence_operand(tOrP);
            warpgroup_fence_operand(tOrO);

            math_wg_order_barrier.advance();

            // wait for Vi
            // WAIT on smem_pipe_read until its data are available (phase 
            // bit flips from rdPhaseBit value)
            auto barrier_token_v = pipeline_vt.consumer_try_wait(
                pipeline_vt_consumer_state);
            pipeline_vt.consumer_wait(pipeline_vt_consumer_state, 
                barrier_token_v);
            v_index = pipeline_vt_consumer_state.index();

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

            // compute softmax
            Scale::scale(tVrV, mainloop_params.scale);
            Mask::apply_mask(tVrV, tScS, problem_shape);
            softmax.softmax_preexp<false>(tVrV, mainloop_params.softmax);
            softmax.softmax_exp(tVrV, mainloop_params.softmax);
            Dropout{}.apply_dropout(tVrV, state,
                mainloop_params.dropout);

            warpgroup_wait<0>();

            warpgroup_fence_operand(tOrP);
            warpgroup_fence_operand(tOrO);

            // release V(i-1)
            pipeline_vt.consumer_release(pipeline_vt_consumer_state);
            ++pipeline_vt_consumer_state;

            convert(tOrP_A, tOrP);
            permute_A(tOrP);
            softmax.correction_rescale(tOrO);
        }

        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);

        math_wg_order_barrier.advance();

        // wait for Vi
        // WAIT on smem_pipe_read until its data are available (phase 
        // bit flips from rdPhaseBit value)
        auto barrier_token_v = pipeline_vt.consumer_try_wait(
            pipeline_vt_consumer_state);
        pipeline_vt.consumer_wait(pipeline_vt_consumer_state, 
            barrier_token_v);
        v_index = pipeline_vt_consumer_state.index();

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
        pipeline_vt.consumer_release(pipeline_vt_consumer_state);
        ++pipeline_vt_consumer_state;
    }
};

}  // namespace cutlass::fmha::collective

} // namespace cutlass::fmha