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
#include "cute/layout.hpp"
#include <cute/tensor.hpp>
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "algorithm/fmha_copy.hpp"
#include "algorithm/fmha_softmax.hpp"
#include "collective/fmha_varlen.hpp"

namespace cutlass::fmha {

namespace epilogue {

using namespace cute;

/**
 * @brief SM90 FMHA Forward Epilogue TMA Warp Specialized
 * 
 * @tparam CtaTileQKVD_ (CTA_Q, CTA_KV, CTA_D)
 * @tparam ElementO Type of element to Write
 */
template<class CtaTileQKVD_, class ElementO_>
struct Sm90FmhaFwdEpilogueTmaWarpspecialized {
    // total Thread block Tile
    using CtaTileQKVD = CtaTileQKVD_;
    // Thread block tile on this warpgroup
    using CtaTileO = decltype(shape_div(CtaTileQKVD{},
        Shape<_2, _1, _1>{}));
    // Epilogue Tile shape
    using EpilogueTileO = Shape<_64, _64>;
    // global memory output element type
    using ElementO = ElementO_;
    // stride type of O: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideO = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, 
        int>>;
    using Stride = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    // Shared to Global Memory Copy: TMA Store
    using CopyOpS2G = SM90_TMA_STORE;
    // Shared memory Layout Atom O matrix
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail
        ::ss_smem_selector<cute::GMMA::Major::K, ElementO, decltype(
        get<0>(EpilogueTileO{})), decltype(get<1>(EpilogueTileO{}))>());
    // Get the smallest tiled copy we can use to retile the accumulators
    using CopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
    // Store Global Memory Tiled Copy Oputput Matrix
    using GmemTiledCopyO = CopyOpS2G;

    // number of elements in a thread fragment: 4
    constexpr static int FragmentSize = size(EpilogueTileO{}) / 128;

    static_assert(!is_layout<EpilogueTileO>::value && is_tuple<
        EpilogueTileO>::value, "EpilogueTile must be a cute::Tile or "
        "cute::Shape");
    static_assert(cute::rank(CtaTileO{}) == 3, "CtaTileQKVD must be rank-3: "
        "[CTA_Q, CTA_KV, CTA_D]");
    static_assert(cute::rank(EpilogueTileO{}) == 2, "EpilogueTile must be "
        "rank-2: [EPI_TILE_Q, EPI_TILE_D]");
    static_assert(size<0>(CtaTileO{}) % size<0>(shape(EpilogueTileO{}))
        == 0, "EPI_TILE_Q must divide CTA_Q");
    static_assert(size<2>(CtaTileO{}) % size<1>(shape(EpilogueTileO{}))
        == 0, "EPI_TILE_D must divide CTA_D");
    static_assert(cute::rank(StrideO{}) == 3,
        "StrideD must be rank-3: [H_G*H_R*D, 1, [[D, H_G*D], H_G*H_R*D*Q)]]");
    static_assert(cute::get<1>(StrideO{}) == 1,
        "StrideD must be [H_G*H_R*D, 1, [[D, H_G*D], H_G*H_R*D*Q)]]");
    static_assert(cute::rank<2>(StrideO{}) == 2,
        "StrideD must be: [H_G*H_R*D, 1, [[D, H_G*D], H_G*H_R*D*Q)]]");
    static_assert(cute::rank<2,0>(StrideO{}) == 2,
        "StrideD must be: [H_G*H_R*D, 1, [[D, H_G*D], H_G*H_R*D*Q)]]");

private:
    // Number of Output Shared Memory Stages
    constexpr static int StagesO = 2;

    // Shared memory Layout O matrix
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{},
        make_shape(size<0>(EpilogueTileO{}), size<1>(EpilogueTileO{}),
        Int<StagesO>{}), Step<_1, _2, _3>{}));
    
    // Shared memory O matrix storage
    using SmemOStorage = cute::ArrayEngine<ElementO, cosize_v<SmemLayoutO>>;
    
    // Alignment of O tensor
    constexpr static size_t SmemAlignmentO = cutlass::detail
        ::alignment_for_swizzle(SmemLayoutO{});
    
    // Collective Storage for O
    struct CollectiveStorage {
        // shared memory storage of O
        alignas(SmemAlignmentO) SmemOStorage smem_o;
    };
public:
    // TMA Pipeline for storing O
    using StorePipelineO = cutlass::PipelineTmaStore<StagesO>;
    // TMA Pipeline State for storing O
    using StorePipelineStateO = cutlass::PipelineState<StagesO>;
    
    
    // register to shared memory copy
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail
        ::sm90_get_smem_store_op_for_accumulator<StrideO, ElementO>());
    // register to shared memory copy atom
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, ElementO>;

    /**
     * @brief Shared Memory Storage
     */
    struct SharedStorage {
        /**
         * @brief Tensor storage inside shared memory
         */
        struct TensorStorage {
            CollectiveStorage collective;
        } tensors;
    };

    // Tensor storage inside shared memory
    using TensorStorage = typename SharedStorage::TensorStorage;

    /**
     * @brief Host side kernel arguments
     */
    struct Arguments {
        // global memory pointer of O matrix
        ElementO* ptr_O;
        // Stride of O matrix
        StrideO dO;
    };

    // TMA store O
    using TMA_O = decltype(make_tma_copy(CopyOpS2G{}, make_tensor(
        make_gmem_ptr<ElementO>(nullptr), repeat_like(StrideO{}, int32_t(0)), 
        StrideO{}), take<0,2>(SmemLayoutO{}), EpilogueTileO{}, _1{}));

    /**
     * @brief Device side epilogue Params
     * 
     */
    struct Params {
        // TMA store O
        TMA_O tma_store_o;
    };

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
        Arguments const& args, void* workspace = nullptr) {
        // pointer to global memory O
        auto ptr_O = args.ptr_O;
        // stride of O: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
        StrideO dO = args.dO;
        // (Q, D, ((HG, HR), B))
        auto problem_shape_O = select<0,2,3>(problem_shape);

        if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
            auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
            if (cumulative_length_q != nullptr) {
                int max_length_q = get<0>(problem_shape).max_length;
                // for variable sequence lenght, the batch is in units of 
                // row_stride
                get<2,1>(dO) = get<0>(dO);
                get<2,1>(problem_shape_O) = max_length_q
                    * (1 + get<2,1>(problem_shape_O));
                // offset ptr by the amount we add back in later
                ptr_O -= max_length_q * get<0>(dO);
            }
        }

        // TMA Store O
        TMA_O tma_store_o{};
        // Output Tensor O
        Tensor tensor_o = make_tensor(make_gmem_ptr<ElementO>(ptr_O), 
            make_layout(problem_shape_O, args.dO));
        tma_store_o = make_tma_copy_C_sm90(CopyOpS2G{}, tensor_o,
            take<0,2>(SmemLayoutO{}), EpilogueTileO{});

        return {tma_store_o};
    }

    /**
     * @brief Get the workspace size
     * 
     * @tparam ProblemShape problem space type
     * @param problem_shape problem space (Q, KV, D, ((H_G, H_R), B))
     * @param args host side kernel arguments
     * @return size_t workspace size: 0
     */
    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, 
        Arguments const& args) {
        return 0;
    }

    /**
     * @brief Initialize workspace
     * 
     * @tparam ProblemShape problem shape type
     * @param problem_shape (Q, KV, D, ((H_G, H_R), B))
     * @param args user side arguments
     * @param workspace worksapce
     * @param stream CUDA stream
     * @param cuda_adapter CUDA Runtime & Driver API
     * @return cutlass::Status::kSuccess
     */
    template <class ProblemShape>
    static cutlass::Status initialize_workspace(
        ProblemShape const& problem_shape, Arguments const& args,
        void* workspace, cudaStream_t stream,
        CudaHostAdapter* cuda_adapter = nullptr) {
        return cutlass::Status::kSuccess;
    }

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
        auto [Q,KV,D,HB] = problem_shape;
        // (Q, D, ((HG, HR), B))
        auto shape = cute::make_shape(Q,D,HB);

        bool implementable = true;
        // 128 bit alignment
        constexpr int tma_alignment_bits_O = cutlass::detail
            ::get_output_alignment_bits<ElementO>();
        // 8 element alignment
        constexpr int min_tma_aligned_elements_O = tma_alignment_bits_O
            / cutlass::sizeof_bits<ElementO>::value;
        implementable = cutlass::detail
            ::check_alignment<min_tma_aligned_elements_O>(shape, StrideO{});

        if (!implementable) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the "
                "minimum alignment requirements for TMA.\n");
        }
        return implementable;
    }


    /**
     * @brief get store pipeline increment number
     * 
     * @tparam TileShape (Q, KV, D) tile shape type
     * @param tile_shape tile shape
     * @return int number of subtiles
     */
    template<class TileShapeQKVD>
    CUTLASS_HOST_DEVICE static constexpr int get_store_pipe_increment(
        TileShapeQKVD tile_shape) {
        return size<1>(zipped_divide(make_layout(select<0,2>(tile_shape)), 
            EpilogueTileO{}));
    }

    /**
     * @brief Issue Tma Descriptor Prefetch -- ideally from a single thread for 
     * best performance
     * 
     * @param params kernel parameters
     */
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
    }

    CUTLASS_HOST_DEVICE
    Sm90FmhaFwdEpilogueTmaWarpspecialized(Params const& params_,
        TensorStorage& shared_tensors) : params(params_) {}

    /**
     * @brief This Epilogue does not need load
     * 
     * @return false 
     */
    CUTLASS_DEVICE
    bool is_producer_load_needed() const {
          return false;
    }

    /**
     * @brief Perform Epilogue Computation & TMA Async Store
     * 
     * @tparam ProblemShape Coord<int, int, int, Coord<Coord<int, int>, int>>
     * @tparam ParamsProblemShape Kernel problem shape, could be variable len
     * @tparam TileShapeQKVD Shape<Q, KV, D>
     * @tparam TileCoordQHGHRB (bidq, _0, (bidh, bidb))
     * @tparam AccEngine: Output Tensor Accumulator Engine
     * @tparam AccLayout: Output Tensor Accumulator Layout
     * @tparam LEngine: L Tensor Accumulator Engine
     * @tparam LLayout: L Tensor Accumulator Layout
     * @tparam TiledMmaPV Tiled MMA PV type
     * @param store_pipeline_o O store pipeline
     * @param store_pipe_o_producer_state O store pipeline producer state
     * @param problem_shape_qkvdhghrb (q, kv, d, ((hg, hr), b))
     * @param params_problem_shape kernel parameter problem shape
     * @param blk_coord_in block coordinate (q, _0, (bidh, bidb))
     * @param accumulators Reigster Accumulator output Tensor O
     * @param tLrL L Register Tensor
     * @param shared_storage shared memory storage
     * @param tiled_mma_pv Tiled MMA PV
     * @param thread_idx Thread Index
     * @param shared_tensors Shared Memory Output Tensors
     */
    template<class ProblemShapeQKVDHGHRB, class ParamsProblemShapeQKVDHGHRB,
        class TileCoordQHGHRB, class AccEngine, class AccLayout,
        class TiledMmaPV, class Softmax>
    CUTLASS_DEVICE auto
    store(StorePipelineO& store_pipeline_o,
        StorePipelineStateO& store_pipe_o_producer_state,
        ProblemShapeQKVDHGHRB const& problem_shape_qkvdhghrb,
        ParamsProblemShapeQKVDHGHRB const& params_problem_shape_qkvdhghrb,
        TileCoordQHGHRB const& blk_coord_in, 
        cute::Tensor<AccEngine, AccLayout> accumulators,
        Softmax& softmax, TiledMmaPV tiled_mma_pv,
        int thread_idx, TensorStorage& shared_tensors,
        SoftmaxParams const& softmaxparams) {
        using namespace cute;
        using ElementAccumulator = typename AccEngine::value_type;
        using ElementCompute = ElementO;
        
        static_assert(is_rmem<AccEngine>::value, "Accumulator must be RF "
            "resident.");
        static_assert(rank(AccLayout{}) == 3, "Accumulator must be "
            "MMA-partitioned: (MMA,MMA_Q,MMA_D)");
        static_assert(rank(ProblemShapeQKVDHGHRB{}) == 4, 
            "ProblemShapeQKVDHGHRB must be rank 4");
        static_assert(is_static<CtaTileQKVD>::value, "CtaTileQKVD must be "
            "static");
        static_assert(rank(CtaTileQKVD{}) == 3, "CtaTileQKVD must be rank "
            "3");
        static_assert(rank(TileCoordQHGHRB{}) == 3, "TileCoordQHGHRB must be "
        "rank 3");
        static_assert(is_same_v<ElementAccumulator, float>);
        // Indexing variables
        auto [Q, KV, D, L] = problem_shape_qkvdhghrb;
        auto blk_coord = blk_coord_in;

        [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

        using X = Underscore;

        int o_index = get<0>(blk_coord);

        // Represent the full output tensor, slice to get the tile this CTA is 
        // responsible for
        // (Q, D, ((HG, HR), B))
        Tensor mO_qdl_p = params.tma_store_o.get_tma_tensor(
            select<0,2,3>(problem_shape_qkvdhghrb));
        // offset mode 0 by (max_length - real_length)
        // offset mode 2,1 by cumulative_length + real_length
        // the ptr is already offset by - max_length
        // so in total this achieves 
        int offs_0 = 0;
        int offs_2_1 = 0;

        if constexpr (is_variable_length_v<tuple_element_t<0, 
            ParamsProblemShapeQKVDHGHRB>>) {
            auto cumulative_length_q = get<0>(
                params_problem_shape_qkvdhghrb).cumulative_length;
            if (cumulative_length_q != nullptr) {
                int max_length_q = get<0>(
                    params_problem_shape_qkvdhghrb).max_length;
                offs_0 = max_length_q - get<0>(problem_shape_qkvdhghrb);
                offs_2_1 = cumulative_length_q[get<2,1>(blk_coord)]
                    + get<0>(problem_shape_qkvdhghrb);
                get<2,1>(blk_coord) = 0;
            }
        }

        // (seq_q, 0, ((0, hr), b))
        Tensor mO_qdl = domain_offset(make_coord(offs_0, _0{}, make_coord(_0{}, 
            offs_2_1)), mO_qdl_p);
        Tensor mO = coalesce(mO_qdl, select<0, 2>(CtaTileO{}));
        // (CTA_Q, CTA_D)
        Tensor gO = local_tile(mO, select<0, 2>(CtaTileO{}), make_coord(
            o_index, _0{}, get<2>(blk_coord)));
        
        // Apply epilogue subtiling

        // (EPI_TILE_Q,EPI_TILE_D,EPI_Q,EPI_D)
        Tensor gO_epi = flat_divide(gO, EpilogueTileO{});
        // Construct the corresponding pipelined smem tensors
        auto ptr_sO = shared_tensors.collective.smem_o.begin();
        // (EPI_TILE_Q, EPI_TILE_D, PIPE)
        Tensor sO_epi = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(ptr_sO), SmemLayoutO{}));

        // Tiled Copy Atom
        TiledCopy tiled_copy_O_atom = make_tiled_copy_C_atom(CopyAtomO{}, 
            tiled_mma_pv);
        // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
        TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<
            CopyOpR2S, ElementO>{}, tiled_copy_O_atom);
        // thread copy for register to smem copy
        ThrCopy thread_r2s = tiled_r2s.get_slice(warp_group_thread_idx);

        // thread-partition  ((R2S,R2S_V),MMA_Q,MMA_D)
        Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);
        // thread for register to smem O (R2S,R2S_Q,R2S_D,PIPE)
        Tensor tRS_sO = thread_r2s.partition_D(sO_epi);

        auto mma_tile_q = size<0>(CtaTileO{}) / size<1>(tRS_rAcc);
        auto mma_tile_d = size<2>(CtaTileO{}) / size<2>(tRS_rAcc);
        auto epi_tile_q = size<0>(EpilogueTileO{});
        auto epi_tile_d = size<1>(EpilogueTileO{});

        // Allocate D registers
        Layout tRS_rO_layout = make_layout(take<0,3>(shape(
            thread_r2s.partition_S(sO_epi))));
        // (R2S,R2S_Q,R2S_D)
        Tensor tRS_rO = make_tensor<ElementO>(tRS_rO_layout);
        
        // Vectorized fragment view
        // thread-level register fragment ((R2S,R2S_V), MMA_Q, MMA_D)
        Tensor tRS_rAcc_frg = recast<Array<ElementAccumulator,
            FragmentSize>>(tRS_rAcc);
        // thread-level shared memory fragment (R2S,R2S_Q,R2S_D)
        Tensor tRS_rO_frg = recast<Array<ElementO, FragmentSize>>(tRS_rO);
        CUTE_STATIC_ASSERT(size<0>(tRS_rAcc) % FragmentSize == 0,
            "Fragment size does not vectorize properly");

        // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
        ThrCopy thrblk_s2g = params.tma_store_o.get_slice(Int<0>{});
        // (S2G,S2G_Q,S2G_D,PIPE)
        Tensor bSG_sO = thrblk_s2g.partition_S(sO_epi);
        // (S2G,S2G_Q,S2G_D,EPI_Q,EPI_D)
        Tensor bSG_gO = thrblk_s2g.partition_D(gO_epi);

        CUTE_STATIC_ASSERT(epi_tile_q % mma_tile_q == 0, "MMA_TILE_M must "
            "divide EPI_TILE_M");

        CUTE_STATIC_ASSERT(mma_tile_d % epi_tile_d == 0, "EPI_TILE_N must "
            "divide MMA_TILE_N");

        // Thread synchronizer for previously issued waits or fences
        // to ensure visibility of smem reads/writes to threads or TMA unit
        auto synchronize = [&] () CUTLASS_LAMBDA_FUNC_INLINE { 
            cutlass::arch::NamedBarrier::sync(size(TiledMmaPV{}), 
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

        // Predication for TMA store (one warp issues TMA store)
        bool issue_tma_store = (warp_group_thread_idx / NumThreadsPerWarp) == 0;

        // We can delay issue of TMA store by one iteration to achieve better 
        // interleaving of non-TMA instructions
        // Sync requirements of smem reuse may preclude this optimization
        [[maybe_unused]] int epi_q_prev = 0;
        [[maybe_unused]] int epi_d_prev = 0;
        
        // The TMA store sequence for one subtile iteration
        auto tma_store_fn = [&] (int epi_q, int epi_d) 
            CUTLASS_LAMBDA_FUNC_INLINE {
            // Write the tile from smem to gmem with TMA
            // ensure smem writes are visible to TMA
            cutlass::arch::fence_view_async_shared();
            synchronize(); // ensure all threads have issued their async fence
            if (issue_tma_store) {
                copy(params.tma_store_o, bSG_sO(_,_,_,
                    store_pipe_o_producer_state.index()), bSG_gO(_,_,_,epi_q,
                        epi_d));
            }
    
            // Commit the TMA stores for this stage
            if (issue_tma_store) {
                store_pipeline_o.producer_commit(store_pipe_o_producer_state);
            }
            ++store_pipe_o_producer_state;
            ++issued_stores;
    
            // Wait for the next smem buffer to be available
            if (issue_tma_store) {
                store_pipeline_o.producer_acquire(store_pipe_o_producer_state);
            }
            synchronize();
        };

        static constexpr int nrows = size<0>(CtaTileO{}) / 64;

        // number of rows
        Tensor inv_tLrL = make_tensor<bfloat16_t>(Layout<Shape<_2, 
            Int<nrows>>>{});

        //
        // BEGIN EPILOGUE
        //

        // For each output tile
        CUTLASS_PRAGMA_UNROLL
        for (int epi_d = 0; epi_d < size<3>(gO_epi); ++epi_d) {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_q = 0; epi_q < size<2>(gO_epi); ++epi_q) {
                [[maybe_unused]] bool is_first_iteration = epi_q == 0
                    && epi_d == 0;
                bool is_last_iteration = epi_q == size<2>(gO_epi)-1
                    && epi_d == size<3>(gO_epi)-1;

                int mma_q = epi_q;
                int mma_d = (epi_d * size<1>(EpilogueTileO{})) / mma_tile_d;
                // (R2S,R2S_V)
                Tensor tRS_rAcc_frg_qd = tRS_rAcc_frg(_,mma_q,mma_d);

                // Vectorized fragment loop with visitor callback entry point
                int r2s_v = epi_d % (mma_tile_d / epi_tile_d);

                bool compute_inv = epi_d == 0;
                if(compute_inv) {
                    softmax.compute_inverse_L(inv_tLrL, epi_q, softmaxparams);
                }

                Tensor inv_tLrL2 = make_tensor<__nv_bfloat162>(_2{});
                inv_tLrL2(_0{}) = __bfloat162bfloat162(inv_tLrL(_0{},
                     epi_q).to_nv_bfloat16());
                inv_tLrL2(_1{}) = __bfloat162bfloat162(inv_tLrL(_1{},
                    epi_q).to_nv_bfloat16());

                tRS_rO_frg(_0{}) = cutlass::NumericArrayConverter<
                    ElementO, ElementAccumulator, FragmentSize>{}
                    (tRS_rAcc_frg_qd(r2s_v));

                Tensor O_data = recast<__nv_bfloat162>(tRS_rO_frg);

                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 16; i += 2) {
                    O_data(i) = __hmul2(O_data(i), inv_tLrL2(_0{}));
                    O_data(i + 1) = __hmul2(O_data(i + 1), inv_tLrL2(_1{}));
                }

                Tensor O_bf16 = recast<__nv_bfloat16>(tRS_rO_frg);

                // N Dimension swap
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 4; i++) {
                    cutlass::swap(O_bf16(8 * i + 1), O_bf16(8 * i + 4));
                    cutlass::swap(O_bf16(8 * i + 3), O_bf16(8 * i + 6));
                }

                bool is_even = thread_idx % 2 == 0;

                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 4; i++) {
                    int read_thr_idx = (thread_idx % 4) >> 1;
                    __nv_bfloat162 src0 = __shfl_sync(0xffffffff,
                        O_data(4 * i), read_thr_idx, 4);
                    __nv_bfloat162 src1 = __shfl_sync(0xffffffff,
                        O_data(4 * i + 2), read_thr_idx, 4);
                    __nv_bfloat162 upper = is_even ? src0 : src1;
                    src0 = __shfl_sync(0xffffffff, O_data(4 * i),
                        read_thr_idx + 2, 4);
                    src1 = __shfl_sync(0xffffffff, O_data(4 * i + 2), 
                        read_thr_idx + 2, 4);
                    __nv_bfloat162 lower = is_even ? src0 : src1;
                    O_data(4 * i) = upper;
                    O_data(4 * i + 2) = lower;

                    src0 = __shfl_sync(0xffffffff, O_data(4 * i + 1), 
                        read_thr_idx, 4);
                    src1 = __shfl_sync(0xffffffff, O_data(4 * i + 3), 
                        read_thr_idx, 4);
                    upper = is_even ? src0 : src1;
                    src0 = __shfl_sync(0xffffffff, O_data(4 * i + 1),
                        read_thr_idx + 2, 4);
                    src1 = __shfl_sync(0xffffffff, O_data(4 * i + 3), 
                        read_thr_idx + 2, 4);
                    lower = is_even ? src0 : src1;
                    O_data(4 * i + 1) = upper;
                    O_data(4 * i + 3) = lower;
                }

                
                // The latest we can delay the TMA store is right before the 
                // smem store of the next iteration
                // since the current TMA store needs to be committed before we 
                // can acquire the next smem buffer
                // Issue TMA stores for the previous subtile
                if (not is_first_iteration) {
                    tma_store_fn(epi_q_prev, epi_d_prev);
                }
                epi_q_prev = epi_q;
                epi_d_prev = epi_d;

                // Copy tile from register to smem
                copy(tiled_r2s, tRS_rO, tRS_sO(_,_,_,
                    store_pipe_o_producer_state.index()));
            } // for epi_m
        } // for epi_n

        // Issue TMA stores for the last subtile
        tma_store_fn(epi_q_prev, epi_d_prev);

        return cute::make_tuple(store_pipe_o_producer_state);
    }

    /**
     * @brief Returns pipeline producer state
     * 
     * @param pipeline_o store pipeline
     * @param store_pipe_o_producer_state store pipeline producer state 
     * @return CUTLASS_DEVICE 
     */
    CUTLASS_DEVICE auto
    store_tail(StorePipelineO& store_pipeline_o,
        StorePipelineStateO& store_pipe_o_producer_state) {
        // wait for all TMA stores to complete
        store_pipeline_o.producer_tail(store_pipe_o_producer_state);
        // reset store counter
        issued_stores = 0;

        return cute::make_tuple(store_pipe_o_producer_state);
    }

private:
    Params const& params;
    int issued_stores = 0;
};

}  // namespace cutlass::fmha::collective

} // namespace cutlass::fmha