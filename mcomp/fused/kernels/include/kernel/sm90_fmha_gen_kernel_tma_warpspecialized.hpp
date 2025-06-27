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

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/grid_dependency_control.h"

#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "algorithm/fmha_copy.hpp"
#include "algorithm/fmha_dropout.hpp"
#include "collective/fmha_mask.hpp"
#include "collective/fmha_varlen.hpp"
// test
#include "collective/sm90_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm90_fmha_fwd_epilogue_tma_warpspecialized.hpp"


namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::epilogue;

/**
 * @brief SM90 FMHA Generation Kernel TMA Warp Specialized Pingpong
 * 
 * @tparam ProblemShape_ (1, K, D, ((P H_G H_R) B))
 * @tparam CollectiveMainloop_ Collective Mainloop type
 * @tparam CollectiveEpilogue_ Collective Epilogue type
 * @tparam TileScheduler_ Tile scheduler type
 */
template<class ProblemShape_, class CollectiveMainloop_,
    class CollectiveEpilogue_, class TileScheduler_>
struct Sm90FmhaGenKernelTmaWarpspecializedPingpong {
    //
    // Type Aliases
    //
    using ProblemShape = ProblemShape_;
    
    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    // tile shape type
    // type of tile shape of QK compute: (HG, KV, D)
    using TileShape = typename CollectiveMainloop::TileShape;
    // type of tile shape of QK compute: (HG, KV, D)
    using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
    // type of tile shape of QK compute: (HG, D, KV)
    using TileShapePV = typename CollectiveMainloop::TileShapePV;
    // type of tiled MMA of QK^T
    using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
    // type of tiled MMA of PV
    using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
    // element type of Q
    using ElementQ = typename CollectiveMainloop::ElementQ;
    // stride type of Q: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideQ = typename CollectiveMainloop::StrideQ;
    // element type of K
    using ElementK = typename CollectiveMainloop::ElementK;
    // stride type of K: (H_R* D, 1, ((0, D), H_R*D*KV))
    using StrideK = typename CollectiveMainloop::StrideK;
    // element type of V
    using ElementV = typename CollectiveMainloop::ElementV;
    // stride type of V: (H_R* D, 1, ((0, D), H_R*D*KV))
    using StrideV = typename CollectiveMainloop::StrideV;
    // element type of accumulator S: half_t
    using ElementAccumulatorS
        = typename CollectiveMainloop::ElementAccumulatorS;
    // element type of accumulator O: float
    using ElementAccumulatorO
        = typename CollectiveMainloop::ElementAccumulatorO;
    // cluster shape: (1, 2, 1)
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    // Device API mainloop arguments
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    // Kernel API mainloop parameters
    using MainloopParams = typename CollectiveMainloop::Params;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    // element type of O: bfloat16_t
    using ElementO = typename CollectiveEpilogue::ElementO;
    // stride type of O: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideO = typename CollectiveEpilogue::StrideO;
    // element type of LSE: half_t
    using ElementLSE = typename CollectiveEpilogue::ElementLSE;
    // stride type of O: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideLSE = typename CollectiveEpilogue::StrideLSE;
    // stride type of O: (H_G*H_R*D, 1, ((D, H_G*D), H_G*H_R*D*Q))
    using StrideLSEMAX = typename CollectiveEpilogue::StrideLSEMAX;
    // Device API epilogue arguments
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    // Kernel API epilogue parameters
    using EpilogueParams = typename CollectiveEpilogue::Params;

    // Tile scheduler derived types
    using TileScheduler = TileScheduler_;
    // User API TileScheduler arguments
    using TileSchedulerArguments = typename TileScheduler::Arguments;
    // Kernel API TileScheduler parameters
    using TileSchedulerParams = typename TileScheduler::Params;
    // Tile Scheduler Shared Memory Storage
    using TileSchedulerStorage = typename TileScheduler::SharedStorage;

    // Warp specialization thread count per threadblock

    // number of load warpgroups
    static constexpr uint32_t NumLoadWarpGroups = 1;
    // number of MMA warpgroups
    static constexpr uint32_t NumMmaWarpGroups = 2;
    // number of total MMA threads
    static constexpr uint32_t NumMMAThreads
        = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    // maximum number of threads per thread block = 384
    static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads
        + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
    // minimum thread blocks per SM = 1
    static const int MinBlocksPerMultiprocessor = 1;

    // Register requirement for Load WGs
    static constexpr uint32_t LoadRegisterRequirement = 40;
    // Register requirement for MMA WGs
    static constexpr uint32_t MmaRegisterRequirement = 232;

    // Order Sequence barrier for store
    static constexpr uint32_t StagesPerMathWarpGroup = 2;
    using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
        StagesPerMathWarpGroup, NumMmaWarpGroups>;
    // Ordered Sequence barrier storage in shared memory
    using MathWarpGroupOrderBarrierSharedStorage =
        cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
        MathWarpGroupOrderBarrier::SequenceDepth,
        MathWarpGroupOrderBarrier::SequenceLength>;

    /**
     * @brief Kernel level shared memory storage
     */
    struct SharedStorage {
        /**
         * @brief Pipeline shared memory storage
         */
        struct PipelineStorage {
            // Storage of Load Q Pipeline type
            using MainloopPipelineQStorage = typename CollectiveMainloop
                ::PipelineQStorage;
            // Storage of Load K Pipeline type
            using MainloopPipelineKStorage = typename CollectiveMainloop
                ::PipelineKStorage;
            // Storage of Load V Pipeline type
            using MainloopPipelineVStorage = typename CollectiveMainloop
                ::PipelineVStorage;
            // Ordered Sequence barrier storage in shared memory
            using MathWarpGroupOrderBarrierStorage
                = MathWarpGroupOrderBarrierSharedStorage;
            // load q pipeline
            alignas(16) MainloopPipelineQStorage pipeline_q;
            // load k pipeline
            alignas(16) MainloopPipelineKStorage pipeline_k;
            // load v pipeline
            alignas(16) MainloopPipelineVStorage pipeline_v;
            // math workgroup order barrier
            alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
        } pipelines;

        alignas(16) TileSchedulerStorage scheduler;

        /**
         * @brief Tensor storage inside shared memory
         */
        struct TensorStorage : cute::aligned_struct<128, _1> {
            // Storage of Q, K, V tensors
            using MainloopTensorStorage = typename CollectiveMainloop
                ::TensorStorage;
            // Storage of O tensors
            using EpilogueTensorStorage = typename CollectiveEpilogue
                ::TensorStorage;
            // epilogue shared memory, O
            EpilogueTensorStorage epilogue;
            // mainloop shared memory, Q, K, V
            MainloopTensorStorage mainloop;
        } tensors;
    };
    // size of shared memory storage
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    /**
     * @brief Device side arguments
     */
    struct Arguments {
        // problem shape
        ProblemShape problem_shape{};
        // mainloop arguments
        MainloopArguments mainloop{};
        // epilogue arguments
        EpilogueArguments epilogue{};
        // hardware information
        KernelHardwareInfo hw_info{};
        // tile scheduler arguments
        TileSchedulerArguments scheduler{};
    };
    /**
     * @brief Kernel entry point API
     */
    struct Params {
        // problem shape
        ProblemShape problem_shape{};
        // mainloop parameters
        MainloopParams mainloop{};
        // epilogue parameters
        EpilogueParams epilogue{};
        // hardware information
        KernelHardwareInfo hw_info{};
        // tile scheduler parameters
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    /**
     * @brief Convert to underlying arguments. In this case, a simple copy for 
     * the aliased type.
     * 
     * @param args device arguments
     * @param workspace workspace
     * @return Params kernel parameters
     */
    static Params to_underlying_arguments(Arguments const& args,
        void* workspace) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        (void) workspace;
        // (q, kv, d, ((hg, hr), b))
        auto problem_shape = args.problem_shape;

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM "
                "count.\n For optimal performance, populate the arguments "
                "KernelHardwareInfo struct with the SM count.");
            sm_count = KernelHardwareInfo::query_device_multiprocessor_count(
                args.hw_info.device_id);
        }
        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid "
            "SM count to " << sm_count);

        // Get maximum number of clusters that could co-exist on the target 
        // device
        int max_active_clusters = args.hw_info.max_active_clusters;
        if (max_active_clusters <= 0) {
            max_active_clusters = 0;
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid "
                "max cluster count.\n For optimal performance, populate the "
                "arguments KernelHardwareInfo struct with the "
                "max_active_clusters.");
        }
        else {
            CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent "
                "grid cluster count to " << max_active_clusters);
        }
        // kernel hardware information
        KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, 
            max_active_clusters};

        // Calculate workspace pointers
        uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
        // workspace offset
        size_t workspace_offset = 0;
        // workspace pointer for epilogue
        void* epilogue_workspace = workspace_ptr + workspace_offset;
        workspace_offset += CollectiveEpilogue::get_workspace_size(
            args.problem_shape, args.epilogue);
        workspace_offset = round_nearest(workspace_offset,  
            MinWorkspaceAlignment);
        // workspace pointer for tile scheduler
        void* scheduler_workspace = workspace_ptr + workspace_offset;
        workspace_offset += TileScheduler::template get_workspace_size<
            ProblemShape, ElementAccumulatorS, ElementAccumulatorO>(args.scheduler,
            args.problem_shape, args.hw_info, NumMmaWarpGroups);
        workspace_offset = round_nearest(workspace_offset,  
            MinWorkspaceAlignment);
        // workspace pointer for mainloop
        void* mainloop_workspace = nullptr;
        constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue
            ::get_store_pipe_increment(TileShapePV{});

        return Params{args.problem_shape, CollectiveMainloop
            ::to_underlying_arguments(args.problem_shape, args.mainloop, 
            mainloop_workspace), CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, epilogue_workspace), hw_info,
            TileScheduler::to_underlying_arguments(problem_shape, TileShape{}, 
            ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, 
            NumEpilogueSubTiles)};
    }

    /**
     * @brief Can implment this kernel?
     * 
     * @param args device arguments
     * @return true can implement kernel
     * @return false cannot implement kernel
     */
    static bool can_implement(Arguments const& args) {
        bool implementable = (cute::rank(ProblemShape{}) == 4
            && cute::rank<3>(ProblemShape{}) == 2
            && cute::rank<3, 0>(ProblemShape{}) == 2);
        if (!implementable) {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape "
                "don't meet the requirements.\n");
            return implementable;
        }

        implementable &= CollectiveMainloop::can_implement(args.problem_shape,
            args.mainloop);
        implementable &= CollectiveEpilogue::can_implement(args.problem_shape,
            args.epilogue);
        implementable &= TileScheduler::can_implement(args.scheduler);
        
        return implementable;
    }

    /**
     * @brief Get the workspace size from arguments
     * 
     * @param args device side arguments
     * @return size_t workspace size
     */
    static size_t get_workspace_size(Arguments const& args) {
        // total warkspace size
        size_t workspace_size = 0;

        workspace_size += CollectiveEpilogue::get_workspace_size(
            args.problem_shape, args.epilogue);
        workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

        workspace_size += TileScheduler::template get_workspace_size<
            ProblemShape, ElementAccumulatorS, ElementAccumulatorO>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
        workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

        return workspace_size;
    }
    /**
     * @brief initialize workspace
     * @param args host arguments
     * @param workspace workspace for FMHA kernel
     * @param stream CUDA stream
     * @param cuda_adpater
     * 
     * @return cutlass::Status initialized?
     */

    static cutlass::Status initialize_workspace(Arguments const& args,
        void* workspace = nullptr, cudaStream_t stream = nullptr,
        CudaHostAdapter* cuda_adapter = nullptr) {
        Status status = Status::kSuccess;
        // workspace pointer
        uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
        // workspace offset
        size_t workspace_offset = 0;
        // number of subtiles for epilogue: 2
        static constexpr uint32_t NumEpilogueSubTiles = 2;
        // number of accumulator matrices: 2
        static constexpr uint32_t NumAccumulatorMtxs = 2;
        
        status = CollectiveEpilogue::initialize_workspace(args.problem_shape, 
            args.epilogue, workspace_ptr + workspace_offset, stream, 
            cuda_adapter);
        workspace_offset += CollectiveEpilogue::get_workspace_size(
            args.problem_shape, args.epilogue);
        workspace_offset = round_nearest(workspace_offset,  
            MinWorkspaceAlignment);

        if (status != Status::kSuccess) {
            return status;
        }
        
        status = TileScheduler::template initialize_workspace<ProblemShape, 
            ElementAccumulatorS, ElementAccumulatorO>(args.scheduler,
            workspace_ptr + workspace_offset, stream, args.problem_shape,
            args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, 
            NumAccumulatorMtxs, cuda_adapter);
        workspace_offset += TileScheduler::template get_workspace_size<
        ProblemShape, ElementAccumulatorS, ElementAccumulatorO>(args.scheduler,
            args.problem_shape, args.hw_info, NumMmaWarpGroups);
        workspace_offset = round_nearest(workspace_offset,  
            MinWorkspaceAlignment);

        if (status != Status::kSuccess) {
            return status;
        }
        return cutlass::Status::kSuccess;
    }

    /**
     * @brief Computes the kernel launch grid shape based on runtime parameters
     * 
     * @param params kernel arguemnts
     * @return dim3 grid shape
     */
    static dim3 get_grid_shape(Params const& params) {
        // Given device SM count, set grid size s.t. we do not launch more 
        // thread blocks than we can run concurrently
        TileSchedulerArguments args{};
        return TileScheduler::get_grid_shape(params.scheduler,
            params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, 
            args);
    }
    /**
     * @brief Get the block shape
     * 
     * @return dim3 (384, 1, 1) thread shape
     */
    static dim3 get_block_shape() {
        dim3 block(MaxThreadsPerBlock, 1, 1);
        return block;
    }
    
    /**
     * @brief apply variable length in the input batch
     * 
     * @param params kernel parameters
     * @param problem_shape problem shape
     * @param batch_idx batch index to apply variable length
     * @return cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, 
     * int>>
     */
    CUTLASS_DEVICE auto apply_batch(const Params &params,
        ProblemShape const& problem_shape, int batch_idx) {
        return apply_variable_length(params.problem_shape, batch_idx);
    }
    /**
     * @brief Perform Kernel
     * 
     * @param params kernel parameters
     * @param smem shared memory pointer
     */
    CUTLASS_DEVICE void operator()(const Params &params, char* smem) {
        using namespace cute;
        using X = Underscore;

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#  define ENABLE_SM90_KERNEL_LEVEL 1
#endif
// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
        printf("ERROR : Arch conditional MMA instruction used without "
            "targeting appropriate compute capability. Aborting.\n");
#else
        // Preconditions
        static_assert(cute::rank(StrideQ{}) == 3, "StrideQ must be rank-3: [Q, D, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideK{}) == 3, "StrideK must be rank-3: [K, D, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideV{}) == 3, "StrideV must be rank-3: [K, D, L]. If batch mode is not needed, set L stride to Int<0>.");
        static_assert(cute::rank(StrideO{}) == 3, "StrideO must be rank-3: [Q, D, L]. If batch mode is not needed, set L stride to Int<0>.");

        /**
         * @brief Warpgroup Role
         * 
         */
        enum class WarpGroupRole {
            // Producer warpgroup
            Producer = 0,
            // 0th consumer warpgroup
            Consumer0 = 1,
            // 1st consumer warpgroup
            Consumer1 = 2
        };
        /**
         * @brief Warp Role in a Producer Warpgroup
         */
        enum class ProducerWarpRole {
            // Mainloop TMA load & Transpose V
            Mainloop = 0,
            // Transpose
            Warp1 = 1,
            // Transpose
            Warp2 = 2,
            // Transpose
            Warp3 = 3
        };

        // Kernel level shared memory storage
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

        // thread index
        int thread_idx = int(threadIdx.x);
        // lane index
        int lane_idx = canonical_lane_idx();
        // warp index
        int warp_idx = canonical_warp_idx_sync();
        // warp index in the warpgroup
        int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
        // mma index in the warpgroup
        int mma_thread_idx = thread_idx - NumThreadsPerWarpGroup;
        // warpgroup index
        int warp_group_idx = canonical_warp_group_idx();
        // warpgroup role
        WarpGroupRole warp_group_role = WarpGroupRole(warp_group_idx);
        // warp role for producer warpgroup
        auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
        // predicated lane
        int lane_predicate = cute::elect_one_sync();
        // thread block ID in a thread block cluster
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        // consumer warpgroup index
        int warp_group_consumer_idx = warp_group_idx - NumLoadWarpGroups;

        // Issue Tma Descriptor Prefetch from a single thread
        if ((warp_idx == 0) && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }
        
        // Mainloop Q Load pipeline type
        using MainloopPipelineQ = typename CollectiveMainloop
            ::MainloopPipelineQ;
        // mainloop Q load pipeline parameters
        typename MainloopPipelineQ::Params mainloop_pipeline_q_params;

        if (warp_group_role == WarpGroupRole::Producer && producer_warp_role
            == ProducerWarpRole::Mainloop) {
                mainloop_pipeline_q_params.role = MainloopPipelineQ 
                ::ThreadCategory::Producer;
        }
        if (warp_group_role == WarpGroupRole::Consumer0
            || warp_group_role == WarpGroupRole::Consumer1) {
                mainloop_pipeline_q_params.role = MainloopPipelineQ
                ::ThreadCategory::Consumer;
        }

        // leader thread issues TMA Load instruction
        mainloop_pipeline_q_params.is_leader = (producer_warp_role
            == ProducerWarpRole::Mainloop)
            && (warp_group_role == WarpGroupRole::Producer);
        // 1 warpgroup is consumer number
        mainloop_pipeline_q_params.num_consumers = NumThreadsPerWarpGroup;
        mainloop_pipeline_q_params.transaction_bytes
            = params.mainloop.tma_transaction_bytes_q;

        // Mainloop Q load pipeline
        MainloopPipelineQ mainloop_pipeline_q(
            shared_storage.pipelines.pipeline_q, mainloop_pipeline_q_params,
            ClusterShape{});
        
        // Mainloop K Load pipeline type
        using MainloopPipelineK = typename CollectiveMainloop
            ::MainloopPipelineK;
        // mainloop K load pipeline parameters
        typename MainloopPipelineK::Params mainloop_pipeline_k_params;

        if (warp_group_role == WarpGroupRole::Producer && producer_warp_role
            == ProducerWarpRole::Mainloop) {
                mainloop_pipeline_k_params.role = MainloopPipelineK
                ::ThreadCategory::Producer;
        }
        if (warp_group_role == WarpGroupRole::Consumer0
            || warp_group_role == WarpGroupRole::Consumer1) {
                mainloop_pipeline_k_params.role = MainloopPipelineK
                ::ThreadCategory::Consumer;
        }

        // leader thread issues TMA Load instruction
        mainloop_pipeline_k_params.is_leader = (producer_warp_role
            == ProducerWarpRole::Mainloop)
            && (warp_group_role == WarpGroupRole::Producer);
        // 1 warpgroup is consumer number
        mainloop_pipeline_k_params.num_consumers = NumMMAThreads;
        mainloop_pipeline_k_params.transaction_bytes
            = params.mainloop.tma_transaction_bytes_k;
        
        // Mainloop K load pipeline
        MainloopPipelineK mainloop_pipeline_k(
            shared_storage.pipelines.pipeline_k, mainloop_pipeline_k_params,
            ClusterShape{});
        
        // Mainloop V Load pipeline type
        using MainloopPipelineV = typename CollectiveMainloop
            ::MainloopPipelineV;
        // mainloop V load pipeline parameters
        typename MainloopPipelineV::Params mainloop_pipeline_v_params;

        if (warp_group_role == WarpGroupRole::Producer) {
            if(producer_warp_role == ProducerWarpRole::Mainloop) {
                mainloop_pipeline_v_params.role = MainloopPipelineV 
                    ::ThreadCategory::ProducerConsumer;
            } else {
                mainloop_pipeline_v_params.role = MainloopPipelineV 
                    ::ThreadCategory::Consumer;
            }
        }

        // leader thread issues TMA Load instruction
        mainloop_pipeline_v_params.is_leader = (producer_warp_role
            == ProducerWarpRole::Mainloop)
            && (warp_group_role == WarpGroupRole::Producer);
        // 1 warpgroup is consumer number
        mainloop_pipeline_v_params.num_consumers = NumThreadsPerWarpGroup;
        mainloop_pipeline_v_params.transaction_bytes
            = params.mainloop.tma_transaction_bytes_v;
            
        // Mainloop V load pipeline
        MainloopPipelineV mainloop_pipeline_v(
            shared_storage.pipelines.pipeline_v, mainloop_pipeline_v_params,
            ClusterShape{});
        
        // Epilogue Store pipeline
        using EpiStorePipelineO = typename CollectiveEpilogue::StorePipelineO;
        // epilogue O store pipeline parameters
        typename EpiStorePipelineO::Params epi_store_pipeline_o_params;
        epi_store_pipeline_o_params.always_wait = true;
        // Epilogue O store pipeline
        EpiStorePipelineO epi_store_pipeline_o(epi_store_pipeline_o_params);

        // Math Workgroup Ordered Barrier Parameters
        typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
        // DMA Load WG will not participate in these Ordered Barrier syncs
        params_math_wg_order_barrier.group_id = warp_group_consumer_idx;
        // Number of threads / participants in a group
        params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup;
        MathWarpGroupOrderBarrier math_wg_order_barrier(
            shared_storage.pipelines.math_wg_order, 
            params_math_wg_order_barrier);

        // Initialize starting pipeline states for the collectives
        // Epilogue store pipe is producer-only (consumer is TMA unit,
        // waits via scoreboarding)

        // mainloop load Q pipeline consumer state
        typename CollectiveMainloop::PipelineStateQ 
            mainloop_pipe_q_consumer_state;
        // mainloop load K pipeline consumer state
        typename CollectiveMainloop::PipelineStateKV 
            mainloop_pipe_k_consumer_state;
        // mainloop transpose V^T pipeline consumer state
        typename CollectiveMainloop::PipelineStateKV
            mainloop_pipe_v_consumer_state;

        // For the DMA Load (producer) we start with an opposite phase
        // i.e., we skip all waits since we know that the buffer is indeed empty

        // mainloop load Q pipeline producer state
        PipelineState mainloop_pipe_q_producer_state = cutlass
            ::make_producer_start_state<MainloopPipelineQ>();
        // mainloop load K pipeline producer state
        PipelineState mainloop_pipe_k_producer_state = cutlass
            ::make_producer_start_state<MainloopPipelineK>();
        // mainloop load V pipeline producer state
        PipelineState mainloop_pipe_v_producer_state = cutlass
            ::make_producer_start_state<MainloopPipelineV>();
        // epilogue store O pipeline producer state
        PipelineState epi_store_pipe_o_producer_state = cutlass::make_producer_start_state<EpiStorePipelineO>();
        
        // Get the appropriate blocks for this thread block -- potential for thread block locality
        
        // Tiled MMA for QK^T
        TiledMmaQK tiled_mma_qk;
        // Tiled MMA for PV
        TiledMmaPV tiled_mma_pv;
        // (BLK_Q,BLK_KV,BLK_D)
        auto blk_shape = TileShape{};

        // In a warp specialized kernel, collectives expose data movement
        // and compute operations separately

        // Mainloop
        CollectiveMainloop collective_mainloop;
        // Epilogue
        CollectiveEpilogue collective_epilogue(params.epilogue,
            shared_storage.tensors.epilogue);
        
        // number of O tiles in block tile
        auto o_tile_count = CollectiveEpilogue::get_store_pipe_increment(
            blk_shape);
        int q_offset = 0;
        
        // Persistent tile scheduler
        TileScheduler scheduler{params.scheduler, shared_storage.scheduler};
        if (warp_group_role == WarpGroupRole::Consumer1) {
            // Advance 2nd Math WG to the next work tile for the startup
            q_offset = 1;
            // Advance 2nd Math WG pipeline states to the end of 1st Math WG
            mainloop_pipe_q_consumer_state.advance(1);
            epi_store_pipe_o_producer_state.advance(o_tile_count);
        }

        // work tile information
        auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

        // Wait for all thread blocks in the Cluster
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();

        if(not work_tile_info.is_valid())
            return;

        if (producer_warp_role == ProducerWarpRole::Mainloop) {
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            // Ensure that the prefetched kernel does not touch
            // unflushed global memory prior to this instruction
            cutlass::arch::wait_on_dependent_grids();
        } else if (warp_group_role == WarpGroupRole::Consumer0
            || warp_group_role == WarpGroupRole::Consumer1) {
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
            // It is possible to have work tiles start off invalid,
            // so we have to check that first.
            if (not scheduler.is_valid()) {
                // Hint on an early release of global memory resources.
                // The timing of calling this function only influences performance,
                // not functional correctness.
                cutlass::arch::launch_dependent_grids();

                return;
            }
            #endif
        }

        CUTLASS_PRAGMA_NO_UNROLL
        while(work_tile_info.is_valid()) {
            // (bidq, 0, (bidh, bidb))
            auto blk_coord = make_coord(work_tile_info.Q_idx, _0{},
                make_coord(work_tile_info.HG_idx
                + size<3, 0, 0>(params.problem_shape)
                * work_tile_info.HR_idx, work_tile_info.B_idx));

            // Separate out problem shape for convenience
            // (1, K, D, ((P, H_G, H_R), B))
            auto logical_problem_shape = apply_batch(params,
                params.problem_shape, get<2,1>(blk_coord));

            // number of horizontal trips for compute
            int mask_tile_count = Mask{}.get_trip_count(blk_coord, 
                TileShape{}, ClusterShape{}, logical_problem_shape);

            if (warp_group_role == WarpGroupRole::Producer) {
                cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
                // drops computing FMHA for out of range in Q dimension

                // Prepare and partition the input tensors. Expects a tuple of 
                // tensors where: get<0>(load_inputs) is the tma tensor A after 
                // local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
                // get<1>(load_inputs) is the tma tensor B after local tiling 
                // so that it has shape (BLK_N,BLK_K,n,k,l)
                auto load_inputs = collective_mainloop.load_init(blk_coord, 
                    logical_problem_shape, params.problem_shape,
                    params.mainloop);
                static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 5,  
                    "Output of load_init must have at least five elements (Q, "
                    "K, V, COORD_Q, COORD_KV)");

                auto blk_coord_q = get<3>(load_inputs);
                auto blk_coord_kv = get<4>(load_inputs);
                auto input_tensors = cute::make_tuple(get<0>(load_inputs), 
                    get<1>(load_inputs), get<2>(load_inputs));

                collective_mainloop.load(params.mainloop, mainloop_pipeline_q, 
                    mainloop_pipeline_k, mainloop_pipeline_v, 
                    mainloop_pipeline_v, mainloop_pipe_q_producer_state, 
                    mainloop_pipe_k_producer_state, 
                    mainloop_pipe_v_producer_state, input_tensors, 
                    blk_coord_q, blk_coord_kv, mask_tile_count, 
                    block_rank_in_cluster, shared_storage.tensors.mainloop);
            } else if (warp_group_role == WarpGroupRole::Consumer0
                || warp_group_role == WarpGroupRole::Consumer1) {
                cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
                auto bidq = 2 * get<0>(blk_coord) + q_offset;
                auto bidhghrb = get<2>(blk_coord);
                auto mma_blk_coord = make_coord(bidq, _0{}, bidhghrb);

                // Allocate the accumulators for the (Q,D) blk_shape
                // (MMA,MMA_Q,MMA_D)=((2,2,D/8),MMA_Q,MMA_D):((1,2,4),0,0)
                Tensor tOrO = partition_fragment_C(tiled_mma_pv,
                    select<0,1>(TileShapePV{}));
                static constexpr int nrows = size<0>(TileShapePV{}) / 64;
                // (TILE_Q):(1)
                Softmax<nrows> softmax;
 
                collective_mainloop.mma(mainloop_pipeline_q,
                    mainloop_pipeline_k, mainloop_pipeline_v, 
                    mainloop_pipe_q_consumer_state, 
                    mainloop_pipe_k_consumer_state, 
                    mainloop_pipe_v_consumer_state, tOrO, softmax, 
                    mask_tile_count, mma_thread_idx,
                    shared_storage.tensors.mainloop,
                    params.mainloop, mma_blk_coord, logical_problem_shape, 
                    math_wg_order_barrier);
    
                #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
                if (scheduler.is_last_tile(work_tile_info)) {
                    // Hint on an early release of global memory resources.
                    // The timing of calling this function only influences 
                    // performance, not functional correctness.
                    cutlass::arch::launch_dependent_grids();
                    }
                #endif
    
                // Order two Math WG's Epilogue one after the other
                math_wg_order_barrier.wait();
                    
                // Epilogue and write to gO
                auto [epi_store_pipe_o_producer_state_next]
                    = collective_epilogue.store(epi_store_pipeline_o, 
                    epi_store_pipe_o_producer_state, logical_problem_shape, 
                    params.problem_shape, mma_blk_coord, tOrO, softmax, 
                    tiled_mma_pv, mma_thread_idx,
                    shared_storage.tensors.epilogue, params.mainloop.softmax);
    
                // TMA store pipeline wait is only visible to TMA-issuing warp, 
                // so for multiple-consumer kernels
                // we need to wait for all TMA stores to complete before 
                // issuing consumer order barrier arrives
                // to ensure next math consumer doesn't overwrite smem of 
                // in-flight TMA stores of current consumer.
                auto [epi_store_pipe_o_producer_state_next_]
                    = collective_epilogue.store_tail(epi_store_pipeline_o, 
                    epi_store_pipe_o_producer_state_next);
                    
                // Update starting load/store pipeline states for the next tile
                // state has already been incremented by 1 tile in collective 
                // calls, advance once again for ping pong
                epi_store_pipe_o_producer_state
                    = epi_store_pipe_o_producer_state_next_;
                epi_store_pipe_o_producer_state.advance(o_tile_count);
    
                // Cue for next Math WG's Epilogue to start
                math_wg_order_barrier.arrive();
            }
            // Get next work tile
            scheduler.advance_to_next_work(1, shared_storage.scheduler);
            work_tile_info = scheduler.get_current_work();
        }

        if(warp_group_role == WarpGroupRole::Producer) {
            // Make sure all Consumer Warp Groups have been waited upon
            collective_mainloop.load_tail(mainloop_pipeline_q, 
                mainloop_pipeline_k, mainloop_pipeline_v, mainloop_pipeline_v,
                mainloop_pipe_q_producer_state, mainloop_pipe_k_producer_state, 
                mainloop_pipe_v_producer_state);
        }
#endif
    }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel
