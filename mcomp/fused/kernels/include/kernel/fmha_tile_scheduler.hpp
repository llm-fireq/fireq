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
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"
#include "../src/fmha_coord.h"
#include "cooperative_groups.h"
#include "cooperative_groups/details/sync.h"
#include "cooperative_groups/details/reduce.h"

namespace cutlass::fmha::kernel {

////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Get the maximum CTA occupancy
 * 
 * @param max_sm_per_gpc maximum SMs per GPC
 * @param cluster_shape Thread block cluster shape
 * @param sm_count number of SMs to use
 * @return uint32_t number of CTAs per device
 */
CUTLASS_HOST_DEVICE
static uint32_t get_max_cta_occupancy(int max_sm_per_gpc,
    FmhaCoord cluster_shape, int sm_count) {
    // Provided SM count could possibly be less than the assumed maximum SMs 
    // per GPC
    auto cluster_size = cluster_shape.q() * cluster_shape.kv();
    // minimum number of GPC
    int const min_num_gpc = sm_count < max_sm_per_gpc ? 1 : sm_count
        / max_sm_per_gpc;
    // maximum CTA occupency per GPC
    int const max_cta_occupancy_per_gpc = max_sm_per_gpc - (max_sm_per_gpc
        % cluster_size);
    // number of CTAs per device
    int cta_per_device = min_num_gpc * max_cta_occupancy_per_gpc;

    // The calculation below allows for larger grid size launch for different 
    // GPUs.
    int const num_gpc_residual = sm_count < max_sm_per_gpc ? 0 : sm_count
        % max_sm_per_gpc;
    // maximum CTA occupency per residual GPC
    int const max_cta_occupancy_per_residual_gpc = num_gpc_residual
        - (num_gpc_residual % cluster_size);
    cta_per_device += max_cta_occupancy_per_residual_gpc;

    cta_per_device = sm_count < cta_per_device ? sm_count : cta_per_device;
    return cta_per_device;
}

/**
 * @brief Parameters for FMHA Split Q SM90 persistent tile scheduler
 * 
 */
struct FmhaSplitQPersistentTileSchedulerSm90Params { 
    // fast divide module for major cluster shape: /% 2
    FastDivmodU64Pow2 divmod_cluster_shape_major_{};
    // fast divide module for major thread block: /% (# of Q blocks / 2)
    FastDivmodU64 divmod_cluster_blk_major_{};
    // fast divide module for q blocks: /% # of Q blocks
    FastDivmodU64 divmod_q_block_{};
    // fast divide module for heads: /% grouped heads
    FastDivmodU64 divmod_h{};
    // fast divide module for batches: /% kv heads
    FastDivmodU64 divmod_b{};
    // number of blocks per problem
    uint64_t blocks_per_problem_ = 0;
    // number of tiles in Q dimension
    uint32_t problem_tiles_q_ = 0;
    // number of tiles in KV dimension
    uint32_t problem_tiles_kv_ = 1;
    // number of tiles in D dimension, fixed to 1
    uint32_t problem_tiles_d_ = 1;
    // number of tiles in grouped head dimension
    uint32_t problem_tiles_hg_ = 0;
    // number of tiles in KV head dimension
    uint32_t problem_tiles_hr_ = 0;
    // number of tiles in batch dimension
    uint32_t problem_tiles_b_ = 0;
    // number of thread blocks in Q dimension
    uint32_t cluster_shape_q_ = 0;
    // number of thread blocks in KV dimension
    uint32_t cluster_shape_kv_ = 1;
    // number of cluster tiles
    int num_clusters = 0;
    // tile count global memory semaphore pointer
    int* tile_count_semaphore = nullptr;
  
    // Initializes members. This variant of the method should only be used when
    // problem_shape and tile_shape contain modes of only rank 1.

    /**
     * @brief Initializes members. This variant of the method should only be 
     * used when problem_shape and tile_shape contain modes of only rank 1.
     * 
     * @param problem_shape (q, kv, d, hg, hr, d)
     * @param tile_shape (TILE_Q, TILE_KV, TILE_D)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV=1, CLUSTER_D=1)
     * @param hw_info kernel hardware information
     * @param tile_count_semaphore_ Tile counting global memory semaphore
     */
    void
    initialize(BatchedFmhaCoord problem_shape, FmhaCoord tile_shape,
        FmhaCoord cluster_shape, KernelHardwareInfo const& hw_info,
        int* tile_count_semaphore_) {
        dim3 problem_blocks = get_tiled_cta_shape_qdhghrb(problem_shape, 
            tile_shape, cluster_shape);
        initialize(problem_blocks, cluster_shape, hw_info,
            tile_count_semaphore_);
        problem_tiles_hg_ = problem_shape.hg();
        problem_tiles_hr_ = problem_shape.hr();
        problem_tiles_b_ = problem_shape.batch();

        divmod_h = FastDivmodU64(problem_tiles_hg_);
        divmod_b = FastDivmodU64(problem_tiles_hr_);
    }

    /**
     * @brief Version of initialize that takes in as input the number of CTAs 
     * in the Q and KV and HG, HR, B dimensions. This is useful for calculating 
     * the tiled shape when a mode of problem and/or CTA shape has rank > 1, 
     * for which using CuTe algebra for calculating tile shapes is easiest.
     * 
     * @param problem_blocks (q, d=1, hg * hr * b)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV=1, CLUSTER_D=1)
     * @param hw_info kernel hardware information
     * @param tile_count_semaphore_ Tile counting global memory semaphore
     */
    void
    initialize(dim3 problem_blocks, FmhaCoord cluster_shape,
        KernelHardwareInfo const& hw_info, int* tile_count_semaphore_) {
        assert(problem_blocks.y == 1);
        assert(cluster_shape.kv() == 1);
        assert(cluster_shape.d() == 1);
  
        CUTLASS_UNUSED(hw_info);
  
        // Round up to cluster size along each mode
        auto problem_blocks_q = round_up(problem_blocks.x, cluster_shape.q());
    
        problem_tiles_q_ = problem_blocks_q / cluster_shape.q();
        cluster_shape_q_ = cluster_shape.q();
  
        //
        // Set members
        //
        blocks_per_problem_ = problem_blocks_q * problem_blocks.z;
        divmod_q_block_ = FastDivmodU64(problem_blocks_q);
    
        // Always Along Q
        divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.q());
        divmod_cluster_blk_major_ = FastDivmodU64(problem_blocks_q
            / cluster_shape.q());

        num_clusters = (int)(blocks_per_problem_ / 2UL - 1UL);
        
        tile_count_semaphore = tile_count_semaphore_;
    }

    /**
     * @brief Given the inputs, computes the physical grid we should launch. 
     * This variant of the method should only be used when problem_shape and 
     * tile_shape contain modes of only rank 1.
     * 
     * @param problem_shape (q, kv, d, hg, hr, b)
     * @param cta_shape (TILE_Q, TILE_KV, TILE_D)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV=1, CLUSTER_D=1)
     * @param hw_info kernel hardware information
     * @param truncate_by_problem_size reduce grid to KV
     * @param bypass_sm90_occupancy_calculation bypassing occupancy calculation
     * @return dim3 grid shape (CLUSTER_Q, # of CLUSTERS, 1)
     */
    CUTLASS_HOST_DEVICE static
    dim3 get_grid_shape(BatchedFmhaCoord problem_shape, FmhaCoord cta_shape,
        FmhaCoord cluster_shape, KernelHardwareInfo hw_info,
        bool truncate_by_problem_size=true,
        bool bypass_sm90_occupancy_calculation=false) {
        // (q, 1, hg * hr * b)
        dim3 problem_blocks = get_tiled_cta_shape_qdhghrb(problem_shape, 
            cta_shape, cluster_shape);
        return get_grid_shape(problem_blocks, cluster_shape, hw_info,
            truncate_by_problem_size, bypass_sm90_occupancy_calculation);
    }

    /**
     * @brief Version of get_grid_shape that takes in as input the number of 
     * CTAs in the M and N and L dimensions. This is useful for calculating the 
     * tiled shape when a mode of problem and/or CTA shape has rank > 1, for 
     * which using CuTe algebra for calculating tile shapes is easiest.
     * 
     * @param problem_blocks (q, d=1, hg * hr * b)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV=1, CLUSTER_D=1)
     * @param hw_info kernel hardware information
     * @param truncate_by_problem_size reduce grid to KV
     * @param bypass_sm90_occupancy_calculation bypassing occupancy calculation
     * @return dim3 grid shape (CLUSTER_Q, # of CLUSTERS, 1)
     */
    CUTLASS_HOST_DEVICE static
    dim3 get_grid_shape(dim3 problem_blocks, FmhaCoord cluster_shape,
        KernelHardwareInfo hw_info, bool truncate_by_problem_size=true,
        bool bypass_sm90_occupancy_calculation=false) {
        int const sm_count = hw_info.sm_count;
        int const max_active_clusters = hw_info.max_active_clusters;
    
        // Round up to nearest multiple of swizzle_size along each mode
        auto problem_blocks_q = round_up(problem_blocks.x,  cluster_shape.q());
        // roundup(q, 2) * hg * hr * b
        int problem_blocks_total = problem_blocks_q * problem_blocks.z;
    
        // to be (# blocks, 1, 1)
        dim3 launch_grid;
        launch_grid = dim3(1, 1, 1);
  
        auto possibly_truncate = [&](int x, int y) {
            if (truncate_by_problem_size) {
                return platform::min(x, y);
            }
            else {
                return x;
            }
        };
  
        // The else path is generic, however, we can avoid some divs if we know 
        // cluster size is 1
        auto cluster_size = cluster_shape.q();
        if (cluster_size == 1) {
            // Always Along Q
            launch_grid.x = possibly_truncate(sm_count, problem_blocks_total);
        }
        // In case the maximum number of clusters that could co-exist on the 
        // target device is already calculated using 
        // cudaOccupancyMaxActiveClusters
        else if (max_active_clusters != 0 && max_active_clusters * cluster_size 
            <= sm_count) {
            // Always Along Q
            launch_grid.x = possibly_truncate(max_active_clusters
                * cluster_shape.q(), problem_blocks_total);
            CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the "
                "scheduler using cudaOccupancyMaxActiveClusters = "
                "(" << launch_grid.x << ", " << launch_grid.y << ", "
                << launch_grid.z << ")\n");
        }
        else {
            // number of CTAs per device = 132
            int cta_per_device = sm_count;
            if (!bypass_sm90_occupancy_calculation) { 
            /*
            * Optimal grid size calculation is based on
            * GH100: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 
            * 144 SMs per full GPU. Hence, maximum SMs per GPC = 18
            */
            constexpr int max_sm_per_gpc = 18;
            cta_per_device = get_max_cta_occupancy(max_sm_per_gpc, 
                cluster_shape, sm_count);
            } 
            // Always Along Q
            launch_grid.x = possibly_truncate(cta_per_device,
                problem_blocks_total);
            CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the "
                "scheduler using heuristics = "
                "(" << launch_grid.x << ", " << launch_grid.y << ", "
                << launch_grid.z << ")\n");
        }

        return launch_grid;
    }

    /**
     * @brief Get the number of CTA tiles in this problem. This variant of the 
     * method should only be used when problem_shape and tile_shape contain 
     * modes of only rank 1.
     * 
     * @param problem_shape (q, kv, d, hg, hr, b)
     * @param cta_shape (TILE_Q, TILE_KV, TILE_D)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV = 1, CLUSTER_D = 1)
     * @return dim3 ((q / TILE_Q) / CLUSTER_Q, 1, hg * hr * b)
     */
    CUTLASS_HOST_DEVICE
    static dim3 get_tiled_cta_shape_qdhghrb(BatchedFmhaCoord problem_shape, 
        FmhaCoord cta_shape, FmhaCoord cluster_shape) {
        // number of CTAs in Q direction
        auto cta_q = (problem_shape.q() + cta_shape.q() - 1) / cta_shape.q();
    
        return get_tiled_cta_shape_qdhghrb(problem_shape, cluster_shape, 
            cta_q);
    }

    /**
     * @brief Version of get_tiled_cta_shape_mnl that takes in as input the 
     * number of CTAs in the M and N dimensions. This is useful for calculating 
     * the tiled shape when a mode of problem and/or CTA shape has rank > 1, 
     * for which using CuTe algebra for calculating tile shapes is easiest.
     * 
     * @param problem_shape ((q / TILE_Q), kv, d, hg, hr, b)
     * @param cluster_shape (CLUSTER_Q, CLUSTER_KV = 1, CLUSTER_D = 1)
     * @param cta_q number of thread blocks along Q
     * @param cta_d number of thread blocks along D = 1
     * @return dim3 ((q / TILE_Q) / CLUSTER_Q, 1, hg * hr * b)
     */
    CUTLASS_HOST_DEVICE
    static dim3 get_tiled_cta_shape_qdhghrb(BatchedFmhaCoord problem_shape, 
        FmhaCoord cluster_shape, uint32_t cta_q, uint32_t cta_d = 1) {
        // Round up to nearest multiple of cluster dim along each mode
        auto problem_blocks_q = ((cta_q + cluster_shape.q() - 1) / cluster_shape.q()) * cluster_shape.q();
    
        return {
            static_cast<uint32_t>(problem_blocks_q), 1,
            static_cast<uint32_t>(problem_shape.hg() * problem_shape.hr()
                * problem_shape.batch())
      };
    }
};

////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Persistent Thread Block (TB) scheduler
 */
struct FmhaFwdSplitQPersistentTileScheduler {
private:
    // current workload index in linear format
    int64_t current_work_linear_idx_;
    // total grid size
    uint64_t total_grid_size_;
public:
    /**
     * @brief Workload tile information
     * 
     */
    struct WorkTileInfo {
        // Query tile index
        int32_t Q_idx = 0;
        // Key, value tile index
        int32_t KV_idx = 0;
        // Head dimension tile index
        int32_t D_idx = 0;
        // Grouped query head index
        int32_t HG_idx = 0;
        // Key, value head index
        int32_t HR_idx = 0;
        // batch index
        int32_t B_idx = 0;
        // is this tile valid?
        bool is_valid_tile = false;

        /**
         * @brief is this tile valid?
         * 
         * @return true valid tile
         * @return false invalid tile
         */
        CUTLASS_HOST_DEVICE bool is_valid() const {
            return is_valid_tile;
        }
    
        /**
         * @brief return a invalid work tile
         * 
         * @return WorkTileInfo
         */
        CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() {
            return {-1, -1, -1, -1, -1, -1, false};
        }
    
        /**
         * @brief Unused. Always return true
         * 
         * @param k_tiles_per_output_tile unused
         * @return true
         */
        CUTLASS_HOST_DEVICE bool is_final_split(
            uint32_t k_tiles_per_output_tile) const {
            return true;
        }
    
        /**
         * @brief Unused. Always return -1
         * 
         * @return int32_t -1
         */
        CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const {
            return -1;
        }
    };

    /**
     * @brief Parameters for persistent tile scheduler
     */
    using Params = FmhaSplitQPersistentTileSchedulerSm90Params;

public:
    /**
     * @brief Arguments for persistent tile scheduler
     */
    struct Arguments {};

    using SharedStorage = int;
    // Sink scheduler params as a member
    Params scheduler_params;
public:
    
    /**
     * @brief Map user facing arguments to device facing params
     * 
     * @tparam ProblemSize Problem size type
     * @tparam ClusterShape Thread block cluster shape type
     * @tparam TileShape tile shape type
     * @param problem_size_qkvdhghrb problem size (Q, KV, D, ((H_G, H_R), B))
     * @param tile_shape tile shape
     * @param cluster_shape cluster shape
     * @param hw_info hardware information
     * @param arguments Tile scheduler arguments
     * @param workspace global memory tile count semaphore
     * @param epilogue_subtile number of epilogue subtiles
     * @return Params Kernel parameters
     */
    template <class ProblemShapeQKVDHGHRB, class TileShape, class ClusterShape>
    static Params
    to_underlying_arguments(ProblemShapeQKVDHGHRB problem_shape_qkvdhghrb,
        TileShape tile_shape, ClusterShape cluster_shape,
        KernelHardwareInfo const& hw_info, Arguments const& arguments,
        void* workspace, const uint32_t epilogue_subtile) {
        
        // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
        static_assert(cute::is_static<TileShape>::value);
        static_assert(cute::is_static<ClusterShape>::value);

        Params params;
        params.initialize(to_fmha_coord(problem_shape_qkvdhghrb),
            to_fmha_coord(tile_shape), to_fmha_coord(cluster_shape),
            hw_info, (int *)workspace);

        return params;
    }

    /**
     * @brief Always can implement
     * 
     * @param args Device API arguments
     * @return true
     */
    CUTLASS_HOST_DEVICE
    static bool can_implement(Arguments const& args) {
        return true;
    }

    CUTLASS_HOST_DEVICE
    FmhaFwdSplitQPersistentTileScheduler() { }

    /**
     * @brief Constructor
     * 
     * @param params_ Device API arguments
     * @param tile_count_smem shared memoyr for tile counting
     */
    CUTLASS_DEVICE explicit FmhaFwdSplitQPersistentTileScheduler(
        Params const& params_, SharedStorage& tile_count_smem)
        : scheduler_params(params_) {
        // MSVC requires protecting use of CUDA-specific nonstandard syntax,
        // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)

        int blk_rank = block_rank_in_cluster();
        dim3 cls_shape = cluster_shape();
        int cls_size = cls_shape.x * cls_shape.y * cls_shape.z;

        int linear_idx;

        // CIDX = 0, TIDX = 0 thread gets next linear index
        if(block_rank_in_cluster() == 0 && threadIdx.x == 0) {
            tile_count_smem = atomicSub(scheduler_params.tile_count_semaphore, 
                1);
            cutlass::arch::fence_view_async_shared();
        }

        cute::cluster_arrive_relaxed();
        cute::cluster_wait();

        cooperative_groups::cluster_group cluster
            = cooperative_groups::this_cluster();

        int *dst_smem = cluster.map_shared_rank(
            &tile_count_smem, 0);

        // elected threads in a warp reads index
        if(threadIdx.x % 32 == 0) {
            linear_idx = *dst_smem;
        }

        // broadcast to other threads in a warp
        linear_idx = __shfl_sync(0xffffffff, linear_idx, 0);

        // all threads in thread block cluster gets the tile number
        current_work_linear_idx_ = (int64_t)(cls_size * linear_idx + blk_rank);

        total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y)
            * uint64_t(gridDim.z);

#else
        CUTLASS_ASSERT(false && "This line should never be reached");
#endif
    }

    /**
     * @brief Returns the initial work tile info that will be computed over
     * 
     * @tparam ClusterShape thread block cluster shape
     * @param cluster_shape (q, kv, 1) = (2, 1, 1)
     * @return WorkTileInfo work tile information
     */
    template <class ClusterShape>
    CUTLASS_DEVICE
    WorkTileInfo initial_work_tile_info(ClusterShape cluster_shape) {
        return get_current_work();
    }

    /**
     * @brief Get the current work tile
     * 
     * @return WorkTileInfo current work tile
     */
    CUTLASS_DEVICE
    WorkTileInfo get_current_work() const {
        return get_current_work_for_linear_idx(current_work_linear_idx_);
    }

    /**
     * @brief Get the current work tile for linear index
     * 
     * @param linear_idx linear index
     * @return WorkTileInfor
     */
    CUTLASS_DEVICE
    WorkTileInfo get_current_work_for_linear_idx(int64_t linear_idx) const {
        if (linear_idx < 0LL) {
            return WorkTileInfo::invalid_work_tile();
        }

        // Map worker's linear index into the CTA tiled problem shape to the 
        // corresponding MNL indices
        uint64_t work_idx_hghrb, blk_per_grid_dim;
        // work_idx_hghrb = linear_idx / num_q_blocks;
        // blk_per_grid_dim = linear_idx % num_q_blocks;
        scheduler_params.divmod_q_block_(work_idx_hghrb,
            blk_per_grid_dim, (uint64_t)linear_idx);

        // Q_idx, 0
        auto [work_idx_q, work_idx_d] = get_work_idx_q_and_d(blk_per_grid_dim,
            scheduler_params.divmod_cluster_shape_major_,
            scheduler_params.divmod_cluster_blk_major_);

        uint64_t work_idx_hg, work_idx_hr, work_idx_b;
        // work_idx_hr = work_idx_hghrb / hg
        // work_idx_hg = work_idx_hghrb % hg
        scheduler_params.divmod_h(work_idx_hr, work_idx_hg, work_idx_hghrb);

        // work_idx_b = work_idx_hr / hr
        // work_idx_hr = work_idx_hr % hr
        scheduler_params.divmod_b(work_idx_b, work_idx_hr, work_idx_hr);

        return {work_idx_q, 0, work_idx_d, static_cast<int32_t>(work_idx_hg), 
            static_cast<int32_t>(work_idx_hr), static_cast<int32_t>(work_idx_b),
            true};
    }

    /**
     * @brief get work_idx_q, work_idx_d from blk_per_grid_dim
     * 
     * @param blk_per_grid_dim number of blocks per grid dimension
     * @param divmod_cluster_shape_major /% 2
     * @param divmod_cluster_blk_major /% (BLOCKS_Q / 2)
     * @return cute::tuple<int32_t, int32_t> (Q_idx, 0)
     */
    static CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t> get_work_idx_q_and_d(
        uint64_t blk_per_grid_dim,
        FastDivmodU64Pow2 const& divmod_cluster_shape_major,
        FastDivmodU64 const& divmod_cluster_blk_major) {
        auto [cta_q_in_cluster, _, __]
            = cute::block_id_in_cluster();

        return get_work_idx_q_and_d(blk_per_grid_dim, 
            divmod_cluster_shape_major, divmod_cluster_blk_major, 
            cta_q_in_cluster);
    }

    /**
     * @brief Get the work idx q and kv
     * 
     * @param blk_per_grid_dim number of blocks per grid dimension
     * @param divmod_cluster_shape_major /% 2
     * @param divmod_cluster_blk_major /% (BLOCKS_Q / 2)
     * @param cta_q_in_cluster number of Q CTAs in cluster = 2
     * @param cta_d_in_cluster number of D CTAs in cluster = 1
     * @return cute::tuple<int32_t, int32_t> (major_idx, minor_idx)
     */
    static CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t> get_work_idx_q_and_d(
        uint64_t blk_per_grid_dim,
        FastDivmodU64Pow2 const& divmod_cluster_shape_major,
        FastDivmodU64 const& divmod_cluster_blk_major,
        uint64_t cta_q_in_cluster, uint64_t cta_d_in_cluster = 0) {
        uint64_t cluster_id, cluster_major_offset = 0;
        // cluster_id = blk_per_grid_dim / 2
        // cluster_major_offset = blk_per_grid_dim % 2
        divmod_cluster_shape_major(cluster_id, cluster_major_offset, 
            blk_per_grid_dim);

        uint64_t cluster_idx_minor, cluster_idx_major;
        // cluster_idx_minor = cluster_id / (BLOCKS_Q / 2)
        // cluster_idx_major = cluster_id % (BLOCKS_Q / 2)
        divmod_cluster_blk_major(cluster_idx_minor, cluster_idx_major, 
            cluster_id);

        auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor);
        auto major_work_idx = static_cast<int32_t>(cluster_idx_major
            * divmod_cluster_shape_major.divisor + cluster_major_offset);
     
        return {major_work_idx, minor_work_idx};
    }

    /**
     * @brief Advance to the next work tile
     * 
     * @param advance_count number of work tiles to advance
     * @param tile_count_smem shared memoyr for tile counting
     */
    CUTLASS_DEVICE
    void advance_to_next_work(uint32_t advance_count,
        SharedStorage& tile_count_smem) {
        int blk_rank = block_rank_in_cluster();
        dim3 cls_shape = cluster_shape();
        int cls_size = cls_shape.x * cls_shape.y * cls_shape.z;

        int linear_idx;

        // CIDX = 0, TIDX = 0 thread gets next linear index
        if(block_rank_in_cluster() == 0 && threadIdx.x == 0) {
            tile_count_smem = atomicSub(scheduler_params.tile_count_semaphore, 
                1);
            cutlass::arch::fence_view_async_shared();
        }

        cute::cluster_arrive_relaxed();
        cute::cluster_wait();

        cooperative_groups::cluster_group cluster
            = cooperative_groups::this_cluster();

        int *dst_smem = cluster.map_shared_rank(
            &tile_count_smem, 0);

        // elected threads in a warp reads index
        if(threadIdx.x % 32 == 0) {
            linear_idx = *dst_smem;
        }

        // broadcast to other threads in a warp
        linear_idx = __shfl_sync(0xffffffff, linear_idx, 0);

        // all threads in thread block cluster gets the tile number
        current_work_linear_idx_ = (int64_t)(cls_size * linear_idx + blk_rank);
    }

    /**
     * @brief Is this tile last one?
     * 
     * @param work_tile_info current work tile
     * @param advance_count advancing count
     * @return true this tile is last one
     * @return false this tile is not last on
     */
    CUTLASS_DEVICE
    bool is_last_tile(WorkTileInfo& work_tile_info, uint32_t advance_count = 1) 
        const {
        if (continue_current_work(work_tile_info)) {
            return false;
        }
        return not get_current_work_for_linear_idx(
            current_work_linear_idx_).is_valid();
    }

    /**
     * @brief Computes the linear index within a batch given M and N tile 
     * offsets within the batch. This essentially inverts the mapping performed 
     * in get_work_idx_q_and_d
     * 
     * @param tile_q Q tile index
     * @param tile_d D tile index = 0
     * @param divmod_cluster_shape_major /% 2
     * @param divmod_cluster_blk_major /% (BLOCKS_Q / 2)
     * @return uint64_t linear index
     */
    static CUTLASS_DEVICE
    uint64_t get_linear_idx_from_q_and_d(int32_t tile_q, int32_t tile_d,
        FastDivmodU64Pow2 const& divmod_cluster_shape_major,
        FastDivmodU64 const& divmod_cluster_blk_major) {
        uint64_t major_work_idx;
        // Always Along Q
        major_work_idx = static_cast<uint64_t>(tile_q);

        return major_work_idx;
    }

    /**
     * @brief Given the inputs, computes the total number of output blocks over 
     * which this problem will compute. Note that this is only the logical size 
     * of our grid, not the physical grid we will actually launch.
     * 
     * @tparam ProblemShape (q, kv, d, (hg, hr, b)) type
     * @tparam BlockShape (q, kv, d) type
     * @tparam ClusterShape (q, kv, 1) type
     * @param problem_shape (q, kv, d, (hg, hr, b))
     * @param cta_shape (q, kv, d)
     * @param cluster_shape (q, kv, 1)
     * @return dim3 tiled CTA shape (q / (TILED_Q / CLUSTER_Q), 1, 1)
     */
    template<class ProblemShapeQKVDHGHRB, class BlockShape, class ClusterShape>
    CUTLASS_HOST_DEVICE static
    dim3 get_tiled_cta_shape_qdhghrb(
        ProblemShapeQKVDHGHRB problem_shape_qkvdhghrb,
        BlockShape cta_shape, ClusterShape cluster_shape) {
        // number of CTAs in Q dimension
        auto cta_q = cute::size(cute::ceil_div(
            cute::shape<0>(problem_shape_qkvdhghrb),
            cute::shape<0>(cta_shape)));
        // number of CTAs in D dimension
        auto cta_d = cute::size(cute::ceil_div(
            cute::shape<2>(problem_shape_qkvdhghrb), 
            cute::shape<2>(cta_shape)));

        return Params::get_tiled_cta_shape_qdhghrb(
            to_fmha_coord(problem_shape_qkvdhghrb),
            to_fmha_coord(cluster_shape), cta_q, cta_d);
    }

    /**
     * @brief Reloaded interface that receives WorkTileInfo to deduce next 
     * work. Kernel helper function to get next work tile
     * 
     * @param work_tile_info work tile to process
     * @param 
     * @return Tuple<WorkTileInfo, bool> (work_tile, true)
     */
    CUTLASS_DEVICE
    auto fetch_next_work(WorkTileInfo work_tile_info,
        SharedStorage& tile_count_smem) {
        if (continue_current_work(work_tile_info)) {
            return cute::make_tuple(work_tile_info, true);
        }

        advance_to_next_work(1, tile_count_smem);
        return cute::make_tuple(get_current_work(), true);
    }

    // Given the inputs, computes the total number of output blocks over which this problem will compute.
    // Note that this is only the logical size of our grid, not the physical grid we will actually launch.

    /**
     * @brief Given the inputs, computes the total number of output blocks over 
     * which this problem will compute.
     * 
     * @tparam ProblemShape (q, kv, d, ((hg, hr), b)) type
     * @tparam TileShape (q, kv, d) type
     * @tparam AtomThrShape (2, 1, 1) type
     * @tparam ClusterShape (2, 1, 1) type
     * @param problem_shape (q, kv, d, ((hg, hr), b)) type
     * @param tile_shape (q, kv, d) type
     * @param atom_thr_shape (2, 1, 1) type
     * @param cluster_shape (2, 1, 1) type
     * @return dim3 CTA shape (q, kv, d)
     */

    template<class ProblemShapeQKVDHGHRB, class TileShape, class AtomThrShape,
        class ClusterShape>
    CUTLASS_HOST_DEVICE static
    dim3 get_tiled_cta_shape_qdhghrb(ProblemShapeQKVDHGHRB problem_shape,
        TileShape tile_shape, AtomThrShape atom_thr_shape,
        ClusterShape cluster_shape) {
        // # of tiles of Q, 1, # of tiles of HG, HR, B
        auto [tiles_q, tiles_d, tiles_b] = product_each(ceil_div(
            select<0,2,3>(problem_shape), select<0, 2>(tile_shape)));
        // # tiles of Q
        auto cta_q = round_nearest(tiles_q * size<0>(atom_thr_shape),
            size<0>(cluster_shape));
        // # tiles of D = 1
        auto cta_d = round_nearest(tiles_d * size<2>(atom_thr_shape),
            size<2>(cluster_shape));

        return Params::get_tiled_cta_shape_qdhghrb(to_fmha_coord(problem_shape),
            to_fmha_coord(cluster_shape), cta_q, cta_d);
    }

    /**
     * @brief Kernel helper function to get next work tile
     * 
     * @tparam TileSchedulerPipeline Tile scheduler pipeline type
     * @tparam TileSchedulerPipelineState Tile scheduler pipeline state type
     * @param work_tile_info work tile information
     * @param scheduler_pipeline tile scheduler pipeline
     * @param scheduler_pipe_consumer_state tile scheduler consumer state
     * @return Tuple<WorkTileInfo, bool> (work_tile, true)
     */
    template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
    CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo work_tile_info,
        TileSchedulerPipeline& scheduler_pipeline,
        TileSchedulerPipelineState scheduler_pipe_consumer_state,
        SharedStorage& tile_count_smem) {
        return fetch_next_work(work_tile_info, tile_count_smem);
    }

    /**
     * @brief Work tile to CTA coordinate
     * 
     * @param work_tile_info worktile
     * @return Coord<int, int, int, Coord<Coord<int, int>, int>>
     */
    CUTLASS_DEVICE
    static auto work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
        // Get every cta coord in three dimensions of the cluster
        auto [cta_q_in_cluster, cta_d_in_cluster, cta_b_in_cluster]
            = cute::block_id_in_cluster();

        return make_coord(work_tile_info.Q_idx
            + static_cast<int32_t>(cta_q_in_cluster), _, work_tile_info.D_idx
            + static_cast<int32_t>(cta_d_in_cluster), make_coord(
            make_coord(work_tile_info.HG_idx, work_tile_info.HR_idx), 
            work_tile_info.B_idx));
    }

    /**
     * @brief Work tile to CTA coordinate
     * 
     * @param work_tile_info work tile
     * @param block_id_in_cluster block index in thread block cluster
     * @return Coord<int, int, int, Coord<Coord<int, int>, int>>
     */
    CUTLASS_DEVICE
    static auto work_tile_to_cta_coord(WorkTileInfo work_tile_info,
        dim3 block_id_in_cluster) {
        // Get every cta coord in three dimensions of the cluster
        auto [cta_q_in_cluster, cta_d_in_cluster, cta_b_in_cluster]
            = block_id_in_cluster;

        return make_coord(work_tile_info.Q_idx
            + static_cast<int32_t>(cta_q_in_cluster), _, work_tile_info.D_idx
            + static_cast<int32_t>(cta_d_in_cluster), make_coord(
            make_coord(work_tile_info.HG_idx, work_tile_info.HR_idx), 
            work_tile_info.B_idx));
    }

    /**
     * @brief Given the inputs, computes the physical grid we should launch.
     * 
     * @tparam ProblemShape (q, kv, d, ((hg, hr), b)) type
     * @tparam BlockShape (q, kv, d) type
     * @tparam ClusterShape (q, kv, 1) type
     * @param params Kernel API parameters
     * @param problem_shape (q, kv, d, ((hg, hr), b))
     * @param cta_shape (q, kv, d)
     * @param cluster_shape (q, kv=1, d=1)
     * @param hw_info kernel hardware information
     * @param arguments Device API arguments
     * @param truncate_by_problem_size reduce dimension to problem size = true
     * @return dim3 Grid shape
     */
    template<class ProblemShapeQKVDHGHRB, class BlockShape, class ClusterShape>
    CUTLASS_HOST_DEVICE static
    dim3 get_grid_shape([[maybe_unused]] Params const& params,
        ProblemShapeQKVDHGHRB problem_shape, BlockShape cta_shape,
        ClusterShape cluster_shape, KernelHardwareInfo hw_info,
        Arguments arguments = Arguments{}, bool truncate_by_problem_size=true) {
        dim3 problem_blocks = get_tiled_cta_shape_qdhghrb(problem_shape, 
                cta_shape, cluster_shape);

        return Params::get_grid_shape(problem_blocks,
            to_fmha_coord(cluster_shape), hw_info,
            /* truncate_by_problem_size = */true);
    }

    /**
     * @brief Given the inputs, computes the physical grid we should launch.
     * 
     * @tparam ProblemShape (q, kv, d, ((hg, hr), b)) type
     * @tparam TileShape (q, kv, d) type
     * @tparam AtomThrShape Atomic thread (q, kv, d) type
     * @tparam ClusterShape (q, kv, 1) type
     * @param params Kernel API parameters
     * @param problem_shape (q, kv, d, ((hg, hr), b))
     * @param tile_shape (q, kv, d)
     * @param atom_thr_shape (2, 1, 1)
     * @param cluster_shape (q, kv, 1)
     * @param hw_info kernel hardware information
     * @return dim3 Grid shape
     */
    template<class ProblemShapeQKVDHGHRB, class TileShape, class AtomThrShape,
        class ClusterShape>
    static dim3 get_grid_shape(Params const& params,
        ProblemShapeQKVDHGHRB problem_shape, TileShape tile_shape,
        AtomThrShape atom_thr_shape, ClusterShape cluster_shape,
        KernelHardwareInfo hw_info) {

        dim3 problem_blocks = get_tiled_cta_shape_qdhghkb(problem_shape, 
            tile_shape, atom_thr_shape, cluster_shape);
        Arguments args{};

        return Params::get_grid_shape(problem_blocks,
            to_fmha_coord(cluster_shape), hw_info,
            /* truncate_by_problem_size = */true);
    }

    /**
     * @brief Convert CTA-level work tile info to cluster-level tile coord
     * 
     * @param work_tile_info Work tile
     * @return Coord<int, int, int, Coord<Coord<int, int>, int>
     */
    CUTLASS_DEVICE
    auto work_tile_to_cluster_coord_qkvdhghrb(
        WorkTileInfo work_tile_info) const {
        // TileScheduler works at CTA-level, kernel works at cluster-level
        int q_coord = idx2crd(work_tile_info.Q_idx
            / scheduler_params.cluster_shape_q_,
            scheduler_params.problem_tiles_q_);
        int d_coord = idx2crd(work_tile_info.D_idx,
            scheduler_params.problem_tiles_d_);
        int hg_coord = idx2crd(work_tile_info.HG_idx,
                scheduler_params.problem_tiles_hg_);
        int hr_coord = idx2crd(work_tile_info.HR_idx,
            scheduler_params.problem_tiles_hr_);
        int b_coord = idx2crd(work_tile_info.B_idx,
            scheduler_params.problem_tiles_b_);
        return make_coord(q_coord, _, d_coord, make_coord(make_coord(hg_coord, 
            hr_coord), b_coord));
    }

    /**
     * @brief Returns whether the block assigned this work should compute the 
     * epilogue for the corresponding output tile. For the basic tile 
     * scheduler, this is always true.
     * 
     * @return true always return true
     */
    CUTLASS_HOST_DEVICE
    static bool compute_epilogue(WorkTileInfo const&, Params const&) {
        return true;
    }

    /**
     * @brief Returns whether the block assigned this work should compute the 
     * epilogue for the corresponding output tile. For the basic tile 
     * scheduler, this is always true.
     * 
     * @return true always return true
     */
    CUTLASS_HOST_DEVICE
    static bool compute_epilogue(WorkTileInfo const&) {
        return true;
    }

    // Performs the reduction across splits for a given output tile. Since this scheduler does
    // not split output tiles, no reduction is needed.

    /**
     * @brief Performs the reduction across splits for a given output tile. 
     * Since this scheduler does not split output tiles, no reduction is needed.
     * 
     * @tparam FrgTensorO Fragment Tensor Output
     */
    template <class FrgTensorO>
    CUTLASS_DEVICE
    static void fixup(Params const&, WorkTileInfo const&, FrgTensorO&, 
        uint32_t, uint32_t) {}

    /**
     * @brief Performs the reduction across splits for a given output tile. No 
     * fixup is required for work units returned by this scheduler.
     * 
     * @tparam FrgTensorO Fragment Tensor Output
     * @return CUTLASS_DEVICE 
     */
    template <class FrgTensorO>
    CUTLASS_DEVICE
    void fixup(WorkTileInfo const&, FrgTensorO&, uint32_t, uint32_t) const { }

    /**
     * @brief Returns whether the current WorkTileInfo passed in should 
     * continue to be used. Since this scheduler only schedules work in units 
     * of single, full output tiles, the WorkTileInfo passed in should not be 
     * used after having been processed.
     * 
     * @return false no continuing use
     */
    CUTLASS_DEVICE
    static bool continue_current_work(WorkTileInfo&) {
        return false;
    }

    /**
     * @brief Get the kv tile iterator
     * 
     * @tparam ProblemShape (q, kv, d, ((hg, hr), b)) type
     * @tparam TileShape (q, kv, d) type
     * @tparam Shape unused
     * @param work_tile_info work tile information 
     * @param problem_shape_ (q, kv, d, ((hg, hr), b)))
     * @param tile_shape (q, kv, d)
     * @return ForwardCoordIterator<Coord, Shape, Order> coord iterator
     */
    template <class ProblemShape, class TileShape, class Shape>
    CUTLASS_DEVICE
    auto get_kv_tile_iterator(WorkTileInfo const& work_tile_info,
        ProblemShape problem_shape_, TileShape tile_shape, Shape) {
        auto kv_tiles = cute::ceil_div(cute::get<1>(problem_shape_),
        cute::get<1>(tile_shape));
        return cute::make_coord_iterator(kv_tiles);
    }

    /**
     * @brief Get the work kv tile count
     * 
     * @tparam ProblemShape (q, kv, d, ((hg, hr), b)) type
     * @tparam TileShape (q, kv, d) type
     * @param work_tile_info work tile information 
     * @param problem_shape (q, kv, d, ((hg, hr), b)))
     * @param tile_shape (q, kv, d)
     * @return int number of kv tiles
     */
    template <class ProblemShape, class TileShape>
    CUTLASS_HOST_DEVICE
    static int get_work_kv_tile_count(WorkTileInfo const& work_tile_info, 
        ProblemShape problem_shape, TileShape tile_shape) {
        // All work units returned by this scheduler cover the entire K 
        // iteration space of the output tile assigned to the work unit.
        return cute::size(cute::ceil_div(cute::get<1>(problem_shape), 
            cute::get<1>(tile_shape)));
    }

    /**
     * @brief Get the work k tile start index
     * 
     * @return int32_t 0
     */
    CUTLASS_HOST_DEVICE
    static uint32_t get_work_kv_tile_start(WorkTileInfo const&) {
        // All work units returned by this scheduler start from K tile 0
        return 0u;
    }

    /**
     * @brief does this tile scheduler needs separate reduction?
     * 
     * @param params Kernel API parameters
     * @return false no need separate reduction
     */
    CUTLASS_DEVICE
    static bool need_separate_reduction(Params const& params) {
        return false;
    }

    /**
     * @brief does this work tile needs separate reduction?
     * 
     * @param work_tile_info work tile
     * @param params Kernel API parameters
     * @return false no need separate reduction
     */
    CUTLASS_DEVICE
    bool is_work_tile_for_reduction(WorkTileInfo const& work_tile_info,
        Params const& params) {
        return false;
    }

    /**
     * @brief Does nothing
     * 
     * @tparam FrgTensorO Fragment Tensor Output
     * @param params Kernel API parameters
     * @param work_tile_info work tile
     * @param accumulators final accumulation result
     * @param num_barriers number of barriers
     * @param barrier_idx barrier index
     */
    template <class FrgTensorO>
    CUTLASS_DEVICE void separate_reduction(Params const& params,
        WorkTileInfo const& work_tile_info, FrgTensorO& accumulators,
        uint32_t num_barriers, uint32_t barrier_idx) {}

    /**
     * @brief Shares the accumulator set with peers in the global workspace
     * 
     * @tparam FrgTensorO Fragment tensor Output
     * @param params Kernel API paramters
     * @param work_tile_info work tile
     * @param accumulators final accumulation result
     * @param num_barriers number of barriers
     * @param barrier_idx barrier index
     */
    template <class FrgTensorO>
    CUTLASS_DEVICE static void share(Params const& params,
        WorkTileInfo const& work_tile_info, FrgTensorO& accumulators,
        uint32_t num_barriers, uint32_t barrier_idx) {}

    /**
     * @brief is valid warpgroup in work tile
     * 
     * @param work_tile_info work tile
     * @return true always return true
     */
    CUTLASS_DEVICE
    static bool valid_warpgroup_in_work_tile(
        WorkTileInfo const& work_tile_info) {
        return true;
    }

    /**
     * @brief Is this tile scheduler require separate reduction?
     * 
     * @param params Kernel API parameters
     * @return false no need for separate reduction
     */
    CUTLASS_DEVICE
    static bool requires_separate_reduction(Params const& params) {
        return false;
    }

public:
    /**
     * @brief The basic tile scheduler does not require any additional workspace
     * 
     * @tparam ProblemShape (Q, KV, D, ((H_G, H_R), B))
     * @tparam ElementAccumulatorS element type of tensor S
     * @tparam ElementAccumulatorO element type of tensor O
     * @return size_t scratch workspace size
     */
    template <class ProblemShape, class ElementAccumulatorS,
        class ElementAccumulatorO>
    static size_t get_workspace_size(Arguments const&, ProblemShape,
        KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
        return sizeof(int);
    }
    /**
     * @brief Initialize worksapce
     * 
     * @tparam ProblemShape (Q, KV, D, ((H_G, H_R), B))
     * @tparam ElementAccumulatorS element type of tensor S
     * @tparam ElementAccumulatorO element type of tensor O
     * @param workspace workspace pointer
     * @param cuda_adapter CUDA Runtime & Driver API adapter
     * @return cutlass::Status always return kSuccess
     */
    template <class ProblemShape, class ElementAccumulatorS,
        class ElementAccumulatorO>
    static cutlass::Status initialize_workspace(Arguments const&,
        void* workspace, cudaStream_t, ProblemShape, KernelHardwareInfo const&, 
        uint32_t, const uint32_t = 1, uint32_t = 1,
        CudaHostAdapter* cuda_adapter = nullptr) {
        return Status::kSuccess;
    }
};


////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel
