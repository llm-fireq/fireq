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
/*!
  \file
  \brief An universal device layer for cutlass 3.x-style kernels.
*/

#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief FMHA Device API
 * 
 * @tparam Kernel_ FMHA Kernel Type
 */
template <class Kernel_>
class FMHA {
public:
    using Kernel = Kernel_;
    // maximum number of threads per thread block = 384
    static int const kThreadCount = Kernel::MaxThreadsPerBlock;

    /// Argument structure: User API
    using Arguments = typename Kernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename Kernel::Params;
private:
    /// Kernel API parameters object
    Params params_;
    /**
     * @brief is this kernel initialized?
     * 
     * @param set set the initialization state
     * @return true initialized
     * @return false not initialized
     */
    bool is_initialized(bool set = false) {
        static bool initialized = false;
        if (set) initialized = true;
        return initialized;
    }

public:
    /**
     * Access the Params structure
     */
    Params const& params() const {
        return params_;
    }

    /**
     * @brief Determines whether the GEMM can execute the given problem.
     * 
     * @param args 
     * @return Status 
     */
    static Status
    can_implement(Arguments const& args) {
        if (Kernel::can_implement(args))
            return Status::kSuccess;
        else
            return Status::kInvalid;
    }

    /**
     * @brief Get the workspace size
     * 
     * @param args User API Level Arguments
     * @return size_t required workspace size
     */
    static size_t
    get_workspace_size(Arguments const& args) {
        size_t workspace_bytes = 0;
        workspace_bytes += Kernel::get_workspace_size(args);

        CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

        return workspace_bytes;
    }

    /**
     * @brief Computes the grid shape
     * 
     * @param params Kernel Level Parameters
     * @return dim3 Grid Shape
     */
    static dim3
    get_grid_shape(Params const& params) {
        return Kernel::get_grid_shape(params);
    }

    /**
     * @brief Computes the maximum number of active blocks per multiprocessor
     * 
     * @param smem_capacity Required Shared Memory Capacity
     * @return int allocation result
     */
    static int maximum_active_blocks(int smem_capacity = -1) {
        CUTLASS_TRACE_HOST("FMHA::maximum_active_blocks()");
        int max_active_blocks = -1;
        int smem_size = Kernel::SharedStorageSize;

        // first, account for dynamic smem capacity if needed
        cudaError_t result;
        if (smem_size >= (48 << 10)) {
            CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
            result = cudaFuncSetAttribute(device_kernel<Kernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            if (cudaSuccess != result) {
                result = cudaGetLastError(); // to clear the error bit
                CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: "
                    << cudaGetErrorString(result));
                return -1;
            }
        }

        // query occupancy after setting smem size
        result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, device_kernel<Kernel>, 
            Kernel::MaxThreadsPerBlock, smem_size);

        if (cudaSuccess != result) {
            result = cudaGetLastError(); // to clear the error bit
            CUTLASS_TRACE_HOST("  cudaOccupancyMaxActiveBlocksPerMultiprocessor"
                "() returned error: " << cudaGetErrorString(result));
            return -1;
        }

        CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
    }

    /**
     * @brief Initializes GEMM state from arguments.
     * 
     * @param args User API arguments
     * @param workspace workspace pointer
     * @param stream CUDA stream
     * @return Status initialization State
     */
    Status initialize(Arguments const& args, void* workspace = nullptr,
        cudaStream_t stream = nullptr) {
        CUTLASS_TRACE_HOST("FMHA::initialize() - workspace "
            << workspace << ", stream: " << (stream ? "non-null" : "null"));

        // Initialize the workspace
        Status status = Kernel::initialize_workspace(args, workspace, stream);
        if (status != Status::kSuccess) {
            return status;
        }

        // Initialize the Params structure
        params_ = Kernel::to_underlying_arguments(args, workspace);

        if (is_initialized()) return Status::kSuccess;

        // account for dynamic smem capacity if needed
        int smem_size = Kernel::SharedStorageSize;
        if (smem_size >= (48 << 10)) {
            CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
            cudaError_t result = cudaFuncSetAttribute(device_kernel<Kernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            if (cudaSuccess != result) {
                result = cudaGetLastError(); // to clear the error bit
                CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: "
                    << cudaGetErrorString(result));
                return Status::kErrorInternal;
            }
        }

        is_initialized(true);

        return Status::kSuccess;
    }

    /**
     * @brief Update API is preserved in 3.0, but does not guarantee a 
     * lightweight update of params.
     * 
     * @param args User API arguments
     * @param workspace workspace pointer
     * @return Status update state
     */
    Status
    update(Arguments const& args, void* workspace = nullptr) {
        CUTLASS_TRACE_HOST("FMHA()::update() - workspace: " << workspace);

        size_t workspace_bytes = get_workspace_size(args);
        if (workspace_bytes > 0 && nullptr == workspace) {
            return Status::kErrorWorkspaceNull;
        }

        params_ = Kernel::to_underlying_arguments(args, workspace);
        return Status::kSuccess;
    }

    /**
     * @brief Primary run() entry point API that is static allowing users to 
     * create and manage their own params. Supplied params struct must be 
     * construct by calling Kernel::to_underling_arguments()
     * 
     * @param params Kernel API parameters
     * @param stream CUDA stream
     * @return Status run status
     */
    static Status
    run(Params& params, cudaStream_t stream = nullptr) {
        CUTLASS_TRACE_HOST("FMHA::run()");
        dim3 const block = Kernel::get_block_shape();
        dim3 const grid = get_grid_shape(params);

        // configure smem size and carveout
        int smem_size = Kernel::SharedStorageSize;

        Status launch_result;

        auto ret = cudaMemcpyAsync(params.scheduler.tile_count_semaphore,
            &params.scheduler.num_clusters, 4, cudaMemcpyHostToDevice, 
            stream);
        assert(ret == cudaSuccess);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("FMHA::run: Use extended launch API");
#endif
        // Use extended launch API only for mainloops that use it
        dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
            cute::size<1>(typename Kernel::ClusterShape{}),
            cute::size<2>(typename Kernel::ClusterShape{}));
        void const* kernel = (void const*) device_kernel<Kernel>;
        void* kernel_params[] = {&params};
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("FMHA::run: Launching dynamic cluster kernel");
#endif
        launch_result = ClusterLauncher::launch(grid, cluster, block, 
            smem_size, stream, kernel, kernel_params);

        cudaError_t result = cudaGetLastError();
        if (cudaSuccess == result && Status::kSuccess == launch_result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            CUTLASS_TRACE_HOST("FMHA::run: cudaGetLastError reports success");
#endif
            return Status::kSuccess;
        }
        else {
            CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
            return Status::kErrorInternal;
        }
    }

    //
    // Non-static launch overloads that first create and set the internal 
    // params struct of this kernel handle.
    //

    /**
     * @brief Launches the kernel after first constructing Params internal 
     * state from supplied arguments.
     * 
     * @param args User API arguments
     * @param workspace workspace pointer
     * @param stream CUDA stream
     * @return Status run status
     */
    Status
    run(Arguments const& args, void* workspace = nullptr,
        cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace, stream);
        if (Status::kSuccess == status) {
        status = run(params_, stream);
        }
        return status;
    }

    /// Launches the kernel after first constructing Params internal state from supplied arguments.

    /**
     * @brief Launches the kernel after first constructing Params internal 
     * state from supplied arguments.
     * 
     * @param args User API arguments
     * @param workspace workspace pointer
     * @param stream CUDA stream
     * @return Status run status
     */
    Status
    operator()(Arguments const& args, void* workspace = nullptr,
        cudaStream_t stream = nullptr) {
        return run(args, workspace, stream);
    }

    /**
     * @brief Overload that allows a user to re-launch the same kernel without 
     * updating internal params struct.
     * 
     * @param stream CUDA stream
     * @return Status run status
     */
    Status
    run(cudaStream_t stream = nullptr) {
        return run(params_, stream);
    }

    /**
     * @brief Overload that allows a user to re-launch the same kernel without 
     * updating internal params struct.
     * 
     * @param stream CUDA stream
     * @return Status run status
     */
    Status
    operator()(cudaStream_t stream = nullptr) {
        return run(params_, stream);
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::device

////////////////////////////////////////////////////////////////////////////////
