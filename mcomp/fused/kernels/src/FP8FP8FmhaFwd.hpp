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
/*! \file
    \brief Example implementation of fused multi-head attention for the NVIDIA Blackwell SM100
    architecture using CUTLASS 3.

    MQA/GQA
    -------

    The head dimension can be represented as a tuple, where the K/V strides in the
    first dimension is zero. This has the effect of MQA or GQA.
    * MHA is (head_size:head_stride).
    * MQA is (head_size:head_stride) in Q and (head_size:_0) in K and V.
    * GQA is (grouped_heads,heads_kv):(head_stride,grouped_heads*head_stride) in Q
      and (grouped_heads,heads_kv):(0,head_stride) in K and V

    Output Scale
    ------------

    The output scale gets passed to the collective mainloop, and is applied
    using FP32 compute pre-quantization

    Variable Sequence Length
    ------------------------

    For variable sequence length, pass in VariableLength objects
    (max_seqlen, cumulative_seqlen_ptr) in the problem shape for
    seqlen Q and KV.

    Support
    ---------

    Right now e4m3 with fp32 compute is using a 256x256 tiling and a head dimension
    of 128 is supported.


    Example usage:
      $ ./examples/77_blackell_fmha/77_blackell_fmha_fp8 \
            --b=2048 --h=2048 --d=2048 --q=2048 --k=2048
*/

#define DSHOW(x) print(#x ": "); print(x); print("\n");
#define DSHOWT(x) print(#x ": "); print_tensor(x); print("\n");

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/device_memory.h"


#include "device/fmha.hpp"
#include "collective/fmha_mask.hpp"
#include "collective/sm90_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm90_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm90_fmha_fwd_kernel_tma_warpspecialized.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;


///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

struct FwdRunner {
    using Element = cutlass::float_e4m3_t;

    using ElementOut = cutlass::bfloat16_t;
    using ElementAccumulatorPV = float;

    using ClusterShape = Shape<_2, _1, _1>;

    // Q K D ((H_G H_R) B)
    using ProblemShape = cute::tuple<int, int, int, 
        cute::tuple<cute::tuple<int, int>, int>>;
    
    using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, 
        int>>;  // Q D (H_G H_R B)
    using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, 
        int>>;  // K D (H_G H_R B)
    using StrideV = StrideK;
    using StrideO = StrideQ;

    using TileScheduler = cutlass::fmha::kernel
        ::FmhaFwdSplitQPersistentTileScheduler;

    template<class TileShape>
    using Mainloop = cutlass::fmha::collective
        ::Sm90FmhaFwdMainloopTmaWarpspecialized<TileShape, ClusterShape, 
        Element, Element, Element>;
    template<class TileShape>
    using Epilogue = cutlass::fmha::epilogue
        ::Sm90FmhaFwdEpilogueTmaWarpspecialized<TileShape, ElementOut>;
    template<class TileShape>
    using Operation = cutlass::fmha::device::FMHA<cutlass::fmha::kernel
        ::Sm90FmhaFwdKernelTmaWarpspecializedPingpong<ProblemShape,
        Mainloop<TileShape>, Epilogue<TileShape>, TileScheduler>>;

    //
    // Data members
    //

    /// Initialization
    StrideQ stride_Q;
    StrideK stride_K;
    StrideV stride_V;
    StrideO stride_O;

    // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    int q;
    int kv;
    int d;
    int h_g;
    int h_r;
    int b;
    bool initialized;

    using Op64 = Operation<Shape<_128, _256, _64>>;
    using Op128 = Operation<Shape<_128, _128, _128>>;
    using Op256 = Operation<Shape<_128, _64, _256>>;

    cutlass::DeviceAllocation<uint8_t> workspace;

    FwdRunner(int head_group, int num_heads, int head_size) : d(head_size),
        h_g(head_group), h_r(num_heads) {
        assert(d == 64 || d == 128 || d == 256);
        // Change device_id to another value if you are running on a machine with multiple GPUs and wish
        // to use a GPU other than that with device ID 0.
        hw_info.device_id = 0;
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

        initialized = false;

        b = -1;
        q = -1;
        kv = -1;

        workspace.reset(16);
    }

    void forward(void *query, void *key, void *value, void *output,
        int batch_size, int seq_len, cudaStream_t stream) {
        b = batch_size;
        q = seq_len;
        kv = seq_len;

        stride_Q = make_stride(h_g * h_r * d, _1{}, make_stride(
            make_stride(d, h_g * d), h_g * h_r * d * q));
        stride_O = stride_Q;
        stride_K = make_stride(h_r * d, _1{}, make_stride(make_stride(_0{}, 
            d), h_r * d * kv));
        stride_V = stride_K;

        ProblemShape problem_size(make_tuple(q, kv, d, make_tuple(
            make_tuple(h_g, h_r), b)));

        if(d == 64) {
            typename Op64::Arguments arguments{
                problem_size, {(Element *)query, stride_Q, (Element *)key, 
                stride_K, (Element *)value, stride_V}, {(ElementOut *)output, 
                stride_O}, hw_info,
                {}
            };

            cutlass::Status status = cutlass::Status::kSuccess;

            Op64 op64;
                
            status = op64.can_implement(arguments);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "This kernel is not supported. Last CUDA error "
                    << "is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        
            status = op64.initialize(arguments, workspace.get(), stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to initialize the CUTLASS kernel. Last "
                    << "CUDA error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
    
            // Run
            status = op64.run(stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA "
                    << "error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        }
        else if(d == 128) {
            typename Op128::Arguments arguments{
                problem_size, {(Element *)query, stride_Q, (Element *)key, 
                stride_K, (Element *)value, stride_V}, {(ElementOut *)output, 
                stride_O}, hw_info,
                {}
            };

            cutlass::Status status = cutlass::Status::kSuccess;

            Op128 op128;

            status = op128.can_implement(arguments);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "This kernel is not supported. Last CUDA error "
                    << "is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        
            status = op128.initialize(arguments, workspace.get(), stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to initialize the CUTLASS kernel. Last "
                    << "CUDA error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
    
            // Run
            status = op128.run(stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA "
                    << "error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        }
        if(d == 256) {
            typename Op256::Arguments arguments{
                problem_size, {(Element *)query, stride_Q, (Element *)key, 
                stride_K, (Element *)value, stride_V}, {(ElementOut *)output, 
                stride_O}, hw_info,
                {}
            };

            cutlass::Status status = cutlass::Status::kSuccess;

            Op256 op256;
                
            status = op256.can_implement(arguments);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "This kernel is not supported. Last CUDA error "
                    << "is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        
            status = op256.initialize(arguments, workspace.get(), stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to initialize the CUTLASS kernel. Last "
                    << "CUDA error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
    
            // Run
            status = op256.run(stream);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA "
                    << "error is: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
                return;
            }
        }

        return;
    }

};


///////////////////////////////////////////////////////////////////////////////////////////////////





/////////////////////////////////////////////////////////////////////////////////////////////////
