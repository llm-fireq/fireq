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

namespace cutlass::fmha::kernel {

/**
 * @brief Forward Declarations
 * 
 * @tparam kTag tag type
 * @tparam Default Default option type
 * @tparam Options Options type
 */
template<auto kTag, typename Default, typename... Options>
struct find_option;

/**
 * @brief Find options
 * 
 * @tparam kTag tag type
 * @tparam Default Default option type
 */
template<auto kTag, typename Default>
struct find_option<kTag, Default> {
    using option_value = Default;
};

/**
 * @brief Recursive Find Options
 * 
 * @tparam kTag tag type
 * @tparam Default Default option type
 * @tparam Option Option to select
 * @tparam Options other options
 */
template<auto kTag, typename Default, typename Option, typename... Options>
struct find_option<kTag, Default, Option, Options...> :
  std::conditional_t<
    Option::tag == kTag,
    Option,
    find_option<kTag, Default, Options...>
  >
{};

/**
 * @brief The option that want to find
 * 
 * @tparam kTag tag type
 * @tparam Default Default option type
 * @tparam Options other options
 */
template<auto kTag, typename Default, typename... Options>
using find_option_t = typename find_option<kTag, Default, Options...>::option_value;

/**
 * @brief Tags
 */
enum class Tag {
    // persistent scheduler
    kIsPersistent,
    // number of MMA warpgroups
    kNumMmaWarpGroups,
    // Load Q tensor separately
    kLoadsQSeparately,
    // Locked mainloop execution
    kIsMainloopLocked,
    // Locked epilogue execution
    kIsEpilogueLocked,
    // number of Q stages
    kStagesQ,
    // number of KV stages
    kStagesKV,
    // Epilogue kind
    kEpilogueKind,
    // number of blocks per SM
    kBlocksPerSM,
    // size of thread block cluster in M direction
    kClusterM,
    // Accumulator of QK type
    kAccQK
};

/**
 * @brief Options
 * 
 * @tparam kTag tag type
 * @tparam Value option value type
 */
template<auto kTag, class Value>
struct Option {
  static constexpr auto tag = kTag;
  using option_value = Value;
};

}  // namespace cutlass::fmha::kernel
