/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/numeric_conversion.h"

#include "cute/util/type_traits.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include <tensor_impl.hpp>


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/**
 * @brief The universal converter
 * 
 * @tparam SrcType source tensor data type
 * @tparam DstType destination tensor data type
 * @tparam LayoutIn source tensor layout type
 * @tparam LayoutOut destination tensor layout type
 */
template <class SrcType, class DstType, class LayoutIn, class LayoutOut>
struct FMHALayoutAwareConvertImpl {
    /**
     * @brief Universal convert function
     * 
     * @tparam EngineIn input tensor engine type
     * @tparam EngineOut output tensor engine type
     * @param src source tensor
     * @param dst destination tensor
     */
    template<class EngineIn, class EngineOut>
    CUTLASS_DEVICE static void convert(cute::Tensor<EngineIn, LayoutIn> const& 
        src, cute::Tensor<EngineOut, LayoutOut>& dst) {
        static_assert(cute::is_same_v<SrcType, typename EngineIn::value_type>
            && cute::is_same_v<DstType, typename EngineOut::value_type>);
        static_assert(cute::cosize_v<LayoutIn> == cute::cosize_v<LayoutOut>);
        // number of elements to convert
        constexpr int N = decltype(cute::max_common_vector(LayoutIn{},
            LayoutOut{})){};

        // source array type
        using SrcArray = cutlass::Array<SrcType, N>;
        // destination array type
        using DstArray = cutlass::Array<DstType, N>;
        // numeric array converter type
        using Converter = cutlass::NumericArrayConverter<DstType, SrcType, N,
            cutlass::FloatRoundStyle::round_to_nearest>;

        // source vector array
        auto&& src_vm = cute::recast<SrcArray>(src);
        // destination vector array
        auto&& dst_vm = cute::recast<DstArray>(dst);

        // convert array
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < src_vm.size(); ++i)
            dst_vm(i) = Converter::convert(src_vm(i));
    }
};

/**
 * @brief Layout aware convert function
 * 
 * @tparam EngineIn input tensor engine type
 * @tparam EngineOut output tensor engine type
 * @tparam LayoutIn input tensor layout type
 * @tparam LayoutOut output tensor layout type
 * @param src input tensor
 * @param dst output tensor
 */
template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut>
CUTLASS_DEVICE void FMHALayoutAwareConvert( // Accept mutable temporaries
    cute::Tensor<EngineIn, LayoutIn> const& src, cute::Tensor<EngineOut, 
    LayoutOut>&& dst) {
    FMHALayoutAwareConvert(src, dst);
}

/**
 * @brief Layout aware convert function
 * 
 * @tparam EngineIn input tensor engine type
 * @tparam EngineOut output tensor engine type
 * @tparam LayoutIn input tensor layout type
 * @tparam LayoutOut output tensor layout type
 * @param src input tensor
 * @param dst output tensor
 */
template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut>
CUTLASS_DEVICE void FMHALayoutAwareConvert(cute::Tensor<EngineIn,
    LayoutIn> const& src, cute::Tensor<EngineOut, LayoutOut>& dst) {
    // source value type
    using SrcType = typename EngineIn::value_type;
    // destination value type
    using DstType = typename EngineOut::value_type;

    // source vector memory
    Tensor src_vm = coalesce(src);
    // destination vector memory
    Tensor dst_vm = coalesce(dst);
    // source layout
    Layout src_layout = src_vm.layout();
    // destination layout
    Layout dst_layout = dst_vm.layout();

    // convert
    FMHALayoutAwareConvertImpl<SrcType, DstType, decltype(src_layout),
        decltype(dst_layout)>::convert(src_vm, dst_vm);
}

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::collective::detail {
/**
 * @brief Get the logical pointer
 * 
 * @tparam PointerType data type
 * @param ptr pointer to convert to logical pointer
 * @return subbyte_iterator or PointerType *
 */
template <class PointerType>
static constexpr
CUTLASS_HOST_DEVICE auto get_logical_ptr(PointerType const* ptr) {
    if constexpr (cute::sizeof_bits_v<PointerType> < 8)
        return subbyte_iterator<PointerType const>(ptr);
    else
        return ptr;
}

/**
 * @brief Get the Shared Memory Layout
 * 
 * @tparam Stages number of shared memory stages
 * @tparam LayoutAtom Shared memory layout type
 * @tparam TileShape Tile shape type
 * @tparam Stride Stride type
 * @param layout_atom shared memory layout
 * @param tile_shape tile shape
 * @param stride stride
 * @return constexpr Layout Shared memory layout
 */
template<int Stages, class LayoutAtom, class TileShape, class Stride>
static constexpr
CUTLASS_HOST_DEVICE auto get_smem_layout(LayoutAtom layout_atom,
    TileShape const& tile_shape, Stride const& stride) {
    if constexpr (not cute::is_layout<Stride>::value) {
        return tile_to_shape(layout_atom, append(tile_shape, Int<Stages>{}),
            cute::conditional_t< ::cutlass::gemm::detail::is_major<0,Stride>(), 
            Step<_2,_1,_3>, Step<_1,_2,_3>>{});
    }
    else {
        auto gmem_tile = composition(stride, tile_shape);
        return make_layout_like(append(gmem_tile, make_layout(Int<Stages>{}, 
            0)));
    }
}

/**
 * @brief Get the Global Memory Layout
 * 
 * @tparam Shape global memory shape type
 * @tparam Stride stride type
 * @param shape global memory shape
 * @param stride stride
 * @return Layout or Stride global memory layout
 */
template<class Shape, class Stride>
static constexpr
CUTLASS_HOST_DEVICE auto get_gmem_layout(Shape const& shape,
    Stride const& stride) {
    if constexpr (not cute::is_layout<Stride>::value) {
        return make_layout(shape, stride);
    }
    else {
        return stride;
    }
}

/**
 * @brief Mixed Input Convert Utility
 * 
 * @tparam Collective Collective Mainloop
 */
template<class Collective>
struct MixedInputUtils {
private:
    // shared memory layout Q type
    using SmemLayoutQ = typename Collective::SmemLayoutQ;
    // shared memory layout K type
    using SmemLayoutK = typename Collective::SmemLayoutK;
    // shared memory layout V type
    using SmemLayoutV = typename Collective::SmemLayoutV;
    // shared memory layout scale K type
    using SmemLayoutScaleK = typename Collective::SmemLayoutScaleK;
    // shared memory layout scale V type
    using SmemLayoutScaleV = typename Collective::SmemLayoutScaleV;
    // shared memory copy atom scale K type
    using SmemCopyAtomScaleK = typename Collective::SmemCopyAtomScaleK;
    // shared memory copy atom scale K type
    using SmemCopyAtomScaleV = typename Collective::SmemCopyAtomScaleV;
    // element type
    using ElementScale = typename Collective::ElementScale;

public:
    /**
     * @brief size of scale K shared memory layout
     * 
     * @return constexpr int size of shared memory layout scale
     */
    static constexpr auto
    elements_per_smem_scale_k() {
        return cute::cosize_v<SmemLayoutScaleK>;
    }
    /**
     * @brief size of scale V shared memory layout
     * 
     * @return constexpr int size of shared memory layout scale
     */
    static constexpr auto
    elements_per_smem_scale_v() {
        return cute::cosize_v<SmemLayoutScaleV>;
    }
    /**
     * @brief return 0
     * 
     * @return constexpr int 0
     */
    static constexpr auto
    elements_per_smem_zero() {
        return 0;
    }

    // These methods use some the public members of the class. For that reason, 
    // we define them after the public section.

    /**
     * @brief Compute the size of TMA transaction bytes of Q X D
     * 
     * @return constexpr uint32_t size of TMA transaction bytes
     */
    static constexpr uint32_t
    compute_tma_transaction_bytes_qd() {
        return cutlass::bits_to_bytes(size<0>(SmemLayoutQ{}) * size<1>(
            SmemLayoutQ{}) * static_cast<uint32_t>(
            cute::sizeof_bits_v<cutlass::float_e4m3_t>));
    }
    /**
     * @brief Compute the size of TMA transaction bytes of KV X D
     * 
     * @return constexpr uint32_t size of TMA transaction bytes
     */
    static constexpr uint32_t
    compute_tma_transaction_bytes_kvd() {
        return cutlass::bits_to_bytes(size<0>(SmemLayoutK{}) * size<1>(
            SmemLayoutK{}) * static_cast<uint32_t>(
            cute::sizeof_bits_v<cutlass::int4b_t>));
    }
    /**
     * @brief Compute the size of TMA transaction bytes of D X KV
     * 
     * @return constexpr uint32_t size of TMA transaction bytes
     */
    static constexpr uint32_t
    compute_tma_transaction_bytes_dkv() {
        return cutlass::bits_to_bytes(size<0>(SmemLayoutV{}) * size<1>(
            SmemLayoutV{}) * static_cast<uint32_t>(
            cute::sizeof_bits_v<cutlass::int4b_t>));
    }
    /**
     * @brief Compute the size of TMA transaction bytes of scale K
     * 
     * @return constexpr uint32_t size of TMA transaction bytes
     */
    static constexpr uint32_t
    compute_tma_transaction_bytes_extra_k() {
        constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(size<0>(
            SmemLayoutScaleK{}) * size<1>(SmemLayoutScaleK{})
            * static_cast<uint32_t>(
            cute::sizeof_bits_v<cutlass::Array<float_e4m3_t, 8>>));
        static_assert(scale_tx_bytes % 128 == 0,
            "Each scale stage must be 128B aligned."); // required by TMA
        return scale_tx_bytes;
    }
    /**
     * @brief Compute the size of TMA transaction bytes of scale V
     * 
     * @return constexpr uint32_t size of TMA transaction bytes
     */
    static constexpr uint32_t
    compute_tma_transaction_bytes_extra_v() {
        constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(size<0>(
            SmemLayoutScaleV{}) * size<1>(SmemLayoutScaleV{})
            * static_cast<uint32_t>(
            cute::sizeof_bits_v<cutlass::Array<float_e4m3_t, 8>>));
        static_assert(scale_tx_bytes % 128 == 0,
            "Each scale stage must be 128B aligned."); // required by TMA
        return scale_tx_bytes;
    }

    /// Utilities to copy K and extra inputs from smem to RF
    template <class SmemTiledCopyK, class TensorKSmemView,
        class TensorKCopyView, class... Ts, class... Us>
    CUTLASS_DEVICE static void copy_tensors_KVD(SmemTiledCopyK const& 
        smem_tiled_copy_K, TensorKSmemView const& tSsK, TensorKCopyView& 
        tSrK_copy_view, cute::tuple<Ts...> const& partitioned_mma_extra_info,
        cute::tuple<Us...> const& tiled_copy_and_views, int d_block,
        int read_stage) {
        // Copy B
        copy(smem_tiled_copy_K, tSsK(_,_,d_block,read_stage),
            tSrK_copy_view(_,_,d_block));

        if (d_block == 0) {
            // We are starting a new k-tile so copy the scale
            // Scale Tiled Copy Shared Memory
            auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
            // Register Copy View of Scale
            auto tSrS_copy_view = cute::get<1>(tiled_copy_and_views);
            // Shared memory Copy View of Scale
            auto tSsS = cute::get<0>(partitioned_mma_extra_info);
            // Copy Scale
            copy(smem_tiled_copy_S, tSsS(_,_,d_block,read_stage),
                tSrS_copy_view(_,_,d_block));
        }
    }

    /**
     * @brief The core converter uses a lookup table to converts i4 -> 8 bit 
     * value.
     * 
     * @tparam EngineIn Input Tensor Engine Type
     * @tparam LayoutIn Input Tensor Layout Type
     * @tparam EngineOut Output Tensor Engine Type
     * @tparam LayoutOut Output Tensor Layout Type
     * @tparam EngineScale Scale Tensor Engine Type
     * @tparam LayoutScale Scale Tensor Layout Type
     * @param src source tensor
     * @param dst destination tensor
     * @param scales_neg negative scale tensor(-8 to -1)
     * @param scales_pos positive scale tensor(0 to 7)
     */
    template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut,
        class EngineScale, class LayoutScale>
    CUTLASS_DEVICE
    static void lookup_table_convert( // Accept mutable temporaries
        Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& 
        dst, Tensor<EngineScale, LayoutScale> const& scales_neg,
        Tensor<EngineScale, LayoutScale> const& scales_pos) {
        lookup_table_convert(src, dst, scales_neg, scales_pos);
    }
    
    /**
     * @brief The core converter uses a lookup table to converts i4 -> 8 bit 
     * value.
     * 
     * @tparam EngineIn Input Tensor Engine Type
     * @tparam LayoutIn Input Tensor Layout Type
     * @tparam EngineOut Output Tensor Engine Type
     * @tparam LayoutOut Output Tensor Layout Type
     * @tparam EngineScale Scale Tensor Engine Type
     * @tparam LayoutScale Scale Tensor Layout Type
     * @param src source tensor
     * @param dst destination tensor
     * @param scales_neg negative scale tensor(-8 to -1)
     * @param scales_pos positive scale tensor(0 to 7)
     */
    template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut,
        class EngineScale, class LayoutScale>
    CUTLASS_DEVICE static void lookup_table_convert(Tensor<EngineIn, LayoutIn> 
        const& src, Tensor<EngineOut, LayoutOut>& dst, Tensor<EngineScale, 
        LayoutScale> const& scales_neg, Tensor<EngineScale, LayoutScale> const& 
        scales_pos) {
        // number of elements to convert
        constexpr int N = cute::cosize(LayoutIn{});
        static_assert(N == 4 || N == 8);
        static_assert(cosize(LayoutScale{}) <= N / 4, "at least 4 consecutive "
            "weights must share the same scale.");
        // source value array: INT4
        using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
        // destination value array: FP8 E4M3
        using DstArray = cutlass::Array<cutlass::float_e4m3_t, 8>;
        // register array of 4 consecutive FP8s = 1 INT32 Reg
        using RegArray = cutlass::AlignedArray<uint32_t, N / 4,
            sizeof(DstArray)>;

        // View the input as reg
        // source register
        auto&& src_reg = cute::recast<uint32_t>(src)(0);
        // destination register
        auto&& r = cute::recast<RegArray>(dst)(0);

        // Determines if to get from the signed or unsigned candidates
        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        uint32_t sign; // ((reg & 0x88888888) | 0x64206420) >> 1 
        asm volatile(
            "{\n"
            "  lop3.b32 %0, %1, %2, %3, %4;\n" \
            "}\n"
            : "=r"(sign)
            : "r"(src_reg), "n"(0x88888888), "n"(0x64206420), "n"(immLut)
        );
        sign = sign >> 1;

        // Ignore sign bit when indexing into LUT
        uint32_t lut_idx = src_reg & 0x77777777;
        // negative scale tensor
        Tensor scales_neg_ = cute::filter(scales_neg);
        // positive scale tensor
        Tensor scales_pos_ = cute::filter(scales_pos);
        // convert
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 4; ++i, lut_idx >>=16, sign >>=16) {
            // negative scale register
            auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 2> 
                const&>(scales_neg_(i));
            // positive scale register
            auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 2> 
                const&>(scales_pos_(i));
            // permute convert
            asm volatile(
                "{\n"
                "  .reg .b32 pos, neg                    ;\n" \
                "  prmt .b32 neg, %3, %4, %1             ;\n" \
                "  prmt .b32 pos, %5, %6, %1             ;\n" \
                "  prmt .b32 %0, pos, neg, %2            ;\n" \
                "}\n"
                : "=r"(r[i])
                : "r"(lut_idx), "r"(sign), "r"(scale_neg_[0]), "r"(scale_neg_[1]), "r"(scale_pos_[0]), "r"(scale_pos_[1])
            );
        }
    }

    /**
     * @brief Utilities to dequantize B.
     * 
     * @tparam Layout Layout type of B
     * @param tensor layout of B
     */
    template <class Layout>
    CUTLASS_DEVICE
    static void static_check_scale(Layout const& tensor) {
        static_assert(shape<0>(Layout{}) >= 4 && stride<0>(Layout{}) == 0, "At "
            "least 4 adjacent weights in a thread must share the same scale.");
    }

    /**
     * @brief Utilities to dequantize B.
     * 
     * @tparam Engine Engine type of scale tensor
     * @tparam Layout Layout type of scaletensor
     * @param tensor scale tensor
     */
    template <class Engine,class Layout>
    CUTLASS_DEVICE
    static void static_check_scale(Tensor<Engine, Layout> const& tensor) {
        static_check_scale(flatten(Layout{}));
    }

    /**
     * @brief Dequantize the K on D blocks
     * 
     * @tparam EngineIn input tensor engine type
     * @tparam EngineOut output tensor engine type
     * @tparam LayoutIn input tensor layout type
     * @tparam LayoutOut output tensor layout type
     * @tparam Ts scale tensor
     * @param tSrK_load B register tensor loaded
     * @param tSrK_mma B register tensor for MMA
     * @param partitioned_extra_info partitioned extra information (i.e. scale)
     * @param d_block d block index
     */
    template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut,
        class... Ts>
    CUTLASS_DEVICE static void dequantize_K_dblock(Tensor<EngineIn, LayoutIn> 
        const& tSrK_load, Tensor<EngineOut, LayoutOut>& tSrK_mma,
        cute::tuple<Ts...>& partitioned_extra_info, int const d_block) {
        static_assert(is_rmem<EngineIn>::value, "Input tensor for K conversion "
            "must come from registers");
        static_assert(is_rmem<EngineOut>::value, "Output tensor for K "
            "conversion must come from registers");
        static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
        static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
        static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);

        // source value type: INT4
        using SrcType = typename EngineIn::value_type;
        // destination value type: FP8 E4M3
        using DstType = typename EngineOut::value_type;
        // source tensor
        Tensor src = tSrK_load(_, _, d_block);
        // destination tensor
        Tensor dst = tSrK_mma(_, _, d_block);
        
        CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
            "The first mode of tensor src must be contiguous in memory");
        // try to make the size of the first mode equal to 32bit
        int constexpr NumValPerSrcReg = cute::min(
            decltype(size(src(_, 0)))::value,
            ceil_div(32, sizeof_bits_v<SrcType>));
        // source vector memory
        Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, 
            Int<NumValPerSrcReg>{}));
        // destination vector memory
        Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, 
            Int<NumValPerSrcReg>{}));
        // number of elements for convert
        constexpr int num_elements = decltype(size(src))::value;
        static_assert(sizeof_bits_v<cutlass::Array<float_e4m3_t, 8>> == 64, 
            "Lookup table only supports 8 8bit scale values now.");
        static_assert(num_elements % 4 == 0 && num_elements >= 4, "Lookup "
            "table requires a vector size of 4x when converting.");
        // negative register scale tensor
        Tensor tSrSK_neg = cute::get<1>(partitioned_extra_info);
        // positive register scale tensor modification to its value is needed
        auto&& tSrSK_pos = cute::get<2>(partitioned_extra_info);
        // negative scale tensor
        Tensor scales_neg = tSrSK_neg(_, _, d_block);
        // positive scale tensor
        Tensor scales_pos = tSrSK_pos(_, _, d_block);
        CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));
        // check scale layout
        static_check_scale(scales_neg);
        static_check_scale(scales_pos);

        // negative scale vector memory
        Tensor scales_neg_vm = cute::group_modes<1,-1>(cute::zipped_divide(
            scales_neg, Int<NumValPerSrcReg>{}));
        // positive scale vector memory
        Tensor scales_pos_vm = cute::group_modes<1,-1>(cute::zipped_divide(
            scales_pos, Int<NumValPerSrcReg>{}));

        if (d_block == 0) {
            // negative scale vector memory
            Tensor scales_neg_vm_ = filter(scales_neg_vm);
            // positive scale vector memory
            Tensor scales_pos_vm_ = filter(scales_pos_vm);

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(scales_neg_vm_.layout()); ++i) {
                // negative scale array
                auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 
                    2> const&>(scales_neg_vm_(i));
                // positive scale array
                auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 
                    2>&>(scales_pos_vm_(i));
                // immediate lookup table
                constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
                // convert
                asm volatile(
                    "{\n"
                    "  lop3 .b32 %0, %2, %4, %5, %6;\n" \
                    "  xor  .b32 %1, %3, %5;        \n" \
                    "}\n"
                    : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])
                    : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00), 
                    "n"(0x80808080), "n"(immLut)
                    );
            }
        }
        // convert from source to destination
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
            lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, 
                i), scales_pos_vm(_, i));
        }
    }

    /**
     * @brief Dequantize the V
     * 
     * @tparam EngineIn input tensor engine type
     * @tparam EngineOut output tensor engine type
     * @tparam LayoutIn input tensor layout type
     * @tparam LayoutOut output tensor layout type
     * @tparam Ts scale tensor
     * @param tOrV_load B register tensor loaded
     * @param tOrV_mma B register tensor for MMA
     * @param partitioned_extra_info partitioned extra information (i.e. scale)
     */
    template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut,
        class... Ts>
    CUTLASS_DEVICE static void dequantize_V(Tensor<EngineIn, LayoutIn> 
        const& tOrV_load, Tensor<EngineOut, LayoutOut>& tOrV_mma,
        cute::tuple<Ts...>& partitioned_extra_info) {
        static_assert(is_rmem<EngineIn>::value, "Input tensor for V conversion "
            "must come from registers");
        static_assert(is_rmem<EngineOut>::value, "Output tensor for V "
            "conversion must come from registers");
        static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
        static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
        static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);

        // source value type: INT4
        using SrcType = typename EngineIn::value_type;
        // destination value type: FP8 E4M3
        using DstType = typename EngineOut::value_type;
        // source tensor
        Tensor src = tOrV_load(_, _);
        // destination tensor
        Tensor dst = tOrV_mma(_, _);
        
        CUTE_STATIC_ASSERT_V(size(src(_, 0)) == cosize(src(_, 0).layout()),
            "The first mode of tensor src must be contiguous in memory");
        // try to make the size of the first mode equal to 32bit
        int constexpr NumValPerSrcReg = cute::min(
            decltype(size(src(_, 0)))::value,
            ceil_div(32, sizeof_bits_v<SrcType>));
        // source vector memory
        Tensor src_vm = cute::group_modes<1,-1>(cute::zipped_divide(src, 
            Int<NumValPerSrcReg>{}));
        // destination vector memory
        Tensor dst_vm = cute::group_modes<1,-1>(cute::zipped_divide(dst, 
            Int<NumValPerSrcReg>{}));
        // number of elements for convert
        constexpr int num_elements = decltype(size(src))::value;
        static_assert(sizeof_bits_v<cutlass::Array<float_e4m3_t, 8>> == 64, 
            "Lookup table only supports 8 8bit scale values now.");
        static_assert(num_elements % 4 == 0 && num_elements >= 4, "Lookup "
            "table requires a vector size of 4x when converting.");
        // negative register scale tensor
        Tensor tOrSV_neg = cute::get<1>(partitioned_extra_info);
        // positive register scale tensor modification to its value is needed
        auto&& tOrSV_pos = cute::get<2>(partitioned_extra_info);
        // negative scale tensor
        Tensor scales_neg = tOrSV_neg(_, _);
        // positive scale tensor
        Tensor scales_pos = tOrSV_pos(_, _);
        CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));
        // check scale layout
        static_check_scale(scales_neg);
        static_check_scale(scales_pos);

        // negative scale vector memory
        Tensor scales_neg_vm = cute::group_modes<1,-1>(cute::zipped_divide(
            scales_neg, Int<NumValPerSrcReg>{}));
        // positive scale vector memory
        Tensor scales_pos_vm = cute::group_modes<1,-1>(cute::zipped_divide(
            scales_pos, Int<NumValPerSrcReg>{}));

        // negative scale vector memory
        Tensor scales_neg_vm_ = filter(scales_neg_vm);
        // positive scale vector memory
        Tensor scales_pos_vm_ = filter(scales_pos_vm);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(scales_neg_vm_.layout()); ++i) {
            // negative scale array
            auto&& scale_neg_ = reinterpret_cast<cutlass::Array<uint32_t, 
                2> const&>(scales_neg_vm_(i));
            // positive scale array
            auto&& scale_pos_ = reinterpret_cast<cutlass::Array<uint32_t, 
                2>&>(scales_pos_vm_(i));
            // immediate lookup table
            constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
            // convert
            asm volatile(
                "{\n"
                "  lop3 .b32 %0, %2, %4, %5, %6;\n" \
                "  xor  .b32 %1, %3, %5;        \n" \
                "}\n"
                : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])
                : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00), 
                "n"(0x80808080), "n"(immLut)
                );
        }
        // convert from source to destination
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
            lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, 
                i), scales_pos_vm(_, i));
        }
    }

    /**
     * @brief Utilities for any additional inputs inside of the TMA load
     * 
     * @tparam Params mainloop parameter type
     * @tparam TensorStorage shared memory storage type
     * @tparam Ts scale tensor type
     * @param mainloop_params mainloop parameters
     * @param load_inputs load input tensors
     * @param shared_tensors shared memory tensors
     * @param cluster_local_block_id local thread block id in cluster
     * @param l_coord batch coordinate
     * @return tuple<Tensor, Tensor> global & shared tensor tuple
     */
    template <class Params, class TensorStorage, class... Ts>
    CUTLASS_DEVICE static auto partition_extra_tma_inputs(Params const& 
        mainloop_params, cute::tuple<Ts...> const& load_inputs, TensorStorage& 
        shared_tensors, uint2 const& cluster_local_block_id,
        int const l_coord) {
        // (BLK_KV,BLK_D,PIPE)
        Tensor sSK  = make_tensor(make_smem_ptr(shared_tensors.smem_scale_k
            .begin()), SmemLayoutScaleK{});
        // (BLK_D,BLK_KV,PIPE)
        Tensor sSV  = make_tensor(make_smem_ptr(shared_tensors.smem_scale_v
            .begin()), SmemLayoutScaleV{});
        // global scale K KVDL tensor
        Tensor gSK_kvdl = get<3>(load_inputs);
        // global scale K DKVL tensor
        Tensor gSV_dkvl = get<4>(load_inputs);
        auto block_tma_sk = mainloop_params.tma_load_scale_k.get_slice(
            cluster_local_block_id.x);
        auto block_tma_sv = mainloop_params.tma_load_scale_v.get_slice(
            cluster_local_block_id.x);
        // (BLK_KV,BLK_D,kv)
        Tensor gSK = gSK_kvdl(_,_,_,_0{},l_coord);
        // (BLK_D,BLK_KV,kv)
        Tensor gSV = gSV_dkvl(_,_,_0{},_,l_coord);
        // (TMA,TMA_KV,TMA_D,kv)
        Tensor tSgSK = block_tma_sk.partition_S(gSK);
        // (TMA,TMA_D,TMA_KV,kv)
        Tensor tOgSV = block_tma_sv.partition_S(gSV);
        // (TMA,TMA_KV,TMA_D,PIPE)
        Tensor tSsSK = block_tma_sk.partition_D(sSK);
        // (TMA,TMA_D,TMA_KV,PIPE)
        Tensor tOsSV = block_tma_sv.partition_D(sSV);
        return cute::make_tuple(tSgSK, tSsSK, tOgSV, tOsSV); 
    }

    /**
     * @brief Utilities for partitioning extra inputs for loading from smem in 
     * the mainloop.
     * 
     * @tparam ThreadMma Thread MMA Slice Type
     * @tparam TensorStorage Shared Tensor Storage Type
     * @param mma_thread_slice Thread MMA Slice
     * @param shared_tensors shared memory tensors
     * @return tuple<Shared Scale, Reg Neg Scale, Reg Pos Scale>
     */
    template <class ThreadMma, class TensorStorage>
    CUTLASS_DEVICE static auto partition_extra_mma_info_k(ThreadMma const& 
        mma_thread_slice, TensorStorage& shared_tensors) {
        // (BLK_N,BLK_SCALE_K,PIPE)
        Tensor sSK = make_tensor(make_smem_ptr(shared_tensors.smem_scale_k
            .begin()), SmemLayoutScaleK{});
        // (THR_N,THR_SCALE_K,PIPE)
        Tensor tSsSK = mma_thread_slice.partition_B(sSK);
        // register negative tensor
        Tensor tSrSK_neg = make_tensor<ElementScale>(
            mma_thread_slice.partition_fragment_B(sSK(_,_,Int<0>{})).layout());
        // register positive tensor
        Tensor tCrSK_pos = make_tensor<ElementScale>(
            mma_thread_slice.partition_fragment_B(sSK(_,_,Int<0>{})).layout()); 
    
        return cute::make_tuple(tSsSK, tSrSK_neg, tSrSK_pos);
    }

    /**
     * @brief Utilities for partitioning extra inputs for loading from smem in 
     * the mainloop.
     * 
     * @tparam ThreadCopy Thread Copy Slice Type
     * @tparam TensorStorage Shared Tensor Storage Type
     * @param thr_thread_slice Thread Copy Slice
     * @param shared_tensors shared memory tensors
     * @return tuple<Shared Scale, Reg Neg Scale, Reg Pos Scale>
     */
    template <class ThreadCopy, class TensorStorage>
    CUTLASS_DEVICE static auto partition_extra_mma_info_v(ThreadCopy const& 
        copy_thread_slice, TensorStorage& shared_tensors) {
        // (BLK_N,BLK_SCALE_K,PIPE)
        Tensor sSV = make_tensor(make_smem_ptr(shared_tensors.smem_scale_v
            .begin()), SmemLayoutScaleV{});
        // (THR_N,THR_SCALE_K,PIPE)
        Tensor tSsSV = copy_thread_slice.partition_S(sSV);
        // register negative tensor
        Tensor tSrSV_neg = make_tensor<ElementScale>(
            copy_thread_slice.partition_S(sSV(_,_,Int<0>{})).layout());
        // register positive tensor
        Tensor tCrSV_pos = make_tensor<ElementScale>(
            copy_thread_slice.partition_S(sSV(_,_,Int<0>{})).layout()); 
    
        return cute::make_tuple(tSsSV, tSrSV_neg, tSrSV_pos);
    }

    /**
     * @brief Returns the tiled copy and copy views for the extra inputs.
     * 
     * @tparam TiledMma Tiled MMA type
     * @tparam Ts scale tensor type
     * @param tiled_mma tiled mma
     * @param partitioned_extra_info scale tensors 
     * @param warp_group_thread_idx thread index in a warpgroup
     * @return tuple<TiledCopy, Tensor>
     */
    template <class TiledMma, class... Ts>
    CUTLASS_DEVICE static auto retile_extra_mma_info_k(TiledMma const& 
        tiled_mma, cute::tuple<Ts...>& partitioned_extra_info, int const 
        warp_group_thread_idx) {
        // Tiled Copy of Scale
        auto smem_tiled_copy_S = make_tiled_copy_B(SmemCopyAtomScaleK{}, 
            tiled_mma);
        // Thread Copy of Scale
        auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(
            warp_group_thread_idx);
        // (CPY,CPY_N,CPY_K)
        Tensor tSrSK_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(
            partitioned_extra_info));
            
        return cute::make_tuple(smem_tiled_copy_S, tSrSK_copy_view);
    }

    /**
     * @brief Returns the tiled copy and copy views for the extra inputs.
     * 
     * @tparam TiledMma Tiled MMA type
     * @tparam Ts scale tensor type
     * @param tiled_mma tiled mma
     * @param partitioned_extra_info scale tensors 
     * @param warp_group_thread_idx thread index in a warpgroup
     * @return tuple<TiledCopy, Tensor>
     */
    template <class TiledMma, class... Ts>
    CUTLASS_DEVICE static auto retile_extra_mma_info_v(TiledMma const& 
        tiled_mma, cute::tuple<Ts...>& partitioned_extra_info, int const 
        warp_group_thread_idx) {
        // Tiled Copy of Scale
        auto smem_tiled_copy_S = make_tiled_copy_B(SmemCopyAtomScaleV{}, 
            tiled_mma);
        // Thread Copy of Scale
        auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(
            warp_group_thread_idx);
        // (CPY,CPY_N,CPY_K)
        Tensor tSrSV_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(
            partitioned_extra_info));
            
        return cute::make_tuple(smem_tiled_copy_S, tSrSV_copy_view);
    }
};

} // cutlass::fmha::collective::detail
