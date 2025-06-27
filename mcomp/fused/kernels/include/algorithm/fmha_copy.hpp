#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include <math.h>

namespace cute {

/**
 * @brief Convert Accumulator S layout to identity
 * Convert acc_layout from ((2, 2, V), MMA_M, MMA_N)
 * to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
 * 
 * @tparam LayoutS Accumulator S layout type
 * @param acc_layout Accumulator S layout
 * @return Layout<Shape,Stride> Identity layout
 */
template<typename LayoutS>
CUTLASS_DEVICE auto convert_layout_C_to_I(LayoutS acc_layout)
{
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank<0>(acc_layout))::value == 3);
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    
    auto n_2 = get<0, 0>(acc_layout);
    auto m_2 = get<0, 1>(acc_layout);
    auto v = get<0, 2>(acc_layout);
    auto mma_m = get<1>(acc_layout);
    auto mma_n = get<2>(acc_layout);

    auto nrow = make_layout(m_2, mma_m);
    auto ncol = make_layout(n_2, v, mma_n);

    return make_layout(nrow, ncol);
}

/**
 * @brief Convert Accumulator P to Input P
 * Convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N)
 * to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
 * 
 * @tparam LayoutS Accumulator P layout type
 * @param acc_layout Accumulator P layout
 * @return Layout<Shape,Stride> GEMM Input layout
 */
template<typename LayoutP>
CUTLASS_DEVICE auto convert_layout_C_to_A(LayoutP acc_layout) {
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank<0>(acc_layout))::value == 3);
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
    static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);

    // (((2, 2), N / 32))
    auto l = logical_divide(get<0, 2>(acc_layout),
        Tile<Layout<Shape<_2, _2>>>{});
    // (4, 2, 2)
    auto input_layout = make_layout(Layout<_4>{}, get<0, 0, 0>(l),
        get<0, 0, 1>(l));
    // MMA_M
    auto mma_m = get<1>(acc_layout);
    // (N / 32, MMA_N)
    auto n_layout = coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)));

    return make_layout(input_layout, mma_m, n_layout);
}

/**
 * @brief Convert Element type from FP16 to FP8
 * 
 * @tparam EngineInput input tensor engine type
 * @tparam Layout in & out tensor layout type
 * @tparam EngineOutput output tensor engine type
 * @param input input tensor
 * @param output output tensor
 */
template <typename EngineInput, typename Layout, typename EngineOutput>
CUTLASS_DEVICE void convert(Tensor<EngineInput, Layout> const &input, 
    Tensor<EngineOutput, Layout> &output) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using input_type = typename EngineInput::value_type;
    using output_type = typename EngineOutput::value_type;

    static_assert(CUTE_STATIC_V(size(input)) % 2 == 0,
        "Fragment size does not vectorize properly");

    Tensor in_frag = cute::recast<cutlass::Array<input_type, 2> const>(input);
    Tensor out_frg = cute::recast<cutlass::Array<output_type, 2>>(output);

    cutlass::NumericArrayConverter<output_type, input_type, 2> convert_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(in_frag); ++i)
        out_frg[i] = convert_op(in_frag[i]);
}

template <typename Fragment>
CUTLASS_DEVICE void permute_A(Fragment &frag) {
    // frag has shape A ((4, 2, 2), MMA_M, (N / 32, MMA_N))
    static_assert(decltype(rank(frag))::value == 3);
    static_assert(decltype(rank<0>(frag))::value == 3);
    static_assert(decltype(size<0, 0>(frag))::value == 4);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(size<0, 2>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 1);
    // frag has shape A ((1, 2, 2), MMA_M, (N / 32, MMA_N))
    Tensor frag_32b = group_modes<1, 3>(recast<uint>(frag));
    static_assert(decltype(rank(frag_32b))::value == 2);
    static_assert(decltype(rank<0>(frag_32b))::value == 3);
    static_assert(decltype(size<0, 0>(frag_32b))::value == 1);
    static_assert(decltype(size<0, 1>(frag_32b))::value == 2);
    static_assert(decltype(size<0, 2>(frag_32b))::value == 2);

    Tensor frag_64b = group_modes<1, 3>(recast<uint64_t>(frag));
    static_assert(decltype(rank(frag_64b))::value == 2);
    static_assert(decltype(rank<0>(frag_64b))::value == 3);
    static_assert(decltype(size<0, 0>(frag_64b))::value == 1);
    static_assert(decltype(size<0, 1>(frag_64b))::value == 1);
    static_assert(decltype(size<0, 2>(frag_64b))::value == 2);

    int lane_id = threadIdx.x % 4;
    int is_01 = (lane_id >> 1) == 0;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < size<1>(frag_64b); i++) {
        CUTLASS_PRAGMA_UNROLL
        for(int j = 0; j < 2; j++) {
            uint64_t src0 = __shfl_sync(0xffffffff, frag_64b(make_coord(_0{}, 
                _0{}, j), i), 2 * lane_id, 4);
            uint64_t src1 = __shfl_sync(0xffffffff, frag_64b(make_coord(_0{}, 
                _0{}, j), i), 2 * lane_id + 1, 4);

            uint16_t *arr = reinterpret_cast<uint16_t *>(
                &frag_32b(make_coord(_0{}, _0{}, j), i));

            uint16_t *src0_arr = reinterpret_cast<uint16_t *>(&src0);
            uint16_t *src1_arr = reinterpret_cast<uint16_t *>(&src1);

            src0_arr = is_01 ? src0_arr : src0_arr + 2;
            src1_arr = is_01 ? src1_arr : src1_arr + 2;

            arr[0] = src0_arr[0];
            arr[1] = src1_arr[0];
            arr[2] = src0_arr[1];
            arr[3] = src1_arr[1];
        }
    }
}

template<class TiledMma_, class SmemLayoutB_, class SmemLayoutAtomB_,
    class ElementB_>
class AsyncTranspositionOperandB {
public:
using TiledMma = TiledMma_;
    using SmemLayoutB = SmemLayoutB_;
    using SmemLayoutAtomB = SmemLayoutAtomB_;
    using ElementB = ElementB_;

    using LDSM_ThrShape  = Shape<_32, _4, _1, _1>;
    using LDSM_ThrStride = Stride<_4, _1, _0, _0>;
    using LDSM_ValShape = Shape<_2, _2, _1, _4>;
    using LDSM_ValStride = Stride<_1, _2, _16, _4>;
    using LDSM_DivShape = Shape<_64, _8>;
    using TiledCopyS2RV = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, ElementB>{}, Layout<LDSM_ThrShape, 
        LDSM_ThrStride>{}, Layout<LDSM_ValShape, LDSM_ValStride>{}));
    using STSM_ThrShape  = Shape<_8, _4, _4, _1>;
    using STSM_ThrStride = Stride<_4, _1, _32, _0>;
    using STSM_ValShape = Shape<_1, _4, _2, _2>;
    using STSM_ValStride = Stride<_0, _1, _4, _8>;
    using STSM_DivShape = Shape<_8, _16>;
    using TiledCopyR2SVT = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, ElementB>{}, Layout<STSM_ThrShape, 
        STSM_ThrStride>{}, Layout<STSM_ValShape, STSM_ValStride>{}));

    constexpr CUTLASS_HOST_DEVICE
    AsyncTranspositionOperandB(int warp_idx_, int warp_group_thread_idx_,    
        TiledMma, SmemLayoutB, SmemLayoutAtomB, ElementB)
        : warp_idx(warp_idx_), warp_group_thread_idx(warp_group_thread_idx_) {}

    template <class TensorSmemB, class TensorTransposedSmemB>
    CUTLASS_DEVICE void operator()(TensorSmemB const& sB,
        TensorTransposedSmemB const& gmma_sB, int read_stage)
    {
        // Set up for transposing V, only used if Transpose_V
        TiledCopyS2RV tiled_s2r_v;
        TiledCopyR2SVT tiled_r2s_vt;

        auto thr_s2r_v = tiled_s2r_v.get_thread_slice(warp_group_thread_idx);
        auto thr_r2s_vt = tiled_r2s_vt.get_thread_slice(warp_group_thread_idx);

        // flat_divide(sVt, LDSM_DivShape{})
        // (64, 8, kHeadDim / 64, kBlockN / 8)
        // ((16, 1), 1, 1, kHeadDim / 64, kBlockN / 32)
        Tensor tTsV = thr_s2r_v.partition_S(flat_divide(sB, LDSM_DivShape{}));  
        // flat_divide(sV, STSM_DivShape{})
        // (8, 16, kHeadDim / 8, (4, kBlockN / 64))
        // ((16, 1), 1, 1, kHeadDim / 64, (2, kBlockN / 64))
        Tensor tTsVT = thr_r2s_vt.partition_D(flat_divide(gmma_sB, STSM_DivShape{}));

        CUTE_STATIC_ASSERT_V(rank(tTsV) == rank(tTsVT));
        CUTE_STATIC_ASSERT_V(size<0>(tTsV) == size<0>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<1>(tTsV) == size<1>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<2>(tTsV) == size<2>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<3>(tTsV) == size<3>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<4>(tTsV) == size<4>(tTsVT));
        // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
        static constexpr int Transpose_ILP
            = (size<2>(tTsV) * size<3>(tTsV))% 2 == 0 ? 2 : 1;
        // ((16, 1), (2, kHeadDim / 64 * kBlockN / 64))
        Tensor tDsV = logical_divide(group_modes<1, rank(tTsV) - 1>(tTsV), 
            Shape<Underscore, Int<Transpose_ILP>>{});
        // ((16, 1), (2, kHeadDim / 64 * kBlockN / 64))
        Tensor tDsVT = logical_divide(group_modes<1, rank(tTsVT) - 1>(tTsVT), 
            Shape<Underscore, Int<Transpose_ILP>>{});

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1, 1>(tDsV); ++i) {
            Tensor tTrVT = make_fragment_like(tDsVT(_, make_coord(_, _0{}),
                _0{}));
            int stage = i % Transpose_ILP;
            static_assert(size<0>(tTrVT) == 16);
            Tensor tTrVT_64 = recast<uint2>(tTrVT);
            cute::copy(tiled_s2r_v, tDsV(_, make_coord(_, i), stage), tTrVT);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_64); ++j) {
                uint32_t upper = tTrVT_64[j].x;
                uint32_t lower = tTrVT_64[j].y;
                tTrVT_64[j].x = __byte_perm(upper, lower, 0x6420);
                tTrVT_64[j].y = __byte_perm(upper, lower, 0x7531);
            }

            Tensor tTrVT_16 = recast<uint16_t>(tTrVT);
            Tensor tTrVT_32 = recast<uint32_t>(tTrVT);

            bool is_even = threadIdx.x % 2 == 0;
            bool is_01 = (threadIdx.x % 4) < 2;

            // K-Dimension Shuffle
            // 00->00, 01->20, 10->01, 11->21, 20->10, 21->30, 30->11, 31->31
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_32); j++) {
                int read_thr_idx = is_even ? 0 : 2;
                uint32_t src0 = __shfl_sync(0xffffffff, tTrVT_32(j), 
                    read_thr_idx, 4);
                read_thr_idx++;
                uint32_t src1 = __shfl_sync(0xffffffff, tTrVT_32(j), 
                    read_thr_idx, 4);

                tTrVT_16(2 * j) = is_01 ? reinterpret_cast<uint16_t *>(
                    &src0)[0] : reinterpret_cast<uint16_t *>(&src0)[1];
                tTrVT_16(2 * j + 1) = is_01 ? reinterpret_cast<uint16_t *>(
                    &src1)[0] : reinterpret_cast<uint16_t *>(&src1)[1];
            }
            /*
            // N-Dimension Shuffle
            bool is_mul_8 = (threadIdx.x >> 2) % 2 == 0;
            // 00->00, 01->40, 40->80, 41->120..

            Tensor tTrVT_64_s = recast<uint64_t>(tTrVT);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_64_s); j++) {
                int read_thr_idx = ((threadIdx.x >> 3) << 2)
                    + (threadIdx.x % 4);
                uint64_t src0 = __shfl_sync(0xffffffff, tTrVT_64_s(j), 
                    read_thr_idx);
                read_thr_idx += 16;
                uint64_t src1 = __shfl_sync(0xffffffff, tTrVT_64_s(j), 
                    read_thr_idx);

                uint32_t upper = is_mul_8 ? reinterpret_cast<uint32_t *>(
                    &src0)[0] : reinterpret_cast<uint32_t *>(&src0)[1];
                uint32_t lower = is_mul_8 ? reinterpret_cast<uint32_t *>(
                    &src1)[0] : reinterpret_cast<uint32_t *>(&src1)[1];

                tTrVT_32(2 * j) = upper;
                tTrVT_32(2 * j + 1) = lower;
            }

            // odd warps should swap in 32bit
            if(warp_idx % 2 == 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size(tTrVT_32) / 2; j++) {
                    cutlass::swap(tTrVT_32(2 * j), tTrVT_32(2 * j + 1));
                }
            }
            */
        
            cute::copy(tiled_r2s_vt, tTrVT, tDsVT(_, make_coord(_, i),  
                stage));
        }
        
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(size(TiledMma{}), 
            cutlass::arch::ReservedNamedBarriers::TransposeBarrier);

        /*
        if(thread0()) {
            print_tensor(sB);
            print_tensor(gmma_sB);
        }
        */
    }
private:
    const int warp_idx;
    const int warp_group_thread_idx;
};

/*
template<class TiledMma_, class SmemLayoutB_, class SmemLayoutAtomB_,
    class ElementB_, class GmmaElementB_>
class AsyncTranspositionINT4OperandB {
public:
using TiledMma = TiledMma_;
    using SmemLayoutB = SmemLayoutB_;
    using SmemLayoutAtomB = SmemLayoutAtomB_;
    using ElementB = ElementB_;
    using GmmaElementB = GmmaElementB_;

    using LDSM_ThrShape  = Shape<_32, _4, _1, _1>;
    using LDSM_ThrStride = Stride<_4, _1, _0, _0>;
    using LDSM_ValShape = Shape<_4, _2, _1, _2>;
    using LDSM_ValStride = Stride<_1, _4, _16, _8>;
    using LDSM_DivShape = Shape<_64, _8>;
    using TiledCopyS2RV = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x4_LDSM_T, ElementB>{}, Layout<LDSM_ThrShape, 
        LDSM_ThrStride>{}, Layout<LDSM_ValShape, LDSM_ValStride>{}));
    using STSM_ThrShape  = Shape<_8, _4, _4, _1>;
    using STSM_ThrStride = Stride<_4, _1, _32, _0>;
    using STSM_ValShape = Shape<_1, _4, _2, _2>;
    using STSM_ValStride = Stride<_0, _1, _4, _8>;
    using STSM_DivShape = Shape<_8, _16>;
    using TiledCopyR2SVT = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, ElementB>{}, Layout<STSM_ThrShape, 
        STSM_ThrStride>{}, Layout<STSM_ValShape, STSM_ValStride>{}));

    constexpr CUTLASS_HOST_DEVICE
    AsyncTranspositionINT4OperandB(int warp_idx_,
        int warp_group_thread_idx_, TiledMma, SmemLayoutB, SmemLayoutAtomB, 
        ElementB, GmmaElementB) : warp_idx(warp_idx_),
        warp_group_thread_idx(warp_group_thread_idx_) {}

    template <class TensorSmemB, class TensorTransposedSmemB, class... Ts>
    CUTLASS_DEVICE void operator()(TensorSmemB const& sB,
        TensorTransposedSmemB const& gmma_sB, int read_stage,
        cute::tuple<Ts...>& partitioned_extra_info)
    {
        // Set up for transposing V, only used if Transpose_V
        TiledCopyS2RV tiled_s2r_v;
        TiledCopyR2SVT tiled_r2s_vt;

        auto thr_s2r_v = tiled_s2r_v.get_thread_slice(warp_group_thread_idx);
        auto thr_r2s_vt = tiled_r2s_vt.get_thread_slice(warp_group_thread_idx);

        // flat_divide(sVt, LDSM_DivShape{})
        // (64, 8, kHeadDim / 64, kBlockN / 8)
        // ((16, 1), 1, 1, kHeadDim / 64, kBlockN / 32)
        Tensor tTsV = thr_s2r_v.partition_S(flat_divide(sB, LDSM_DivShape{}));  
        // flat_divide(sV, STSM_DivShape{})
        // (8, 16, kHeadDim / 8, (4, kBlockN / 64))
        // ((16, 1), 1, 1, kHeadDim / 64, (2, kBlockN / 64))
        Tensor tTsVT = thr_r2s_vt.partition_D(flat_divide(gmma_sB, STSM_DivShape{}));

        CUTE_STATIC_ASSERT_V(rank(tTsV) == rank(tTsVT));
        CUTE_STATIC_ASSERT_V(size<0>(tTsV) == size<0>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<1>(tTsV) == size<1>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<2>(tTsV) == size<2>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<3>(tTsV) == size<3>(tTsVT));
        CUTE_STATIC_ASSERT_V(size<4>(tTsV) == size<4>(tTsVT));
        // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
        static constexpr int Transpose_ILP
            = (size<2>(tTsV) * size<3>(tTsV))% 2 == 0 ? 2 : 1;
        // ((16, 1), (2, kHeadDim / 64 * kBlockN / 64))
        Tensor tDsV = logical_divide(group_modes<1, rank(tTsV) - 1>(tTsV), 
            Shape<Underscore, Int<Transpose_ILP>>{});
        // ((16, 1), (2, kHeadDim / 64 * kBlockN / 64))
        Tensor tDsVT = logical_divide(group_modes<1, rank(tTsVT) - 1>(tTsVT), 
            Shape<Underscore, Int<Transpose_ILP>>{});

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1, 1>(tDsV); ++i) {
            // ((16, 1), (2))
            Tensor tTrVTI4 = make_fragment_like(tDsVT(_, make_coord(_, _0{}),
                _0{}));
            Tensor tTrVTF8 = make_fragment_like<GmmaElementB>(
                tDsVT(_, make_coord(_, _0{}), _0{}));
            static_assert(size<0>(tTrVTI4) == 16);
            Tensor tTrVT_64 = recast<uint2>(tTrVTF8);
            cute::copy(tiled_s2r_v, tDsV(_, make_coord(_, i), read_stage), 
                tTrVTI4);
            Utils::dequantize_V(tTrVTI4, tTrVTF8, partitioned_extra_info);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_64); ++j) {
                uint32_t upper = tTrVT_64[j].x;
                uint32_t lower = tTrVT_64[j].y;
                tTrVT_64[j].x = __byte_perm(upper, lower, 0x6420);
                tTrVT_64[j].y = __byte_perm(upper, lower, 0x7531);
            }

            Tensor tTrVT_16 = recast<uint16_t>(tTrVTF8);
            Tensor tTrVT_32 = recast<uint32_t>(tTrVTF8);

            bool is_even = threadIdx.x % 2 == 0;
            bool is_01 = (threadIdx.x % 4) < 2;

            // K-Dimension Shuffle
            // 00->00, 01->20, 10->01, 11->21, 20->10, 21->30, 30->11, 31->31
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_32); j++) {
                int read_thr_idx = is_even ? 0 : 2;
                uint32_t src0 = __shfl_sync(0xffffffff, tTrVT_32(j), 
                    read_thr_idx, 4);
                read_thr_idx++;
                uint32_t src1 = __shfl_sync(0xffffffff, tTrVT_32(j), 
                    read_thr_idx, 4);

                tTrVT_16(2 * j) = is_01 ? reinterpret_cast<uint16_t *>(
                    &src0)[0] : reinterpret_cast<uint16_t *>(&src0)[1];
                tTrVT_16(2 * j + 1) = is_01 ? reinterpret_cast<uint16_t *>(
                    &src1)[0] : reinterpret_cast<uint16_t *>(&src1)[1];
            }

            // N-Dimension Shuffle
            bool is_mul_8 = (threadIdx.x >> 2) % 2 == 0;
            // 00->00, 01->40, 40->80, 41->120..

            Tensor tTrVT_64_s = recast<uint64_t>(tTrVTF8);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size(tTrVT_64_s); j++) {
                int read_thr_idx = ((threadIdx.x >> 3) << 2)
                    + (threadIdx.x % 4);
                uint64_t src0 = __shfl_sync(0xffffffff, tTrVT_64_s(j), 
                    read_thr_idx);
                read_thr_idx += 16;
                uint64_t src1 = __shfl_sync(0xffffffff, tTrVT_64_s(j), 
                    read_thr_idx);

                uint32_t upper = is_mul_8 ? reinterpret_cast<uint32_t *>(
                    &src0)[0] : reinterpret_cast<uint32_t *>(&src0)[1];
                uint32_t lower = is_mul_8 ? reinterpret_cast<uint32_t *>(
                    &src1)[0] : reinterpret_cast<uint32_t *>(&src1)[1];

                tTrVT_32(2 * j) = upper;
                tTrVT_32(2 * j + 1) = lower;
            }

            // odd warps should swap in 32bit
            if(warp_idx % 2 == 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size(tTrVT_32) / 2; j++) {
                    cutlass::swap(tTrVT_32(2 * j), tTrVT_32(2 * j + 1));
                }
            }
        
            cute::copy(tiled_r2s_vt, tTrVTF8, tDsVT(_, make_coord(_, i),  
                read_stage));
        }
        
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(size(TiledMma{}), 
            cutlass::arch::ReservedNamedBarriers::TransposeBarrier);
    }
private:
    const int warp_idx;
    const int warp_group_thread_idx;
};
*/
}