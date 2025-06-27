#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include <math.h>
#include <curand.h>

namespace cute {

/**
 * @brief Host side kernel arguments
 */
struct SoftmaxArguments {
    // dropout probability
    double drop_p = 0.0;
    // scaling factor to quantize P
    double scale_p = 448.0;
    // scaling factor to dequantize v
    double scale_v = 1.0;
    // scaling factor to quantize O
    double inv_scale_o = 1.0;
};

/**
 * @brief Device side kernel params
 */
struct SoftmaxParams {
    // scale factor to P
    half_t max_offset;
    // scaling factor to quantize O
    float inv_scale_o;
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
SoftmaxParams to_underlying_softmax_arguments(
    ProblemShape const& problem_shape, SoftmaxArguments const& args,
    void* workspace) {
    assert(args.drop_p >= 0.0 && args.drop_p < 1);
    assert(args.scale_p <= 448.0);

    double max_offset = log2(args.scale_p * (1.0 - args.drop_p));
    double inv_scale_o = args.scale_v * args.inv_scale_o
        / ((1.0 - args.drop_p) * args.scale_p);

    return {(half_t)max_offset, (float)inv_scale_o};
}

/**
 * @brief Softmax Operation Structure
 * 
 * @tparam MMA_M
 */
template <int MMA_M>
struct Softmax {
    using TensorM = decltype(make_tensor<half>(Layout<Shape<_2, 
        Int<MMA_M>>>{}));
    using TensorS = decltype(make_tensor<float>(Layout<Shape<_2, 
        Int<MMA_M>>>{}));
    using TensorL = TensorS;

    TensorM tMrM;
    TensorS tsrs;
    TensorL tLrL;

    /**
     * @brief Perform softmax on V matrix(S and P)
     * 
     * @tparam AccQK V matrix type(S and P)
     * @param tVrV V matrix
     * @param params kernel arguments
     */
    template<bool Prologue, class AccQK>
    CUTLASS_DEVICE void softmax_preexp(AccQK &tVrV,
        SoftmaxParams const& params) {
        static_assert(decltype(rank(tVrV))::value == 3);
        static_assert(decltype(rank<0>(tVrV))::value == 3);
        static_assert(decltype(size<0, 0>(tVrV))::value == 2);
        static_assert(decltype(size<0, 1>(tVrV))::value == 2);
        static_assert(decltype(stride<0, 0>(tVrV))::value == 1);
        static_assert(decltype(stride<0, 1>(tVrV))::value == 2);
        static_assert(decltype(size<1>(tVrV))::value == MMA_M);
        static_assert(is_same_v<typename AccQK::value_type, cutlass::half_t>);

        constexpr int MMA_KV = decltype(size<2>(tVrV))::value;
        constexpr int MMA_V = decltype(size<0, 2>(tVrV))::value;

        // copy from m_i to m_old
        Tensor tMrM_old = make_tensor<half>(_2{});
        Tensor fragM_old = recast<half2>(tMrM_old);
        // ((1, 2, V), M, N)
        Tensor fragS = recast<half2>(tVrV);
        // (1, M)
        Tensor fragM = recast<half2>(tMrM);

        // m_old = m, m = max(m_old, rowmax(S)), s = exp(m_old - m)
        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < MMA_M; m++) {
            // copy m_i to m_old
            if constexpr(!Prologue)
                fragM_old(_0{}) = fragM(_0{});
            if constexpr(Prologue)
                fragM_old(_0{}) = __half2half2((half)(-INFINITY));

            // calculate m_i = rowmax(S_i)
            Tensor row_max = make_tensor<half2>(Layout<Shape<_2, _2>>{});
            CUTLASS_PRAGMA_UNROLL
            for(int j = 0; j < 2; j++) {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++)
                    row_max(i, j) = __half2half2(tMrM(i, m));
            }

            // thread_wise max reduction
            CUTLASS_PRAGMA_UNROLL
            for(int n = 0; n < MMA_KV; n++) {
                CUTLASS_PRAGMA_UNROLL
                for(int v = 0; v < MMA_V; v += 2) {
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0; j < 2; j++) {
                        CUTLASS_PRAGMA_UNROLL
                        for(int i = 0; i < 2; i++)
                            row_max(i, j) = __hmax2(row_max(i, j),
                                fragS(make_coord(_0{}, i, v + j), m, n));
                    }
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 2; i++)
                row_max(i, _0{}) = __hmax2(row_max(i, _0{}), row_max(i, _1{}));
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 2; i++)
                tMrM(i, m) = __hmax(row_max(i, _0{}).x, row_max(i, _0{}).y);
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 2; i++)
                tMrM(i, m) = __hmax(tMrM(i, m), tMrM_old(i));

            // max reduction with other 4 threads
            // thread 0 <-> 2, 1 <-> 3
            fragM(_0{}, m) = __hmax2(fragM(_0{}, m),
                __shfl_xor_sync(0xffffffff, fragM(_0{}, m), 2, 4));
            // thread 0 <-> 1, 2 <-> 3
            fragM(_0{}, m) = __hmax2(fragM(_0{}, m),
                __shfl_xor_sync(0xffffffff, fragM(_0{}, m), 1, 4));

            // scale = exp(m_old - m)
            if constexpr(Prologue) {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++)
                    tsrs(i, m) = 1.0f;
            }
            if constexpr(!Prologue) {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++) {
                    tsrs(i, m) = M_LOG2Ef32 * (float)(tMrM_old(i)
                        - tMrM(i, m));
                }
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++) {
                    tsrs(i, m) = exp2f(tsrs(i, m));
                }
            }
        }
        // calculate l_i = s_i * l_i
        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < MMA_M; m++) {
            if constexpr(Prologue) {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++)
                    tLrL(i, m) = 0.f;
            }
            if constexpr(!Prologue) {
                CUTLASS_PRAGMA_UNROLL
                for(int i = 0; i < 2; i++)
                    tLrL(i, m) *= tsrs(i, m);
            }
        }
        // calculate S = log2(e) (S-m) + log(448)
        // thread_wise sum reduction
        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < MMA_M; m++) {
            const half log2_e = (half)M_LOG2Ef32;
            const half2 log2_e_2 = __half2half2(log2_e);

            Tensor minus_m = make_tensor<half2>(_2{});
            
            // -m = log(448) - log2(e) * m
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < 2; i++) {
                minus_m(i) = (tMrM(i, m) == (half)(-INFINITY))
                    ? __half2half2((half)0.f)
                    : __half2half2(params.max_offset.to_half()
                    - tMrM(i, m) * log2_e);
            }
            
            // S = S * log2(e) - m
            CUTLASS_PRAGMA_UNROLL
            for(int n = 0; n < MMA_KV; n++) {
                CUTLASS_PRAGMA_UNROLL
                for(int v = 0; v < MMA_V; v += 2) {
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0; j < 2; j++) {
                        CUTLASS_PRAGMA_UNROLL
                        for(int i = 0; i < 2; i++) {
                            fragS(make_coord(_0{}, i, v + j), m, n) = __hfma2(
                                fragS(make_coord(_0{}, i, v + j), m, n), 
                                log2_e_2, minus_m(i));
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Perform softmax on V matrix(S and P)
     * 
     * @tparam AccQK V matrix type(S and P)
     * @param tVrV V matrix
     * @param params kernel arguments
     */
     template<class AccQK>
     CUTLASS_DEVICE void softmax_exp(AccQK &tVrV,
         SoftmaxParams const& params) {
         static_assert(decltype(rank(tVrV))::value == 3);
         static_assert(decltype(rank<0>(tVrV))::value == 3);
         static_assert(decltype(size<0, 0>(tVrV))::value == 2);
         static_assert(decltype(size<0, 1>(tVrV))::value == 2);
         static_assert(decltype(stride<0, 0>(tVrV))::value == 1);
         static_assert(decltype(stride<0, 1>(tVrV))::value == 2);
         static_assert(decltype(size<1>(tVrV))::value == MMA_M);
         static_assert(is_same_v<typename AccQK::value_type, cutlass::half_t>);
 
         constexpr int MMA_KV = decltype(size<2>(tVrV))::value;
         constexpr int MMA_V = decltype(size<0, 2>(tVrV))::value;
 
         // ((1, 2, V), M, N)
         Tensor fragS = recast<half2>(tVrV);
         Tensor fragExp = recast<uint32_t>(tVrV);
 
         // S = exp(S)
         CUTLASS_PRAGMA_UNROLL
         for(int m = 0; m < MMA_M; m++) {
             CUTLASS_PRAGMA_UNROLL
             for(int n = 0; n < MMA_KV; n++) {
                 CUTLASS_PRAGMA_UNROLL
                 for(int v = 0; v < MMA_V; v += 2) {
                     CUTLASS_PRAGMA_UNROLL
                     for(int j = 0; j < 2; j++) {
                         CUTLASS_PRAGMA_UNROLL
                         for(int i = 0; i < 2; i++) {
                             ;
                             asm volatile ("ex2.approx.f16x2 %0, %1;" : "=r"
                                 (fragExp(make_coord(_0{}, i, v + j), m, n))
                                 : "r" (fragExp(make_coord(_0{}, i, v + j), m, 
                                 n)));
                         }
                     }
                 }
             }
         }
         // calculate l_i = l_i + rowsum(P_i)
         // thread_wise max reduction
         CUTLASS_PRAGMA_UNROLL
         for(int m = 0; m < MMA_M; m++) {    
             CUTLASS_PRAGMA_UNROLL
             for(int n = 0; n < MMA_KV; n++) {
                 Tensor row_sum = make_tensor<half2>(Layout<Shape<_2, _2>>{});
 
                 // initialize row sum
                 CUTLASS_PRAGMA_UNROLL
                 for(int j = 0; j < 2; j++) {
                     CUTLASS_PRAGMA_UNROLL
                     for(int i = 0; i < 2; i++)
                         row_sum(i, j) = __half2half2((half)0.f);
                 }
 
                 // rowsum += S(i, j)
                 CUTLASS_PRAGMA_UNROLL
                 for(int v = 0; v < MMA_V; v += 2) {
                     CUTLASS_PRAGMA_UNROLL
                     for(int j = 0; j < 2; j++) {
                         CUTLASS_PRAGMA_UNROLL
                         for(int i = 0; i < 2; i++) {
                             row_sum(i, j) = __hadd2(row_sum(i, j),
                                 fragS(make_coord(_0{}, i, v + j), m, n));
                         }
                     }
                 }
 
                 // Summation to L
                 CUTLASS_PRAGMA_UNROLL
                 for(int i = 0; i < 2; i++) {
                     row_sum(i, _0{}) = __hadd2(row_sum(i, _0{}),
                         row_sum(i, _1{}));
                 }
                 CUTLASS_PRAGMA_UNROLL
                 for(int i = 0; i < 2; i++) {
                     tLrL(i, m) += (float)__hadd(row_sum(i, _0{}).x,
                         row_sum(i, _0{}).y);
                 }
             }
         }
     }

    /**
     * @brief Rescale O
     * 
     * @tparam AccPV O matrix type
     * @tparam RowS 1D Tensor with half_t
     * @param tOrO O matrix
     * @param tsrs scale vector
     */
    template<class AccPV>
    CUTLASS_DEVICE void correction_rescale(AccPV &tOrO) {
        static_assert(decltype(rank(tOrO))::value == 3);
        static_assert(decltype(rank<0>(tOrO))::value == 3);
        static_assert(decltype(size<0, 0>(tOrO))::value == 2);
        static_assert(decltype(size<0, 1>(tOrO))::value == 2);
        static_assert(decltype(size<1>(tOrO))::value == MMA_M);
        static_assert(is_same_v<typename AccPV::value_type, float>);

        // calculate O_i = s_i * O_i
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<2>(tOrO); n++) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < MMA_M; m++) {
                CUTLASS_PRAGMA_UNROLL
                for(int v = 0; v < size<0, 2>(tOrO); v++) {
                    CUTLASS_PRAGMA_UNROLL
                    for(int i = 0; i < 2; i++) {
                        CUTLASS_PRAGMA_UNROLL
                        for(int j = 0; j < 2; j++)
                            tOrO(make_coord(j, i, v), m, n) *= tsrs(i, m);
                    }
                }
            }
        }
    }

    /**
     * @brief Compute inverse L for epilogue correcting O
     * 
     * @tparam TensorLSE 1D Tensor with float
     * @param tLrL rowsum vector
     */
    template<class TensorInvL>
    CUTLASS_DEVICE void compute_inverse_L(TensorInvL &inv_tLrL,
        int m, SoftmaxParams const& params) {
        static_assert(decltype(rank(inv_tLrL))::value == 2);
        static_assert(decltype(size<0>(inv_tLrL))::value == 2);
        static_assert(decltype(size<1>(inv_tLrL))::value == MMA_M);
        static_assert(is_same_v<typename TensorInvL::value_type, bfloat16_t>);
        
        // max reduction with other 4 threads

        // thread 0 <-> 2, 1 <-> 3
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; i++)
            tLrL(i, m) += __shfl_xor_sync(0xffffffff, tLrL(i, m), 2, 4);

        // thread 0 <-> 1, 2 <-> 3
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; i++)
            tLrL(i, m) += __shfl_xor_sync(0xffffffff, tLrL(i, m), 1, 4);

        // scaled 1/l_i 
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; i++) {
            inv_tLrL(i, m) = (tLrL(i, m) == 0.f || tLrL(i, m) != tLrL(i, m))
                ? (bfloat16_t)0.f : (bfloat16_t)(1.0f / tLrL(i, m));
        }
    }

    /**
     * @brief Compute log sum exp
     * 
     * @tparam TensorLSE logsumexp with half
     * @param tLrL rowsum vector
     */
    template<class TensorLSE>
    CUTLASS_DEVICE void compute_LSE(TensorLSE &tLSErLSE, int m) {
        static_assert(decltype(rank(tLSErLSE))::value == 2);
        static_assert(decltype(size<0>(tLSErLSE))::value == 2);
        static_assert(decltype(size<1>(tLSErLSE))::value == MMA_M);
        static_assert(is_same_v<typename TensorLSE::value_type, half_t>);
        
        // max reduction with other 4 threads
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; i++)
            tLSErLSE(i, m) = (half_t)::log(tLrL(i, m));
        
        // ((1, 2, V), M, N)
        Tensor fragLSE = recast<half2>(tLSErLSE);
        Tensor fragM = recast<half2>(tMrM);
        fragLSE(_0{}, m) = __hsub2(fragLSE(_0{}, m), fragM(_0{}, m));
    }
};

}