#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include <math.h>

namespace cute {

/**
 * @brief Scaleing before masking
 */
struct Scale {
    /**
     * @brief Host side kernel arguments
     */
    struct Arguments {
        // if zero, defaults to 1/sqrt(D)
        double scale_softmax = 0.0;

        // scaling factors to dequantize QK
        double scale_q = 1.0;
        double scale_k = 1.0;
    };

    /**
     * @brief Device side kernel params
     */
    struct Params {
        half scale_softmax;
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
    static Params
    to_underlying_arguments(ProblemShape const& problem_shape,
        Arguments const& args, void* workspace) {
        double scale_softmax = args.scale_softmax;
        if (scale_softmax == 0.0) {
            scale_softmax = 1.0 / std::sqrt((double)get<2>(problem_shape));
        }

        scale_softmax *= args.scale_q * args.scale_k;

        return {(half)scale_softmax};
    }

    /**
     * @brief Scale QK^T
     * 
     * @tparam AccQK S matrix type
     * @param tOrO O matrix
     * @param params kernel marameters
     */
    template<typename AccQK>
    CUTLASS_DEVICE static void
    scale(AccQK &tSrS, Params params) {
        //((2, 2, V), MMA_M, MMA_KV)
        static_assert(decltype(rank(tSrS))::value == 3);
        static_assert(decltype(rank<0>(tSrS))::value == 3);
        static_assert(decltype(size<0, 0>(tSrS))::value == 2);
        static_assert(decltype(size<0, 1>(tSrS))::value == 2);
        static_assert(is_same_v<typename AccQK::value_type, cutlass::half_t>);

        const half2 scales = __half2half2(params.scale_softmax);
        //((1, 2, V), MMA_M, MMA_KV)
        Tensor frag_32b = recast<half2>(tSrS);

        // calculate O_i = s_i * O_i
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<2>(tSrS); n++) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tSrS); m++) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < size<0, 2>(tSrS); v++) {
                    CUTLASS_PRAGMA_UNROLL
                    for(int i = 0; i < size<0, 1>(tSrS); i++) {
                        half2 &S = frag_32b(make_coord(_0{}, i, v), m, n);
                        S = __hmul2(S, scales);
                    }
                }
            }
        }
    }
};

} // namespace cute