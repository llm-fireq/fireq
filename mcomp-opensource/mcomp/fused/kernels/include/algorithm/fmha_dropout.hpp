#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace cute {

/**
 * @brief No Dropout Layer
 */
struct NoDropout {
    /**
     * @brief Host side kernel arguments
     */
    struct Arguments {
        // seed for pseudo-random generator
        unsigned long long seed;
        // offset for pseudo-random generator
        unsigned long long offset;
        // drop probability
        double drop_p = 0.0;
    };

    /**
     * @brief Device side kernel params
     */
    struct Params {
        // seed for pseudo-random generator
        unsigned long long seed;
        // offset for pseudo-random generator
        unsigned long long offset;
        // drop probability
        float drop_p = 0.0f;
        // scaling for survived parameters
        half_t scale_p = half_t(1.0f);
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
        Arguments const& args, void* workspace) {
        return {args.seed, args.offset, 0.0f, (half_t)1.0f};
    }

    /**
     * @brief Initialize Pseudorandom Generator
     * 
     * @param state random generator state
     * @param params kernel parameters
     */
    CUTLASS_DEVICE void init_rand(
        curandStatePhilox4_32_10_t &state, Params params) {}

    /**
     * @brief Apply Dropout to P
     * 
     * @tparam InputP P matrix type
     * @param tVrV P matrix
     * @param state random generator state
     * @param params kernel parameters
     */
    template<class InputP>
    CUTLASS_DEVICE void apply_dropout(InputP &tVrV,
        curandStatePhilox4_32_10_t &state, Params params) {}
};

/**
 * @brief Dropout Layer
 */
struct Dropout: NoDropout {
    using Base = NoDropout;

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
        Arguments const& args, void* workspace) {
        assert(args.drop_p >= 0.0 && args.drop_p < 1);

        double scale_p = 1.0 / (1.0 - args.drop_p);

        return {args.seed, args.offset, (float)args.drop_p, (half_t)scale_p};
    }

    /**
     * @brief Initialize Pseudorandom Generator
     * 
     * @param state random generator state
     * @param params kernel parameters
     */
    CUTLASS_DEVICE void init_rand(
        curandStatePhilox4_32_10_t &state, Params params) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(params.seed, tid, params.offset, &state);
    }

    /**
     * @brief Apply Dropout to P
     * 
     * @tparam InputP P matrix type
     * @param tVrV P matrix
     * @param state random generator state
     * @param params kernel parameters
     */
    template<class InputP>
    CUTLASS_DEVICE void apply_dropout(InputP &tVrV,
        curandStatePhilox4_32_10_t &state, Params params) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K)
        // to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor output = make_tensor(tVrV.data(), 
            convert_layout_C_to_I(tVrV.layout()));

        // calculate O_i = s_i * O_i
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(output); m += 2) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<1>(output); n += 2) {
                float4 rands = curand_uniform4(&state);

                output(m, n) = rands.x < params.drop_p ? half(0.0f)
                    : params.scale_p * output(m, n);
                output(m, n + 1) = rands.y < params.drop_p ? half(0.0f)
                    : params.scale_p * output(m, n + 1);
                output(m + 1, n) = rands.z < params.drop_p ? half(0.0f)
                    : params.scale_p * output(m + 1, n);
                output(m + 1, n + 1) = rands.w < params.drop_p ? half(0.0f)
                    : params.scale_p * output(m + 1, n + 1);
            }
        }
    }
};

}  // namespace cute

