#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include <math.h>

namespace cutlass::fmha::collective {

using namespace cute;

/**
 * @brief No masking
 */
struct NoMask {
    /**
     * @brief Get number of horizontal trips in attention weight
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int number of horizontal trips
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        return ceil_div(get<1>(problem_size), get<1>(tile_shape)
            * get<1>(cluster_shape));
    }
    /**
     * @brief UNUSED. Returns 0.
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int 0
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        return 0;
    }
    /**
     * @brief Get number of horizontal trips in unmasked attention weight
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int number of horizontal trips, same as get_trip_count()
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        return get_trip_count(blk_coord, tile_shape, cluster_shape,
            problem_size);
    }

    /**
     * @brief UNUSED. Does not do anything.
     * 
     * @tparam AccQK Accumulator type of QK^T
     * @tparam IndexQK Tile index type of QK^T
     * @tparam ProblemSize Problem size type
     * @param acc_qk Accumulator type
     * @param index_qk Tile index of QK^T
     * @param problem_size Problem size
     */
    template<class AccQK, class IndexQK, class ProblemSize>
    CUTLASS_DEVICE
    static void apply_mask(
        AccQK& acc_qk,
        IndexQK const& index_qk,
        ProblemSize const& problem_size) {

        return;
    }
};
/**
 * @brief Residual masking
 */
struct ResidualMask : NoMask {

    using Base = NoMask;
    /**
     * @brief If residual exists, 1 trip, if not, 0 trips
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int residual tile count
     */
    template <class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE static int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        if (get<1>(problem_size) % (get<1>(tile_shape) * get<1>(cluster_shape)) 
            != 0) {
            return 1;
        }
        return 0;
    }
    /**
     * @brief Exclude residual tile for horizontal trip
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int if residual exists, trip - 1, if not, trip
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        // if the sequence length does not divide the tile size evenly
        if (get<1>(problem_size) % (get<1>(tile_shape) * get<1>(cluster_shape)) 
            != 0) {
            return get_trip_count(blk_coord, tile_shape, cluster_shape,
                problem_size) - 1;
        }
        return get_trip_count(blk_coord, tile_shape, cluster_shape,
            problem_size);
    }
    /**
     * @brief Mask the residual to -infinity
     * 
     * @tparam AccQK Accumulator type of QK^T
     * @tparam IndexQK Tile index type of QK^T
     * @tparam ProblemSize Problem size type
     * @param acc_qk Accumulator type
     * @param index_qk Tile index of QK^T
     * @param problem_size Problem size
     */
    template<class AccQK, class IndexQK, class ProblemSize>
    CUTLASS_DEVICE
    static void apply_mask(
        AccQK& acc_qk,
        IndexQK const& index_qk,
        ProblemSize const& problem_size) {

        // This is useful is seqlen_k % kBlockN != 0 since it masks
        // the remaining elements out from softmax.
        // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
        // issues as they are transparently taken care of by TMA and the
        // epilogue, if it is instantiated with predication support.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); i++) {
            auto pos = index_qk(i);
            bool do_mask = get<1>(pos) >= get<1>(problem_size);
            acc_qk(i) = do_mask ? -std::numeric_limits<half_t>::infinity()
                : acc_qk(i);
        }
    }
};

/**
 * @brief Causal masking
 */
struct CausalMask : NoMask {

    using Base = NoMask;
    /**
     * @brief Get number of horizontal trips in attention weight
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int number of horizontal trips
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        // See note below on different ways to think about causal attention
        // Again, we'd add the offset_q into the max_blocks_q calculation
        int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, 
            cluster_shape, problem_size) * get<1>(cluster_shape);
        int max_blocks_q = ceil_div((get<0>(blk_coord) + (get<0>(cluster_shape) 
            - (get<0>(blk_coord) % get<0>(cluster_shape))))
            * get<0>(tile_shape), get<1>(tile_shape) * get<1>(cluster_shape));

        return std::min(max_blocks_k, max_blocks_q) / get<1>(cluster_shape);
    }
    /**
     * @brief number of trips within a tile
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int m / n
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_masked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        return ceil_div(get<0>(tile_shape) * get<0>(cluster_shape),
            get<1>(tile_shape) * get<1>(cluster_shape));
    }
    /**
     * @brief Exclude masked tile for horizontal trip
     * 
     * @tparam BlkCoord Block coordinate type
     * @tparam TileShape Tile shape type
     * @tparam ProblemSize Problem size type
     * @param blk_coord block coordinate
     * @param tile_shape tile shape
     * @param cluster_shape thread block cluster shape
     * @param problem_size problem size
     * @return int if residual exists, trip - 1, if not, trip
     */
    template<class BlkCoord, class TileShape, class ClusterShape,
        class ProblemSize>
    CUTLASS_DEVICE
    static int get_unmasked_trip_count(
        BlkCoord const& blk_coord,
        TileShape const& tile_shape,
        ClusterShape const& cluster_shape,
        ProblemSize const& problem_size) {

        return get_trip_count(blk_coord, tile_shape, cluster_shape,
            problem_size) - get_masked_trip_count(blk_coord, tile_shape, 
            cluster_shape, problem_size);
    }
    /**
     * @brief Mask the upper triangular to -infinity
     * 
     * @tparam AccQK Accumulator type of QK^T
     * @tparam IndexQK Tile index type of QK^T
     * @tparam ProblemSize Problem size type
     * @param acc_qk Accumulator type
     * @param index_qk Tile index of QK^T
     * @param problem_size Problem size
     */
    template<class AccQK, class IndexQK, class ProblemSize>
    CUTLASS_DEVICE
    static void apply_mask(
        AccQK& acc_qk,
        IndexQK const& index_qk,
        ProblemSize const& problem_size) {

        // There are two ways to do causal if N_Q != N_K
        // (1) is to assume that the Q is at the beginning of the matrix
        //    - this is what we demonstrate here
        // (2) is that it is at the end of the matrix
        //    - this is usually what we want for inference settings
        //      where we only compute the next row and use cache for the rest
        //    - if you'd like this, you only need to add an offset like so:
        //      get<0>(pos) + offset_q < get<1>(pos)
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); i++) {
            auto pos = index_qk(i);
            if ((get<0>(pos) < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size))) {
                acc_qk(i) = -INFINITY;
            }
        }
    }
};

} // namespace cutlass::fmha