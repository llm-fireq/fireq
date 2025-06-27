#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha {

using namespace cute;

/**
 * @brief structure for kv token sequence length
 */
struct VariableLength {
    // maximum sequence length
    int max_length;
    // cumulative sequence length
    int* cumulative_length = nullptr;
    /**
     * @brief returns maximum length
     * 
     * @return int maximum sequence length
     */
    CUTE_HOST_DEVICE operator int() const {
        return max_length;
    }
};
// is_variable_length<T> is false
template<class T> struct is_variable_length : std::false_type {};
// is_variable_length<VariableLength> is true
template<> struct is_variable_length<VariableLength> : std::true_type {};
// returns whether T is VariableLength
template<class T> constexpr bool is_variable_length_v
    = is_variable_length<T>::value;

/**
 * @brief Apply variable length to problem length
 * 
 * @tparam Shape Problem shape type 
 * @tparam Idx Index type
 * @param shape problem shape (Q, KV, D, (H_G, H_R), B)
 * @param idx batch index B
 * @return cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, 
     * int>>
 */
template<class Shape, class Idx>
CUTE_HOST_DEVICE constexpr auto
apply_variable_length(Shape const& shape, Idx const& idx) {
    return transform_leaf(shape, [&](auto const& s) {
        if constexpr (is_variable_length_v<remove_cvref_t<decltype(s)>>) {
            return s.cumulative_length[idx+1] - s.cumulative_length[idx];
        }
        else {
            return s;
        }
    });
}
/**
 * @brief Apply variable length to problem length
 * 
 * @tparam Shape Problem shape type (Q, K, D, (H_G, H_R), B)
 * @tparam Coord Block coordinate type 
 * @tparam Idx Index type
 * @param shape problem shape
 * @param coord block coordinate
 * @param idx batch index
 * @return cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, 
     * int>>
 */
template<class Shape, class Coord, class Idx>
CUTE_HOST_DEVICE constexpr auto
apply_variable_length(Shape const& shape, Coord const& coord, Idx const& idx) {
    auto new_shape = apply_variable_length(shape, idx);
    auto new_coord = transform_leaf(shape, coord, [&](auto const& s, auto const& c) {
        if constexpr (is_variable_length_v<remove_cvref_t<decltype(s)>>) {
            return cute::make_tuple(c, s.cumulative_length[idx]);
        }
        else {
            return c;
        }
    });
    return cute::make_tuple(new_shape, new_coord);
}

}

namespace cute {

    // true type
    template<>
    struct is_integral<cutlass::fmha::VariableLength> : true_type {};
    /**
     * @brief Print VariableLength structure
     * 
     * @param a VariableLength structure
     */
    CUTE_HOST_DEVICE
    void print(cutlass::fmha::VariableLength a) {
        printf("Varlen<%d, %p>", a.max_length, a.cumulative_length);
    }
    
}
    