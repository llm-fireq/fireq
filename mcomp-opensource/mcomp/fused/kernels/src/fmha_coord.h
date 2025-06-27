#pragma once

#include "cutlass/coord.h"

namespace cutlass {
namespace fmha {


/**
 * @brief FmhaCoord is a structure derived from Coord<3> that specifies a 
 * location within the coordinate space of a GEMM problem.
 */
struct FmhaCoord : public Coord<3, int> {

    /// Integer-valued index
    typedef int Index;
  
    /// Base type is a Coord of rank=6
    typedef Coord<3, Index> Base;
  
    /// GEMM Q dimension - rows of the output O matrix
    static int const kQ = 0;
  
    /// GEMM KV dimension - inner dimension of the FMHA problem
    static int const kKV = 1;
  
    /// GEMM D dimension - columns of the GEMM problem
    static int const kD = 2;
  
    //
    // Methods
    //
  
    /**
     * @brief Default ctor
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord() { }
  
    /// Constructs from Coord<3> and a batch

    /**
     * @brief Constructs from Coord<3> and a batch
     * 
     * @param coord coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord(Coord<3, Index> const& coord): Base(make_Coord(coord[0],
        coord[1], coord[2])) { }
  
    /// Helper to construct from a K, N, M, batch variables

    /**
     * @brief Helper to construct from a Q, KV, D, batch variables
     * 
     * @param q Query
     * @param kv Key, Value
     * @param d Head Dimension
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord(Index q, Index kv, Index d): Base(make_Coord(q, kv, d)) { }
  
    /**
     * @brief Returns the FMHA Q coordinate
     * 
     * @return Index const& Q coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& q() const { return this->at(kQ); }

    /**
     * @brief Returns reference to the FMHA Q coordinate
     * 
     * @return Index & Q coordinate reference
     */
    CUTLASS_HOST_DEVICE
    Index & q() { return this->at(kQ); }
  
    /**
     * @brief Returns the FMHA KV coordinate
     * 
     * @return Index const& KV coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& kv() const { return this->at(kKV); }
  
    /**
     * @brief Returns reference to the FMHA KV coordinate
     * 
     * @return Index & KV coordinate reference
     */
    CUTLASS_HOST_DEVICE
    Index & kv() { return this->at(kKV); }
  
    /**
     * @brief Returns the FMHA D coordinate
     * 
     * @return Index const& D coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& d() const { return this->at(kD); }
  
    /**
     * @brief Returns reference to the FMHA D coordinate
     * 
     * @return Index & D coordinate reference
     */
    CUTLASS_HOST_DEVICE
    Index & d() { return this->at(kD); }

    /**
     * @brief Obtains a Coord<3> from FmhaCoord
     * 
     * @return Coord<3> (q, kv, d)
     */
    CUTLASS_HOST_DEVICE
    Coord<3> qkvd() const {
        return make_Coord(q(), kv(), d());
    }
  
    /**
     * @brief Obtains a Coord<3> from FmhaCoord
     * 
     * @return Coord<3> (d, kv, q)
     */
    CUTLASS_HOST_DEVICE
    Coord<3> dkvq() const {
        return make_Coord(d(), kv(), q());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (kv, q)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> kvq() const {
        return make_Coord(kv(), q());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (q, kv)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> qkv() const {
        return make_Coord(q(), kv());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (q, d)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> qd() const {
        return make_Coord(q(), d());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (d, q)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> dq() const {
        return make_Coord(d(), q());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (kv, d)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> kvd() const {
        return make_Coord(kv(), d());
    }
  
    /**
     * @brief Obtains a Coord<2> from FmhaCoord
     * 
     * @return Coord<2> (d, kv)
     */
    CUTLASS_HOST_DEVICE
    Coord<2> dkv() const {
        return make_Coord(d(), kv());
    }
  
    //
    // Coord operators
    //

    /**
     * @brief Element-wise addition
     * 
     * @param b base type coordinate
     * @return FmhaCoord added FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord operator+(Base const& b) const {
        return FmhaCoord(Base::operator+(b));
    }
  
    /**
     * @brief Element-wise subtraction
     * 
     * @param b base type coordinate
     * @return FmhaCoord subtracted FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord operator-(Base const& b) const {
        return FmhaCoord(Base::operator-(b));
    }
  
    /**
     * @brief Element-wise multiplication
     * 
     * @param b base type coordinate
     * @return FmhaCoord multiplied FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord operator*(Base const& b) const {
        return FmhaCoord(Base::operator*(b));
    }
  
    /**
     * @brief Element-wise division
     * 
     * @param b base type coordinate
     * @return FmhaCoord divided FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord operator/(Base const& b) const {
        return FmhaCoord(Base::operator/(b));
    }
  
    /**
     * @brief In-place addition
     * 
     * @param b base type coordinate
     * @return FmhaCoord& added FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }
  
    /**
     * @brief In-place subtraction
     * 
     * @param b base type coordinate
     * @return FmhaCoord& subtracted FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }
  
    /**
     * @brief In-place multiplication
     * 
     * @param b base type coordinate
     * @return FmhaCoord& multiplied FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }
  
    /**
     * @brief In-place division
     * 
     * @param b base type coordinate
     * @return FmhaCoord& divided FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief BatchedFmhaCoord is a structure derived from Coord<6> that specifies 
 * a location within the coordinate space of a batched GEMM problem.
 */
struct BatchedFmhaCoord : public Coord<6, int> {

    /// Integer-valued index
    typedef int Index;
  
    /// Base type is a Coord of rank=6
    typedef Coord<6, Index> Base;
  
    /// FMHA Q dimension - rows of the output O matrix
    static int const kQ = 0;
  
    /// FMHA KV dimension - inner dimension of the FMHA problem
    static int const kKV = 1;
  
    /// FMHA D dimension - columns of the output O matrix
    static int const kD = 2;

    /// FMHA Grouped Head dimension - inner dimension of the GEMM problem
    static int const kHG = 3;

    /// FMHA KV Head dimension - inner dimension of the GEMM problem
    static int const kHR = 4;
  
    /// FMHA Batch dimension - inner dimension of the GEMM problem
    static int const kBatch = 5;
  
    //
    // Methods
    //
  
    /// Default ctor

    /**
     * @brief Default ctor
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord() { }
  
    /// Constructs from Coord<6>

    /**
     * @brief Constructs from Coord<6>
     * 
     * @param coord coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord(Base const& coord): Base(coord) { }
  
    /// Helper to construct from a Q, KV, D, hg, hr, and batch variables
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord(Index q, Index kv, Index d, Index hg, Index hr, Index b)
        : Base({q, kv, d, hg, hr, b}) { }
  
    /// Returns the GEMM M coordinate

    /**
     * @brief Returns the FMHA Q coordinate
     * 
     * @return Index const& Q coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& q() const { return this->at(kQ); }
  
    /**
     * @brief Returns reference to the FMHA Q coordinate
     * 
     * @return Index & Q coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & q() { return this->at(kQ); }
  
    /**
     * @brief Returns the FMHA KV coordinate
     * 
     * @return Index const& KV coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& kv() const { return this->at(kKV); }
  
    /**
     * @brief Returns reference to the FMHA KV coordinate
     * 
     * @return Index & KV coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & kv() { return this->at(kKV); }
  
    /**
     * @brief Returns the FMHA D coordinate
     * 
     * @return Index const& D coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& d() const { return this->at(kD); }
  
    /**
     * @brief Returns reference to the FMHA D coordinate
     * 
     * @return Index & D coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & d() { return this->at(kD); }

    /**
     * @brief Returns the FMHA grouped head coordinate
     * 
     * @return Index const& grouped head coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& hg() const { return this->at(kHG); }
  
    /**
     * @brief Returns reference to the FMHA grouped head coordinate
     * 
     * @return Index & grouped head coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & hg() { return this->at(kHG); }

    /**
     * @brief Returns the FMHA KV head coordinate
     * 
     * @return Index const& KV head coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& hr() const { return this->at(kHR); }
  
    /**
     * @brief Returns reference to the FMHA KV head coordinate
     * 
     * @return Index & KV head coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & hr() { return this->at(kHR); }
  
    /**
     * @brief Returns the FMHA batch coordinate
     * 
     * @return Index const& batch coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index const& batch() const { return this->at(kBatch); }
  
    /**
     * @brief Returns reference to the FMHA batch coordinate
     * 
     * @return Index & batch coordinate constant index
     */
    CUTLASS_HOST_DEVICE
    Index & batch() { return this->at(kBatch); }


    /**
     * @brief Obtains a FmhaCoord from BatchedFmhaCoord
     * 
     * @return FmhaCoord (q, kv, d)
     */
    CUTLASS_HOST_DEVICE
    FmhaCoord qkvd() const {
      return FmhaCoord(q(), kv(), d());
    }
  
    /// Obtains a Coord<6> from BatchedFmhaCoord

    /**
     * @brief Obtains a Coord<6> from BatchedFmhaCoord
     * 
     * @return Coord<6> (q, kv, d, hg, hr, b)
     */
    CUTLASS_HOST_DEVICE
    Coord<6> qkvdhghrb() const {
      return Coord<6>({q(), kv(), d(), hg(), hr(), batch()});
    }
  
    //
    // Coord operators
    //
  
    /**
     * @brief Element-wise addition
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord added FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord operator+(Base const& b) const {
      return BatchedFmhaCoord(Base::operator+(b));
    }
  
    /**
     * @brief Element-wise subtraction
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord subtracted FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord operator-(Base const& b) const {
      return BatchedFmhaCoord(Base::operator-(b));
    }
  
    /**
     * @brief Element-wise multiplication
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord multiplied FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord operator*(Base const& b) const {
      return BatchedFmhaCoord(Base::operator*(b));
    }
  
    /**
     * @brief Element-wise division
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord divided FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord operator/(Base const& b) const {
      return BatchedFmhaCoord(Base::operator/(b));
    }
  
    /**
     * @brief In-place addition
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord& added FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord& operator+=(Base const& b) {
      Base::operator+=(b);
      return *this;
    }
  
    /**
     * @brief In-place subtraction
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord& subtracted FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord& operator-=(Base const& b) {
      Base::operator-=(b);
      return *this;
    }
  
    /**
     * @brief In-place multiplication
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord& multiplied FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord& operator*=(Base const& b) {
      Base::operator*=(b);
      return *this;
    }
  
    /**
     * @brief In-place division
     * 
     * @param b base type coordinate
     * @return BatchedFmhaCoord& divided FMHA coordinate
     */
    CUTLASS_HOST_DEVICE
    BatchedFmhaCoord& operator/=(Base const& b) {
      Base::operator/=(b);
      return *this;
    }
};

/**
 * @brief Transforms cute::Coord to FmhaCoord or BatchedFmhaCoord
 * 
 * @tparam Tuple cute::Coord
 * @param tuple Coordinates
 * @return FmhaCoord or BatchedFmhaCoord
 */
template <class Tuple>
CUTLASS_HOST_DEVICE
auto to_fmha_coord(Tuple tuple) {
    static_assert(cute::rank(tuple) <= 4,
        "Can only convert tuples of rank <= 4.");

    if constexpr (cute::rank(tuple) == 3) {
        auto tuple_qkvd = cute::append<3>(tuple, cute::Int<0>{});
        return FmhaCoord(cute::size<0>(tuple_qkvd), cute::size<1>(tuple_qkvd), cute::size<2>(tuple_qkvd));
    }
    else {
        static_assert(cute::rank<3>(tuple) == 2);
        static_assert(cute::rank<3, 0>(tuple) == 2);
        return BatchedFmhaCoord(cute::size<0>(tuple), cute::size<1>(tuple), 
            cute::size<2>(tuple), cute::size<3, 0, 0>(tuple),
            cute::size<3, 0, 1>(tuple), cute::size<3, 1>(tuple));
    }
}

}
}