#ifndef MARRAY_FLAME_HPP
#define MARRAY_FLAME_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>

#include "marray_view.hpp"
#include "expression.hpp"
#include "blas.h"
#include "flame.hpp"

using namespace MArray;
using std::tie;
 
void ltlt_blockRL(const row_view<const double>& lx0, const matrix_view<double>& X)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);
 
    row<double> temp{X.length(0)};
 
    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
 
        /*right-looking*/

        LTLT_UNB_0(X[r1 | R2 | r3 | R4][r1 | R2]);

        blas::skew_tridiag_rankk(-1, L[r3 | R4][R2 | r3], subdiag(X[R2 | r3][R2 | r3]), 1, X[r3 | R4][r3 | R4]);
 
        // X44 += l43 x43^T - x43 l43^T
        blas::skr2(1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
 
        // ( R0 | r1 || r2 | R3 )
        // (    T    ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
