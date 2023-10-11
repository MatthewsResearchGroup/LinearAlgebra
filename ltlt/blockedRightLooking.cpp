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

#include <functional>

using namespace MArray;
using std::tie;
 
void ltlt_blockRL(const matrix_view<double>& X, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB, len_type k = -1, bool first_column = false)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);
    auto n = X.length(0);
    if (k == -1)
        k = n;
    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};
    if (first_column)
        blas::skr2(1.0, L[B, m], X[B, m], 1.0, X[B, B]);

    while (B.size() > n - k)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
 
        /*right-looking*/

        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2 | r3 | R4],  (r1 | R2).size(), false);

        blas::skew_tridiag_rankk(-1, L[r3 | R4][R2 | r3], subdiag(X[R2 | r3][R2 | r3]), 1, X[r3 | R4][r3 | R4]);
 
        // X44 += l43 x43^T - x43 l43^T
        blas::skr2(1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
 
        // ( R0 | r1 || r2 | R3 )
        // (    T    ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
