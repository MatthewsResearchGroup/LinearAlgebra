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
 
void ltlt_blockLL(const row_view<const double>& lx0, const matrix_view<double>& X)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);
 
    row<double> temp{X.length(0)};
 
    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);


        /*left-looking*/

        blas::skew_tridiag_gemm(-1, L[R2 | r3 | R4][R0 | r1], subdiag(X[R0 | r1][R0 | r1]), L[R1 | r2][R0| r1].T(), 1, X[R2 | r3 | R4][r1 | R2]);

        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2]); 

        // ( R0 | r1 || r2 | R3 )
        // (    T    ||  m |  B )
 

        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
