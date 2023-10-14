#ifndef __UNBLOCKRIGHTLOOKING_HPP
#define __UNBLOCKRIGHTLOOKING_HPP

// include reqiured header file

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>

#include "flame.hpp"
#include "blas.h"
#include "marray_view.hpp"
#include "expression.hpp"

using namespace MArray;
using std::tie;

void ltlt_unblockeRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1 DYNAMIC>(X);
    auto n = X.length(0);
    if (k == -1)
        k = n;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};

    if (first_column)
        blas::skr2(1.0, L[B, m], X[B, m], 1.0, X[B, B]);

    while(B.size() > n - k)
    {
        // (T  || m  |   B    )
        // (R0 || r1 | r2 | R3) 4 * 4 partition

        auto [R0, r1, r2, R3] = repartition(T, m, B);
        
        L[R3,r2] = X[R3, r1] / X[r2, r1];
        
        blas::skr2(1.0, L[R3, r2], X[R3, r2], 1.0, X[R3, R3]);

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
    }
}


#endif
