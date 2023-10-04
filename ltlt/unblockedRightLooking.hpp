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

void ltlt_unblockeRL(const row_view<const double>& lx0, const matrix_view<double>& X)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1 DYNAMIC>(X);

    row<double> temp{X.length(0)};

    while{B}
    {
        // (T  || m  |   B    )
        // (R0 || r1 | r2 | R3) 4 * 4 partition

        atuo [R0, r1, r2, R3] = repartition<DYNAMIC, 1>(T, m, B, 1)
    }
}


#endif
