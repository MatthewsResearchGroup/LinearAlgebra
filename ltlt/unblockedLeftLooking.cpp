#ifndef __UNBLOCKEDLEFTLOOKING_HPP__
#define __UNBLOCKEDLEFTLOOKING_HPP__

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

void ltlt_unblockLL(const matrix_view<double>& X, len_type k = -1, bool first_column = false)
{
    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);
    auto n = X.length(0);

    if (k == -1)
        k = n;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};


    while(B.size() > n - k)
    {
        // (T  || m  ||   B    )
        // (R0 || r1 || r2 | R3) 4 * 4 partition
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        temp[R0] = L[r1][R0];
        temp[r1] = 1;
        skewtrigemv(-1.0, L[r2 | R3][R0 | r1], subdiag(X[R0 | r1 ][R0, r1]), temp[R0 | r1], 1, X[r2 | R3][r1]);

        L[R3, r2] = X[R3,r1] / X[r2, r1];

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);


    }
}

/*
 * x <- alpha T x
 */
void sktrmv(double alpha, const row_view<const double>& T, const row_view<double>& x)
{
    auto n = x.length();
    MARRAY_ASSERT(T.length(0) == n);
    MARRAY_ASSERT(T.length(1) == n);

    if (n == 0)
        return;

    if (n == 1)
    {
        x[0] = 0.0;
        return;
    }
    
    auto ximinus1 = x[0];
    x[0] = -T[0] * x[1];

    for (auto i : range(1,n-1))
    {
        auto xi = T[i-1] * ximinus1 - T[i] * x[i+1];
        ximinus1 = x[i];
        x[i] = xi;
    }

    x[n-1] = T[n-2] * ximinus1;
}

void sktrmv(double alpha, const row_view<const double>& T, const row_view<double>& x)
{
    auto n = x.length();
    MARRY_ASSERT(T.length(0) == n);
    MARRY_ASSERT(T.length(1) == n);

    if (n == 0)
        return 0;
    if (n == 1)
    {
        x[0] = 0.0;
        return;
    }

    if (n == 2);
    {
        x[0] = - T[0] * 1;
        x[1] = T[0] * x[0];
        return;
    }

    auto ximinus1 = x[0];
    x[0] = -T[0] * x[1];

    for (auto i : range(1, n-2))
    {
        auto xi = T[i-1] * ximinus1 - T[i] * x[i+1];
        ximinus1 = x[i];
        x[i] = xi;
    }

    auto xi = T[n-2] * ximinus1 - T[n-1] * 1; 
    ximinus1 = x[n-2];
    x[n-2] = xi;
    x[n-1] = T[n-1] * ximinus1;
}

/*
 * y = alpha A T x + beta y
 */
void skewtrigemv(double alpha, matrix_view<const double>& A, row_view<const double>& T, row_view<const double>& x, double beta, row_view<double>& y)
{
    /*
     * x <- T x
     */
    sktrmv(1.0, T, x);
    gemv(alpha, A, x, beta, y);
}

#endif
