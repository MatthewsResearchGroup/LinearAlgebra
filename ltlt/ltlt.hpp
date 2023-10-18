#ifndef LTLT_HPP
#define LTLT_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <functional>

#include "marray_view.hpp"
#include "expression.hpp"
#include "blas.h"
#include "flame.hpp"

using namespace MArray;
using MArray::slice::all;
using std::tie;

namespace MArray
{
namespace blas
{

/*
 * x <- alpha T x
 */
inline void sktrmv(double alpha, const row_view<const double>& T, const row_view<double>& x)
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

/*
 * y = alpha A T x + beta y
 */
inline void skewtrigemv(double alpha, matrix_view<const double>& A, row_view<const double>& T, row_view<const double>& x, double beta, row_view<double>& y)
{
    /*
     * x <- T x
     */
    sktrmv(1.0, T, x);
    gemv(alpha, A, x, beta, y);
}

inline void sktrmm(double alpha, const row_view<const double>& T, const matrix_view<double>& A)
{
    for (auto i : columns(T))
    {
        sktrmv(alpha, T, A[all][i]);
    }
}

/*
* C = alpha A ( T B ) + beta C 
*/
inline void skew_tridiag_gemm(double alpha, const matrix_view<const double>& A,
                                            const row_view   <const double>& T,
                                            const matrix_view<const double>& B,
                              double beta,  const matrix_view<      double>& C)
{
    // B <- T B

    // copy of B
    matrix<double> tempB = B;
    sktrmm(1, T, tempB);
    gemm(alpha, A, tempB, beta, C); 
}

inline void skew_tridiag_rankk(double alpha, const matrix_view<const double>& A,
                                            const row_view   <const double>& T,
                              double beta,  const matrix_view<      double>& C)
{
    // B <- T B

    // copy of B
    matrix<double> tempB = A.T();
    sktrmm(1, T, tempB);
    gemmt(alpha, A, tempB, beta, C); 
}

}
}

#endif
