#ifndef LTLT_HPP
#define LTLT_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <functional>

//must come first
#define MARRAY_USE_BLIS

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
    x[0] = alpha * (-T[0] * x[1]);

    for (auto i : range(1,n-1))
    {
        auto xi = alpha * (T[i-1] * ximinus1 - T[i] * x[i+1]);
        ximinus1 = x[i];
        x[i] = xi;
    }

    x[n-1] = alpha * (T[n-2] * ximinus1);
}

/*
 * y = alpha A T x + beta y
 */
inline void skewtrigemv(double alpha, const matrix_view<const double>& A,
                                      const row_view   <const double>& T,
                                      const row_view   <const double>& x,
                        double beta,  const row_view   <      double>& y)
{
    /*
     * x <- T x
     */
    row<double> tempx = x;
    sktrmv(1.0, T, tempx);
    gemv(alpha, A, tempx, beta, y);
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

inline void skew_tridiag_rankk(char uplo,
                               double alpha, const matrix_view<const double>& A,
                                             const row_view   <const double>& T,
                               double beta,  const matrix_view<      double>& C)
{
    // B <- T B

    // copy of B
    matrix<double> tempB = A.T();
    sktrmm(1, T, tempB);
    gemmt(uplo, alpha, A, tempB, beta, C);
}

} //namespace blas
} //namespace MArray

template <typename T> range_t<T> not_first(const range_t<T>& x)
{
    return range(x.from()+1, x.to());
}

void ltlt_unblockRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_unblockLL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_blockLL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

#endif
