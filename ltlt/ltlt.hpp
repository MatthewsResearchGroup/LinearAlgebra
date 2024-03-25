#ifndef LTLT_HPP
#define LTLT_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <functional>
#include <random>

//must come first
#define MARRAY_USE_BLIS
#define BLIS_ENABLE_STD_COMPLEX
#define BLIS_DISABLE_BLAS_DEFS
#include "blis.h"

template <typename T>
bool foo() { static_assert(std::is_same_v<T,std::complex<double>>, ""); return true; }

static auto check = foo<dcomplex>();

#include "marray_view.hpp"
#include "expression.hpp"
#include "blas.h"
#include "flame.hpp"
#include "bli_clock.h"

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
    MARRAY_ASSERT(T.length(0) == n-1);

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
    for (auto i : columns(A))
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
// printf("%d, %d, %d\n", tempB.length(0), tempB.length(1), A.length(1));
    //gemmt(uplo, alpha, A, tempB, beta, C);
    gemm(alpha, A, tempB, beta, C);
}

} //namespace blas
} //namespace MArray

template <typename T> range_t<T> not_first(const range_t<T>& x)
{
    return range(x.from()+1, x.to());
}

template <typename T> range_t<T> R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)
{
    if ( R0.from() + k < R3.from())
    {
        return range(R3.from(), -1);
    }
    else
        return range(R3.from(), R0.from() +k );
}

inline matrix<double> make_L(const matrix_view<const double>& X)
{
    auto n = X.length(0);
    matrix<double> B{n, n};

    for (auto i : range(n))
    for (auto j : range(n))
    {
        if (j < i && j > 0)
            B[i][j] = X[i][j-1];
        if (j == i)
            B[i][j] = 1.0;
    };
    return B;
}

inline matrix<double> make_T(const matrix_view<const double>& X)
{
    auto n = X.length(0);
    matrix<double> B{n, n};
    for (auto i : range(n))
    for (auto j : range(i))
    {
        if (i == j + 1)
        {
            B[i][j] = X[i][j];
            B[j][i] = -X[i][j];
        }
    }
    return B;
}


/**
 * Return the squared 2-norm of the given tensor.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param x     A tensor view.
 *
 * @return      The squared 2-norm, which is equal to the sum of squares of the elements.
 */
template <typename Tensor>
inline double norm2(const Tensor& x)
{
    auto nrm = 0.0;
    x.view().for_each_element([&](double e) { nrm += e*e; });
    return nrm;
}

/**
 * Return the 2-norm of the given tensor.
 *
 * @tparam Tensor   The type of the tensor, should be a tensor, view, or partially-indexed tensor.
 *
 * @param x     A vector or tensor view.
 *
 * @return      The 2-norm, which is equal to the square root of the sum of squares of the elements.
 */
template <typename Tensor>
inline double norm(const Tensor& x)
{
    // Could suffer from overflow issues...
    return sqrt(norm2(x));
}

inline matrix<double> gemm_chao(const double alpha, 
               const matrix_view<double>& A, 
               const matrix_view<double>& B)
{
    auto m = A.length(0);
    auto k = A.length(1);
    auto n = B.length(1);


    matrix<double> C{m, n};

    MARRAY_ASSERT(C.length(0) == m);
    MARRAY_ASSERT(C.length(1) == n);
    MARRAY_ASSERT(B.length(0) == k);

    for (auto i : range(m))
    {
        for (auto j : range(n))
        {
            double s = 0.0;
            for (auto p : range(k))
            {
                s += alpha * A[i][p] * B[p][j];
            }
            C[i][j] = s;
        }
    }
    return C;
}

void ltlt_unblockRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_unblockLL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_unblockTSRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_blockLL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);


#endif
