#ifndef LTLT_HPP
#define LTLT_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <functional>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>

//must come first
#define MARRAY_USE_BLIS
#define BLIS_ENABLE_STD_COMPLEX
#define BLIS_DISABLE_BLAS_DEFS
#include "blis.h"

#include "marray_view.hpp"
#include "expression.hpp"
#include "blas.h"
#include "flame.hpp"
#include "bli_clock.h"
#include "timer.h"
#include "omp.h"

template <typename T>                                                                                                                                    
bool foo() { static_assert(std::is_same_v<T,int>, ""); return true; }

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
    PROFILE_FUNCTION
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
    PROFILE_FLOPS(3*n);
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
    PROFILE_FUNCTION
    row<double> tempx = x;
    // printf("print A inside of skewtrigemv\n\n");
    // for (auto i : range(A.length(0)))
    // {
    // for (auto j : range(A.length(1)))
    // {
    //     printf("%f, ", A[i][j]);
    // }
    // printf("\n");
    // }
    // printf("Print T\n");
    // for (auto i : range(T.length()))
    //     printf("%f, ", T[i]);
    // printf("\n");
    // printf("Print x\n");
    // for (auto i : range(x.length()))
    //     printf("%f, ", x[i]);
    // printf("\n");
    // printf("Print y\n");
    // for (auto i : range(y.length()))
    //     printf("%f, ", y[i]);
    // printf("\n");
    sktrmv(1.0, T, tempx);
    gemv(alpha, A, tempx, beta, y);
    PROFILE_FLOPS(2*A.length(0)*A.length(1));
}

inline void sktrmm(double alpha, const row_view<const double>& T, const matrix_view<double>& A)
{
    /*
     * A =alpha T A
     */
    PROFILE_FUNCTION
    for (auto i : columns(A))
    {
        sktrmv(alpha, T, A[all][i]);
    }
    PROFILE_FLOPS(3*A.length(0)*A.length(1));
}

/*
* C = alpha A ( T B ) + beta C
*/
inline void skew_tridiag_gemm(double alpha, const matrix_view<const double>& A,
                                            const row_view   <const double>& T,
                                            const matrix_view<const double>& B,
                              double beta,  const matrix_view<      double>& C)
{
    PROFILE_FUNCTION
    matrix<double> tempB = B;
    sktrmm(1, T, tempB);
    PROFILE_SECTION("gemm")
    gemm(alpha, A, tempB, beta, C);
    PROFILE_STOP
    PROFILE_FLOPS(2*A.length(0)*A.length(1)*tempB.length(1));
}


/**
 * Perform the upper or lower triangular portion of the skew tridiag matrix multiplication \f$ C = \alpha ABA^T + \beta C \f$.
 *
 * @param uplo  'L' if C is lower-triangular or 'U' if C is upper-triangular.
 *
 * @param alpha Scalar factor for the product `ABA^T`.
 *
 * @param A     A `m`x`k`  matrix or matrix view. Must have either a row or
 *              column stride of one.
 *
 * @param B     A `k`x`k` skew matrix or matrix view.
 *
 * @param beta  Scalar factor for the original matrix `C`.
 *
 * @param C     A `m`x`m` skew matrix or matrix view.
 */

inline void skew_tridiag_rankk(char uplo,
                               double alpha, const matrix_view<const double>& A,
                                             const row_view   <const double>& T,
                               double beta,  const matrix_view<      double>& C)
{
    PROFILE_FUNCTION
    matrix<double> tempB = A.T();
    sktrmm(1, T, tempB);
    PROFILE_SECTION("gemmt")
    gemmt(uplo, alpha, A, tempB, beta, C);
    PROFILE_STOP
    PROFILE_FLOPS(A.length(0)*A.length(1)*tempB.length(1));
}

} //namespace blas
} //namespace MArray

template <typename T, typename U> range_t<T> head(const range_t<T>& x, U n)
{
    if (n < 0)
        return x.size() < -n ? range(x.from(), x.from()) : range(x.from(), x.to()+n);
    else
        return x.size() < n ? x : range(x.from(), x.from()+n);
}

template <typename T, typename U> range_t<T> tail(const range_t<T>& x, U n)
{
    if (n < 0)
        return x.size() < -n ? range(x.to(), x.to()) : range(x.from()-n, x.to());
    else
        return x.size() < n ? x : range(x.to()-n, x.to());
}

template <typename T, typename U> std::pair<range_t<T>,range_t<T>> split(const range_t<T>& x, U n)
{
    return std::make_pair(head(x, n), tail(x, n == 0 ? x.size() : -n));
}

// template <typename T> range_t<T> R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)
// {
//     if ( R0.from() + k < R3.from())
//     {
//         return range(R3.from(), -1);
//     }
//     else
//     {
//         return range(R3.from(), R0.from() + k);
//     }
// }




template <typename T> 
inline auto R3_trunc(const range_t<T>& R0, const range_t<T>& R3, len_type k)
{
    if ( R0.from() + k < R3.from())
    {
        return std::make_tuple(range(R3.from(), -1), range(R3.from(), -1));
    }
    else
        return range(R3.from(), R0.from() + k);
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

inline void matrixprint(const matrix_view<double>& B)
{
    auto m = B.length(0);
    auto n = B.length(1);
    auto baserow = B.base(0);
    auto basecol = B.base(1);

    for (auto i : range(m))
    {
        for (auto j : range(n))
        {
            printf("%f ", B[i+baserow][j+basecol]);
        }
        printf("\n");
    }
} 

inline std::tuple<int, int> partition(int64_t n, int64_t bs, unsigned idx, unsigned nthreads)
{
    if (nthreads == 1)
        return std::tuple(0, n);

    int start = (idx * n) / nthreads;
    int end = ((idx + 1) * n) / nthreads;

    return std::tuple(start, end);

}


void ltlt_unblockRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_unblockLL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_unblockTSRL(const matrix_view<double>& X, len_type k = -1, bool first_column = false);

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_blockLL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_pivot_unblockLL(const matrix_view<double>& X, const row_view<int>& pi, len_type k = -1, bool first_column = false);

// void ltlt_pivot_blockLL(const matrix_view<double>& X, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_pivot_blockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&, const row_view<int>&,len_type,bool)>& LTLT_UNB);

void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type k = -1, bool first_column = false);

void gemm_sktri
     (
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c 
     );

void gemmt_sktri
     ( 
       char    uploc, \  
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c  
     );


void gemv_sktri(double alpha, const matrix_view<const double>& A,\
                                      const row_view   <const double>& T, \
                                      const row_view   <const double>& x,\
                        double beta,  const row_view   <      double>& y);
#endif
