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

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#define MARRAY_USE_BLIS 1
#define PROFILE 1

#include "marray_view.hpp"
#include "expression.hpp"
#include "blas.h"
#include "flame.hpp"
#include "timer.h"
#include "omp.h"

enum
{
    FUSED_L3 = 0x1,
    FUSED_L2 = 0x2,
    PARALLEL_L2 = 0x4,
    SEPARATE_T = 0x8,
    BLOCK_RL_VAR1 = 0x10,

    STEP_0 = 0,
    STEP_1 = FUSED_L2,
    STEP_2 = STEP_1 | PARALLEL_L2,
    STEP_3 = STEP_2 | SEPARATE_T,
    STEP_4 = STEP_3 | FUSED_L3,
    STEP_5 = STEP_4 | BLOCK_RL_VAR1,
};

using namespace MArray;
using MArray::slice::all;
using std::tie;

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

inline matrix<double> make_T(const row_view<double>& t)
{
    auto n = t.length(0);
    matrix<double> B{n+1, n+1};
    for (auto i : range(n+1))
    for (auto j : range(i))
    {
        if (i == j + 1)
        {
            B[i][j] = t[j];
            B[j][i] = -t[j];
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

inline auto unblocked(const std::function<void(const matrix_view<double>&, const row_view<double>&, len_type,bool)>& unblock)
{
    return std::bind(unblock, std::placeholders::_1, std::placeholders::_2, -1, false);
}

inline auto unblocked(const std::function<void(const matrix_view<double>&,const row_view<double>&, const row_view<int>&, len_type, bool)>& unblock)
{
    return std::bind(unblock, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, -1, false);
}

template <typename BL>
auto blocked(const BL& block, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& unblock, int blocksize)
{
    return std::bind(block, std::placeholders::_1, std::placeholders::_2, blocksize, unblock);
}

template <typename BL>
auto blocked(const BL& block, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& unblock, int blocksize)
{
    return std::bind(block, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, blocksize, unblock);
}

extern std::mt19937_64 gen;

struct skparams
{
    const void* t;
    inc_t inct;
    dim_t n;
};

void packing
     (
      struc_t strucc, \
     diag_t  diagc, \
     uplo_t  uploc, \
     conj_t  conjc, \
     pack_t  schema, \
     bool    invdiag, \
     dim_t   panel_dim, \
     dim_t   panel_len, \
     dim_t   panel_dim_max, \
     dim_t   panel_len_max, \
     dim_t   panel_dim_off, \
     dim_t   panel_len_off, \
     dim_t   panel_bcast, \
       const double*   kappa, \
       const double*   c, inc_t incc, inc_t ldc, \
             double*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx \

     );

template <typename T=double>
auto random_matrix(int m, int n, MArray::layout layout=MArray::DEFAULT_LAYOUT)
{
    static std::uniform_real_distribution<> dist;
    matrix<T> A({m, n}, layout);

    for (auto i : range(m))
    for (auto j : range(n))
        if constexpr (MArray::detail::is_complex_v<T>)
            A[i][j] = T{dist(gen), dist(gen)};
        else
            A[i][j] = dist(gen);

    return A;
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

inline void rowprint(const row_view<double>& a)
{
    auto n = a.length(0);
    for (auto i : range(n))
    {
        printf("%f, ", a[i]);
    }
    printf("\n");
}

inline std::tuple<int, int> partition(int64_t n, int64_t bs, unsigned nthreads, unsigned idx)
{
    if (nthreads == 1)
        return std::tuple(0, n);

    int start = (idx * n) / nthreads;
    int end = ((idx + 1) * n) / nthreads;

    return std::tie(start, end);
}

inline std::tuple<int, int> partition2(int64_t start, int64_t bs, unsigned idx)
{
    auto begin = start + idx * bs;
    auto end = start + (idx+1) * bs;
    return std::tie(begin, end);
}

template <int Options>
void ltlt_unblockRL(const matrix_view<double>& X, const row_view<double>& t, len_type k = -1, bool first_column = false);

template <int Options>
void ltlt_unblockLL(const matrix_view<double>& X, const row_view<double>& t, len_type k = -1, bool first_column = false);

template <int Options>
void ltlt_unblockTSRL(const matrix_view<double>& X, const row_view<double>& t, len_type k = -1, bool first_column = false);

template <int Options>
void ltlt_blockRL(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);

void ltlt_blockRL_var1(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);

template <int Options>
void ltlt_blockLL(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);

template <int Options>
void ltlt_pivot_unblockLL(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type k = -1, bool first_column = false);

template <int Options>
void ltlt_pivot_blockRL(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& LTLT_UNB);

template <int Options>
void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type k = -1, bool first_column = false);

void ltlt_pivot_blockRL_var1(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& LTLT_UNB);

template <int Options>
void gemm_sktri
     (
       double  alpha, \
       const matrix_view<const double>&  a, \
       const row_view<const double>& d,  \
       const matrix_view<const double>&  b, \
       double  beta, \
       const matrix_view<double>&  c
     );

template <int Options>
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

template <int Options>
void gemv_sktri(double alpha, const matrix_view<const double>& A,\
                                      const row_view   <const double>& T, \
                                      const row_view   <const double>& x,\
                        double beta,  const row_view   <      double>& y);

template <int Options>
void skr2(char uplo,\
          double alpha, const row_view<const double>& a,\
                        const row_view<const double>& b,\
          double beta,  const matrix_view<   double>& C);

template <int Options>
void ger2(double alpha, const row_view<const double> a,\
                        const row_view<const double> b,\
          double beta,  const row_view<const double> c,\
                        const row_view<const double> d,\
          double gamma, const matrix_view<   double> E);
#endif
