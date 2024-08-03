#ifndef _TESTING_HPP_
#define _TESTING_HPP_

#include "ltlt.hpp"
#include "../catch2/catch.hpp"

#include <iostream>
#include <iomanip>
#include <limits>

using namespace Catch;

extern std::mt19937_64 gen;
static const auto MAX_MATRIX_SIZE = 500;

inline auto rand_size(int max = MAX_MATRIX_SIZE)
{
    return std::uniform_int_distribution<>(1, max)(gen);
}

inline auto random_permutation(int N)
{
    row<int> p = static_cast<std::vector<int>>(range(N));
    std::shuffle(p.begin(), p.end(), gen);

    row<int> p2{N};
    row<int> idx = static_cast<std::vector<int>>(range(N));
    for (auto i : range(N))
    for (auto j : range(i, N))
    {
        if (idx[j] == p[i])
        {
            p2[i] = j-i;
            std::swap(idx[i], idx[j]);
            break;
        }
    }

    return std::make_pair(p, p2);
}

template <typename T=double>
auto random_matrix(int m, int n, MArray::layout layout=MArray::DEFAULT_LAYOUT)
//auto random_matrix(int m, int n)
{
    static std::uniform_real_distribution<> dist;
    matrix<T> A({m, n}, layout);
    //matrix<T> A({m, n}, COLUMN_MAJOR);
    //matrix<T> A{m, n};

    for (auto i : range(m))
    for (auto j : range(n))
        if constexpr (MArray::detail::is_complex_v<T>)
            A[i][j] = T{dist(gen), dist(gen)};
        else
            A[i][j] = dist(gen);

    return A;
}

/* 
 * This function is for debugging
*/
template <typename T=double>
auto full_matrix_same_num(int m, int n, T k)
{
    static std::uniform_real_distribution<> dist;
    matrix<T> A{m, n};

    for (auto i : range(m))
    for (auto j : range(n))
        A[i][j] =  k; 

    return A;
}

template <typename T=double>
auto random_row(int n)
{
    static std::uniform_real_distribution<> dist;
    row<T> A{n};

    for (auto i : range(n))
        if constexpr (MArray::detail::is_complex_v<T>)
            A[i] = T{dist(gen), dist(gen)};
        else
            A[i] = dist(gen);
    return A;
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

template <typename MArray>
void check_zero(const MArray& X, uplo_t uplo = BLIS_LOWER, struc_t struc = BLIS_GENERAL)
{
    if (struc == BLIS_GENERAL)
    {
        for(auto i : range(X.length(0)))
        for(auto j : range(X.length(1)))
        {
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("E[i][j] = " << X[i][j]);
            CHECK_THAT(std::abs(X[i][j]),  WithinAbs(0, 1e-12));
        }
    }
    else if (uplo == BLIS_LOWER)
    {
        for(auto i : range(X.length(0)))
        for(auto j : range(i))
        {
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("E[i][j] = " << X[i][j]);
            CHECK_THAT(std::abs(X[i][j]),  WithinAbs(0, 1e-12));
        }
    }
    else
    {
        for(auto i : range(X.length(0)))
        for(auto j : range(i+1,X.length(0)))
        {
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("E[i][j] = " << X[i][j]);
            CHECK_THAT(std::abs(X[i][j]),  WithinAbs(0, 1e-12));
        }
    }
}



// The following code is designed for debug
//
//

inline void test_bug(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&)>& LTLT)
{
    auto A = random_matrix(n, n, COLUMN_MAJOR);
    matrix<double> B = A - A.T();
    row<double> t{n-1}; 

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    std::cout<< "Print Matrix B before LTLT" << std::endl;
    matrixprint(B);

    auto starting_point =  bli_clock();
    LTLT(B, t);
    auto ending_point = bli_clock();

    auto time = ending_point - starting_point;

    auto Lm = make_L(B);
    auto Tm = make_T(t);
    auto LmT = Lm.T();

    std::cout<< "Print Matrix Lm " << std::endl;
    matrixprint(Lm);
    std::cout<< "Print Matrix Tm " << std::endl;
    matrixprint(Tm);
    // std::cout<< "Print Matrix LmT " << std::endl;
    // matrixprint(LmT);
    
    std::cout<< "Print Matrix LTLT " << std::endl;
    auto B_LTLT = MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    matrixprint(B_LTLT);

    // calculate the error matrix
    B0 -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    double err = norm(B0) / (n * n);

    
    std::cout<< "Print Error Matrix " << std::endl;
    matrixprint(B0);
    std::cout << "Norm of Error Matrix : " << err << std::endl;
    
}

inline void test_debug_piv(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&, const row_view<int>&)>& LTLT)
{
    auto A = random_matrix(n, n, COLUMN_MAJOR);
    matrix<double> B = A - A.T();
    row<double> t{n-1}; 
    row<int> p{n};

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    std::cout<< "Print Matrix B before LTLT" << std::endl;
    matrixprint(B);

    auto starting_point =  bli_clock();
    LTLT(B, t, p);
    auto ending_point = bli_clock();
    pivot_both(B0, p);

    auto time = ending_point - starting_point;

    auto Lm = make_L(B);
    auto Tm = make_T(t);
    auto LmT = Lm.T();

    std::cout<< "Print Matrix Lm " << std::endl;
    matrixprint(Lm);
    std::cout<< "Print Matrix Tm " << std::endl;
    matrixprint(Tm);
    // std::cout<< "Print Matrix LmT " << std::endl;
    // matrixprint(LmT);
    
    std::cout<< "Print Matrix LTLT " << std::endl;
    auto B_LTLT = MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    matrixprint(B_LTLT);

    // calculate the error matrix
    B0 -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    double err = norm(B0) / (n * n);

    
    std::cout<< "Print Error Matrix " << std::endl;
    matrixprint(B0);
    std::cout << "Norm of Error Matrix : " << err << std::endl;
    
}

inline double performance(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&)>& LTLT, int repitation = 3)
{
    auto MinTime = std::numeric_limits<double>::max();
    //double MinTime = 1.0e4;
    // make skew symmetric matrix
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();
    row<double> t{n};

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    for (auto i : range(repitation))
    {
        auto B = B0;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT(B,t);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;
        printf("Rep and time: %d, %f\n", i, time);

        auto Lm = make_L(B);
        auto Tm = make_T(t);
        auto LmT = Lm.T();

        // calculate the error matrix
        B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
        double err = norm(B_deepcopy) / (n * n);
        printf("err is %f\n", err);
        //check_zero(B0);

        MinTime = (time < MinTime)? time : MinTime;

    }
    return MinTime;

}


template<typename T>
inline bool check_RL(const T& majoralgo)
{
    std::vector<T> RightLooking {"ltlt_unblockRL", "ltlt_blockRL"};
    return std::count(RightLooking.begin(), RightLooking.end(), majoralgo)? true : false;
}

#endif
