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
auto random_matrix(int m, int n)
{
    static std::uniform_real_distribution<> dist;
    matrix<T> A{m, n};

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


inline auto unblocked(const std::function<void(const matrix_view<double>&,len_type,bool)>& unblock)
{
    return std::bind(unblock, std::placeholders::_1, -1, false);
}

inline auto unblocked(const std::function<void(const matrix_view<double>&,const row_view<int>&,len_type,bool)>& unblock)
{
    return std::bind(unblock, std::placeholders::_1, std::placeholders::_2, -1, false);
}

template <typename BL>
auto blocked(const BL& block, const std::function<void(const matrix_view<double>&,len_type,bool)>& unblock, int blocksize)
{
    return std::bind(block, std::placeholders::_1, blocksize, unblock);
}

template <typename BL>
auto blocked(const BL& block, const std::function<void(const matrix_view<double>&,const row_view<int>&,len_type,bool)>& unblock, int blocksize)
{
    return std::bind(block, std::placeholders::_1, std::placeholders::_2, blocksize, unblock);
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

inline void test(int n, const std::function<void(const matrix_view<double>&)>& LTLT)
{
    // make skew symmetric matrix
    n = 5;

    INFO("n = " << n);

    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    LTLT(B);

    auto B1 = B0;
    ltlt_unblockLL(B1);
    auto L1 = make_L(B1);
    auto T1 = make_T(B1);

    // verify its correctness
    // make L and T from B

    auto Lm = make_L(B);
    auto Tm = make_T(B);

    std::cout << std::fixed << std::setprecision(10);

    INFO("L:\n" << Lm);
    INFO("L (exact):\n" << L1);
    INFO("T:\n" << Tm);
    INFO("T (exact):\n" << T1);
    INFO("B:\n" << B0);
    INFO("LTLT:\n" << MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), Lm.T()));

    // calculate the error matrix
    B0 -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), Lm.T());

    std::cout << "E:" << std::endl << B0 << std::endl;

    //check_zero(B0);
}

inline void test_piv(int n, const std::function<void(const matrix_view<double>&,const row_view<int>&)>& LTLT)
{
    n = 5;

    // make skew symmetric matrix
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    auto B1 = B0;
    row<int> p1{n};
    ltlt_pivot_unblockLL(B1, p1);
    auto L1 = make_L(B1);
    auto T1 = make_T(B1);

    row<int> p{n};
    LTLT(B, p);
    pivot_both(B0, p);

    // verify its correctness
    // make L and T from B

    auto Lm = make_L(B);
    auto Tm = make_T(B);
    auto LmT = Lm.T();

    std::cout << std::fixed << std::setprecision(10);

    INFO("L:\n" << Lm);
    INFO("L (exact):\n" << L1);
    INFO("T:\n" << Tm);
    INFO("T (exact):\n" << T1);
    INFO("p:\n" << p);
    INFO("p (exact):\n" << p1);
    INFO("B:\n" << B0);
    INFO("LTLT:\n" << MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), Lm.T()));

    // calculate the error matrix
    B0 -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    check_zero(B0);
}


inline double performance(int n, const std::function<void(const matrix_view<double>&)>& LTLT, int repitation = 3)
{
    auto MinTime = std::numeric_limits<double>::max();
    //double MinTime = 1.0e4;
    // make skew symmetric matrix
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    for (auto i : range(repitation))
    {
        auto B = B0;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT(B);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;
        printf("Rep and time: %d, %f\n", i, time);

        auto Lm = make_L(B);
        auto Tm = make_T(B);
        auto LmT = Lm.T();

        // calculate the error matrix
        B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
        double err = norm(B_deepcopy) / (n * n);
        //check_zero(B0);

        MinTime = (time < MinTime)? time : MinTime;

    }
    return MinTime;

}

// The following code is designed for debug
//
//




inline void test_bug(int n, const std::function<void(const matrix_view<double>&)>& LTLT)
{
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();

    // make a copy of B since we need to overwrite part of B
    matrix<double> B0 = B;

    std::cout<< "Print Matrix B before LTLT" << std::endl;
    matrixprint(B);

    auto starting_point =  bli_clock();
    LTLT(B);
    auto ending_point = bli_clock();

    auto time = ending_point - starting_point;

    auto Lm = make_L(B);
    auto Tm = make_T(B);
    auto LmT = Lm.T();

    // std::cout<< "Print Matrix Lm " << std::endl;
    // matrixprint(Lm);
    // std::cout<< "Print Matrix Tm " << std::endl;
    // matrixprint(Tm);
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

template<typename T>
inline bool check_RL(const T& majoralgo)
{
    std::vector<T> RightLooking {"ltlt_unblockRL", "ltlt_blockRL"};
    return std::count(RightLooking.begin(), RightLooking.end(), majoralgo)? true : false;
}

#endif
