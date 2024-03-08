#ifndef _TESTING_HPP_
#define _TESTING_HPP_


#include "ltlt.hpp"

// testing function
inline void test(int n, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    // build the matrix
    matrix <double> A{n, n};

    // initialize matrix A
    std::mt19937_64 rng;
    std::uniform_real_distribution<> uniform;
    A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    // make skew symmetric matrix
    matrix<double> B = A - A.T();
    
    // make a copy of B since we need to overwrite part of B
    matrix<double> B_deepcopy = B;
    
    // starting the decompostion
    // recode the time
    //
    auto starting_point =  bli_clock();
    LTLT_UNB(B, -1, false);
    auto ending_point = bli_clock();

    // calculate the time for decompostion
    auto time = ending_point - starting_point;

    // verify its correctness
    // make L and T from B

    auto Lm = make_L(B);
    auto Tm = make_T(B);
    auto LmT = Lm.T();

    printf("\nPrinting L matrix...\n");
    for (auto i = 0; i < n; i++)
    {
    for (auto j = 0; j < n; j++)
    {
        printf("%f, ", Lm[i][j]);
    }
    printf("\n");
    }

    printf("\nPrinting T matrix...\n");
    for (auto i = 0; i < n; i++)
    {
    for (auto j = 0; j < n; j++)
    {
        printf("%f, ", Tm[i][j]);
    }
    printf("\n");
    }

    // B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    B_deepcopy -= gemm_chao(1.0, gemm_chao(1.0, Lm,Tm), LmT);
    printf("\nPrint the error matrix\n");
    for (auto i = 0; i < n; i++)
    {
    for (auto j = 0; j < n; j++)
    {
        printf("%f, ", B_deepcopy[i][j]);
    }
    printf("\n");
    }
    auto err = norm(B_deepcopy);
    printf("err = %e\n", err);
    MARRAY_ASSERT(err < 1e-12);

    printf("finish successfully in %f second\n", time);
}

// testing function 
inline void test(int n, const std::function<void(const matrix_view<double>&,len_type, const std::function<void(const matrix_view<double>&, len_type, bool)>&)>& LTLT_BLOCK, bool unblockRL = false)
{
    auto blocksize = 20;
    // build the matrix
    matrix <double> A{n, n};

    // initialize matrix A
    std::mt19937_64 rng;
    std::uniform_real_distribution<> uniform;
    A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    // make skew symmetric matrix
    matrix<double> B = A - A.T();
    
    // make a copy of B since we need to overwrite part of B
    matrix<double> B_deepcopy = B;
    
    // starting the decompostion
    // recode the time
    //
    auto starting_point =  bli_clock();
    if (unblockRL == true)
    {
        LTLT_BLOCK(B, blocksize, ltlt_unblockRL);
    }
    else
    {
        LTLT_BLOCK(B, blocksize, ltlt_unblockLL);
    }

    auto ending_point = bli_clock();

    // calculate the time for decompostion
    auto time = ending_point - starting_point;

    // verify its correctness
    // make L and T from B

    auto Lm = make_L(B);
    auto Tm = make_T(B);
    auto LmT = Lm.T();
    // printf("\nPrinting L matrix...\n");
    // printf("Size of L matrix: %d, %d\n", Lm.length(0), Lm.length(1));
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f ", Lm[i][j]);
    // }
    // printf("\n");
    // }

    // printf("\nPrinting T matrix...\n");
    // printf("Size of T matrix: %d, %d\n", Tm.length(0), Tm.length(1));
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f ", Tm[i][j]);
    // }
    // printf("\n");
    // }
    // printf("\nPrinting Lt matrix...\n");
    // printf("Size of LT matrix: %d, %d\n", LmT.length(0), LmT.length(1));
    // printf("Stride of LT matrix: %d, %d\n", LmT.stride(0), LmT.stride(1));
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f ", LmT[i][j]);
    // }
    // printf("\n");
    // }
    // printf("\nPrint origial matrix\n");
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f ", B_deepcopy[i][j]);
    // }
    // printf("\n");
    // }

    auto B_m = gemm_chao(1.0, Lm, Tm);
    auto B_cal = gemm_chao(1.0, B_m, LmT);

    // printf("\nPrint LTLt matrix\n");
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f ", B_cal[i][j]);
    // }
    // printf("\n");
    // }

    // B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    B_deepcopy -= B_cal;
    auto err = norm(B_deepcopy);
    printf("err = %e\n", err);
    printf("\nError matrix...\n");
    for (auto i = 0; i < n; i++)
    {
    for (auto j = 0; j < n; j++)
    {
        // printf("%f ", B_cal[i][j]);
        printf("%f ", B_deepcopy[i][j]);
    }
    printf("\n");
    }
    MARRAY_ASSERT(err < 1e-12 * n * n );
    printf("finish successfully in %f second\n", time);
    
}

#endif
