#ifndef _TESTING_HPP_
#define _TESTING_HPP_


#include "ltlt.hpp"

#if 0
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

    B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    auto err = norm(B_deepcopy);
    MARRAY_ASSERT(err > 1e-12);

    printf("finish successfully in %f second\n", time);
}

// testing function 
inline void test(int n, const std::function<void(const matrix_view<double>&,len_type, const std::function<void(const matrix_view<double>&, len_type, bool)>&)>& LTLT_BLOCK, bool unblockRL = false)
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
    if (unblockRL == true)
    {
        LTLT_BLOCK(B, 4, ltlt_unblockRL);
    }
    else
    {
        LTLT_BLOCK(B, 4, ltlt_unblockLL);
    }

    auto ending_point = bli_clock();

    // calculate the time for decompostion
    auto time = ending_point - starting_point;

    // verify its correctness
    // make L and T from B

    auto Lm = make_L(B);
    auto Tm = make_T(B);
    auto LmT = Lm.T();

    B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    auto err = norm(B_deepcopy);
    MARRAY_ASSERT(err > 1e-12);

    printf("finish successfully in %f second\n", time);
    
}
#endif

#endif
