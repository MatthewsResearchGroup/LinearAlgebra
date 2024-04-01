#ifndef _TESTING_HPP_
#define _TESTING_HPP_

#include "ltlt.hpp"

// testing function
inline std::tuple<double, double>  test(int n, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
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
    
    // B = B_deepcopy;
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

    // printf("\nPrinting L matrix...\n");
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f, ", Lm[i][j]);
    // }
    // printf("\n");
    // }


    // calculate the error matrix
    B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    // B_deepcopy -= gemm_chao(1.0, gemm_chao(1.0, Lm,Tm), LmT);
    // printf("\nPrint the error matrix\n");
    // for (auto i = 0; i < n; i++)
    // {
    // for (auto j = 0; j < n; j++)
    // {
    //     printf("%f, ", B_deepcopy[i][j]);
    // }
    // printf("\n");
    // }
    double err = norm(B_deepcopy) / (n * n);
    printf("err = %e\n", err);
    MARRAY_ASSERT(err < 1e-12);

    // printf("finish successfully in %f second\n", time);
    return std::make_tuple(err, time);
}

// testing function 
// inline void test(int n, const std::function<void(const matrix_view<double>&,len_type, const std::function<void(const matrix_view<double>&, len_type, bool)>&)>& LTLT_BLOCK, bool unblockRL = false)
inline std::tuple<double, double> test(int n, int blocksize, const std::function<void(const matrix_view<double>&,len_type, const std::function<void(const matrix_view<double>&, len_type, bool)>&)>& LTLT_BLOCK, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
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


    LTLT_BLOCK(B, blocksize, LTLT_UNB);

    // if (unblockRL == true)
    // {
    //     LTLT_BLOCK(B, blocksize, ltlt_unblockRL);
    // }
    // else
    // {
    //     LTLT_BLOCK(B, blocksize, ltlt_unblockLL);
    // }

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


    B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    // B_deepcopy -= B_cal;
    auto err = norm(B_deepcopy) / (n * n);
    printf("err = %e\n", err);
    MARRAY_ASSERT(err < 1e-12);
    // printf("finish successfully in %f second\n", time);
    return std::make_tuple(err, time);
    
}


namespace performance
{
inline std::tuple<std::vector<double>, std::vector<double>> test(int n, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB, int repitation = 3)
{
    std::vector<double> time_vec;
    std::vector<double> error_vec;
    // build the matrix
    matrix <double> A{n, n};

    // initialize matrix A
    std::mt19937_64 rng;
    std::uniform_real_distribution<> uniform;
    A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    // make skew symmetric matrix
    matrix<double> B = A - A.T();
    
    // make a copy of B since we need to overwrite part of B
    matrix<double> B_original = B;
    
    // B = B_deepcopy;
    // starting the decompostion
    // recode the time
    //
    for (auto i : range(repitation))
    {
        auto B = B_original;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT_UNB(B, -1, false);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;

        auto Lm = make_L(B);
        auto Tm = make_T(B);
        auto LmT = Lm.T();

        // calculate the error matrix
        B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
        double err = norm(B_deepcopy) / (n * n);

        time_vec.push_back(time);
        error_vec.push_back(err);
        // output_to_csv(std::to_string(LTLT_UNB), )

        // wirte the error and time 

    }
    return std::make_tuple(error_vec, time_vec);
}

inline std::tuple<std::vector<double>, std::vector<double>> test(int n, int blocksize, const std::function<void(const matrix_view<double>&,len_type, const std::function<void(const matrix_view<double>&, len_type, bool)>&)>& LTLT_BLOCK, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB, int repitation = 3)
{
    std::vector<double> time_vec;
    std::vector<double> error_vec;
    // build the matrix
    matrix <double> A{n, n};

    // initialize matrix A
    std::mt19937_64 rng;
    std::uniform_real_distribution<> uniform;
    A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    // make skew symmetric matrix
    matrix<double> B_original = A - A.T();
    
    // starting the decompostion
    for (auto i : range(repitation))
    {
        auto B = B_original;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT_BLOCK(B, blocksize, LTLT_UNB);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;

        auto Lm = make_L(B);
        auto Tm = make_T(B);
        auto LmT = Lm.T();

        // calculate the error matrix
        B_deepcopy -= MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
        double err = norm(B_deepcopy) / (n * n);

        time_vec.push_back(time);
        error_vec.push_back(err);

        // wirte the error and time 
        
    }
    return std::make_tuple(error_vec, time_vec);
}

} // end namespace performance


#endif
