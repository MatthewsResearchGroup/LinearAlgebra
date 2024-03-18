#include "ltlt.hpp"
#include "test.hpp"

// template <class T>
// void print(T& X)
// {
//     auto n = X.length(0);
//     for (auto i : range(n))
//     {
//         for (auto j : range(n))
//         {
//             printf("%e,", X[i][j]);
//         }
//         printf("\n");
//     }
// }

int main()
{
    /*
     *  testing with different matrix size
     *
     */
    auto n = 11; // square matrix size
                //
    // test(n, ltlt_unblockLL);
    //test(n, ltlt_unblockRL);
    test(n, ltlt_unblockTSRL);
    // test(n, ltlt_blockRL, false);
    // test(n, ltlt_blockRL, true);
    // test(n, ltlt_blockLL, false);
    // test(n, ltlt_blockLL, true);
    // for (auto n = 50; n <= 2000; n = n + 50)
    //     test(n, ltlt_unblockLL);

    // matrix<double> A{n, n};

    // // initialize matrix A
    // std::mt19937_64 rng;
    // std::uniform_real_distribution<> uniform;
    // A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    // matrix<double> B = A - A.T();
    // matrix<double> B_o = B;
    // printf("B\n");
    // // print(B);
    // printf("----------------------------\n");
    // // ltlt_unblockLL(B);
    // ltlt_blockRL(B, 4, ltlt_unblockRL);
    // printf("B after ltlt\n");
    // // print(B);
    // printf("----------------------------\n");
    // auto M = make_L(B);
    // printf("L\n");
    // // print(M);
    // printf("----------------------------\n");
    // auto MT = M.T(); 
    // printf("L.T()\n");
    // // print(MT);
    // printf("----------------------------\n");
    // auto T = make_T(B);
    // printf("T\n");
    // // print(T);
    // printf("----------------------------\n");
    // auto Y = MArray::blas::gemm(MArray::blas::gemm(M,T), MT);
    // printf("Y\n");
    // // print(Y);
    // printf("----------------------------\n");

    // B_o -= Y;
    // auto err = norm(B_o);
    // MARRAY_ASSERT(err > 1e-12);
    // // printf("The norm of error matrix: %e", err);

    return 0;

    // 
    // timing
    // bli_clock.h   bli_clock
    // count the FLOPs of different algorithms
    //

    //
    //
    // testing
    //
    // (A - L @ T @ L.T()) v
    //


    // 
    //
    // Poviting 
    //
    //

    //
    // two step right-looking algorithm
    //
    //
}
