#include <cstdlib>
#include "test.hpp"

//std::mt19937_64 gen(time(nullptr));
std::mt19937_64 gen(5);
int main(int argc, char* argv[])
{
    /*
    auto X0 = random_matrix(20000,20000);

    for (auto n : range(20000,20001,1000))
    {
        matrix<double> X = X0[range(n)][range(n)];
        row<double> T{n-1};
        row<int> p{n};

        for (auto r : range(2))
            ltlt_blockRL<STEP_0>(X, T, 256, ltlt_unblockLL<STEP_0>);
    }

    timer::print_timers();
    timer::clear_timers();

    for (auto n : range(20000,20001,1000))
    {
        matrix<double> X = X0[range(n)][range(n)];
        row<double> T{n-1};
        row<int> p{n};

        for (auto r : range(2))
            ltlt_blockRL<STEP_1>(X, T, 256, ltlt_unblockLL<STEP_1>);
    }

    timer::print_timers();

    return 0;
    */

    auto test = [] <int Options>
    {
        for (auto n : {11,24,200})
        for (auto blocksize : {2,3,6,13})
        {
            printf("matrixsize, blocksize = %d, %d\n", n, blocksize);

            printf("Checking for error: unblockedRL\n"); test_bug(n, unblocked(ltlt_unblockRL<Options>));
            printf("Checking for error: unblockedLL\n"); test_bug(n, unblocked(ltlt_unblockLL<Options>));
            if ((n%2) == 1)
            { printf("Checking for error: unblockedTSRL\n"); test_bug(n, unblocked(ltlt_unblockTSRL<Options>)); }
            printf("Checking for error: blockedRL+unbLL\n"); test_bug(n, blocked(ltlt_blockRL<Options>, ltlt_unblockLL<Options>, blocksize));
            printf("Checking for error: blockedRL+unbRL\n"); test_bug(n, blocked(ltlt_blockRL<Options>, ltlt_unblockRL<Options>, blocksize));
            printf("Checking for error: blockedLL+unbLL\n"); test_bug(n, blocked(ltlt_blockLL<Options>, ltlt_unblockLL<Options>, blocksize));
            printf("Checking for error: blockedLL+unbRL\n"); test_bug(n, blocked(ltlt_blockLL<Options>, ltlt_unblockRL<Options>, blocksize));
            //printf("Checking for error: pivot+unbRL\n"); test_debug_piv(n, unblocked(ltlt_pivot_unblockRL<Options>));
            printf("Checking for error: pivot+unbLL\n"); test_debug_piv(n, unblocked(ltlt_pivot_unblockLL<Options>));
            printf("Checking for error: pivot+BRL+unbLL\n"); test_debug_piv(n, blocked(ltlt_pivot_blockRL<Options>, ltlt_pivot_unblockLL<Options>, blocksize));
            //timer::print_timers();
        }
        // testing the multiple cores
    };

    printf("Testing step 0\n");
    test.operator()<STEP_0>();
    printf("Testing step 1\n");
    test.operator()<STEP_1>();
    printf("Testing step 2\n");
    test.operator()<STEP_2>();
    printf("Testing step 3\n");
    test.operator()<STEP_3>();
    printf("Testing step 4\n");
    test.operator()<STEP_4>();
    printf("Testing step 5\n");
    test.operator()<STEP_5>();

    /*
     *
     * Gemv_sktri function testing
     *
     */

    // // Matrix size
    // auto n = 30;
    //
    // // Generate a raomdom square matrix A
    // auto A = random_matrix(n,n);
    // printf("----------Matrix A------------\n");
    // matrixprint(A);
    // printf("------------------------------\n");
    // auto B = full_matrix_same_num(n,n,1.0);
    // auto T = make_T(B);
    // auto t = subdiag(T);
    // //auto T = random_row(n-1);
    // // printf("----------Matrix T------------\n");
    // // for (auto i : range(t.length()))
    // //     std::cout << t[i] << ", ";
    // // printf("\n------------------------------\n");
    // auto x = random_row(n);
    // printf("----------Vector x------------\n");
    // for (auto i : range(x.length()))
    //     std::cout << x[i] << ", ";
    // printf("\n------------------------------\n");
    // auto y = random_row(n);
    // printf("\n----------Vector y------------\n");
    // for (auto i : range(y.length()))
    //     std::cout << y[i] << ", ";
    // printf("\n------------------------------\n");
    // //auto y = full_matrix_same_num(n,1,0.0);

    // auto y_temp = y;
    // printf("\n----------Vector y_temp------------\n");
    // for (auto i : range(y_temp.length()))
    //     std::cout << y_temp[i] << ", ";
    // printf("\n------------------------------\n");

    // gemv_sktri(1.0, A, t, x, 1.0, y);
    // printf("----------y after gemv_sktri------------\n");
    // for (auto i : range(y.length()))
    //     std::cout << y[i] << ", ";
    // printf("\n------------------------------\n");





    /*
     *  GEMV-SKTRI function
     *
     */
    //auto n = 51;
    //auto A = random_matrix(n,n,COLUMN_MAJOR);
    //auto t = random_row(n-1);
    //auto x = random_row(n);
    //auto y = random_row(n);
    //auto y_temp = y;

    //gemv_sktri(1.0, A, t, x, 1.0, y);
    //blas::skewtrigemv(1.0, A, t, x, 1.0, y_temp);
    //y -= y_temp;
    //printf("print error matrix\n");
    //for (auto i : range(y.length()))
    //    std::cout << y[i] << ", ";
    //std::cout << std::endl;




    /******** SKR2 ***********/
    for (auto layout : {ROW_MAJOR, COLUMN_MAJOR})
    for (auto n = 100; n <= 1000; n+=101)
    {
        auto a = random_row(n);
        // printf("----------Vector a------------\n");
        // for (auto i : range(a.length()))
        //     std::cout << a[i] << ", ";
        // printf("\n\n");
        auto b = random_row(n);
        // printf("----------Vector b------------\n");
        // for (auto i : range(b.length()))
        //     std::cout << b[i] << ", ";
        // printf("\n\n");
        auto C = random_matrix(n, n, layout);
        auto C_copy = C;
        //printf("Print X before SKR2\n");
        //matrixprint(C);
        //printf("Print C_COPY before SKR2\n");
        //matrixprint(C_copy);

        skr2<STEP_5>('L', -1.0, a, b, 2.0, C);
        //printf("Print X After SKR2\n");
        //matrixprint(C);

        blas::skr2('L', -1.0, a, b, 2.0, C_copy);
        //printf("Print C_COPY after SKR2\n");
        //matrixprint(C_copy);

        C -= C_copy;
        for (auto i : range(n))
        for (auto j : range(i,n))
            C[i][j] = 0;
        //printf("printf Error matrix\n");
        //matrixprint(C);
        printf("skr2 [n=%d]: Norm of Error Matrix : %e\n", n, norm(C));
    }




    /********* GER2 *********/

    //for (auto n = 7; n <= 100; n+=3)
    //{
    //auto a = random_row(n);
    //auto b = random_row(n);
    //auto c = random_row(n);
    //auto d = random_row(n);
    //auto E = random_matrix(n,n,ROW_MAJOR);

    //auto acopy = a;
    //auto bcopy = b;
    //auto ccopy = c;
    //auto dcopy = d;
    //auto Ecopy = E;

    //double alpha = 1.0;
    //double beta = -1.0;
    //double gamma = 1.0;

    ////printf("E before ger2\n");
    ////matrixprint(E);
    ////printf("E_Copy before ger2\n");
    ////matrixprint(Ecopy);
    //
    //blas::ger(      alpha, a, b, gamma, E);
    //blas::ger(      beta,  c, d, gamma, E);
    //
    //ger2(alpha, acopy, bcopy, beta, ccopy, dcopy, gamma, Ecopy);

    ////printf("E after ger2\n");
    ////matrixprint(E);
    ////printf("E_Copy after ger2\n");
    ////matrixprint(Ecopy);
    //
    //E -= Ecopy;
    ////printf("printf Error matrix\n");
    ////matrixprint(E);
    //printf("Norm of Error Matrix : %e\n", norm(E));
    //}
    //
    //
    /************** GEMM-sktri ***************/

    // //auto m = 257;
    // //auto n = 257;
    // //auto k = 540;
    // auto m = 500;
    // auto n = 500;
    // auto k = 258;
    // auto A = random_matrix(m,k,COLUMN_MAJOR);
    // auto t = random_row(k-1);
    // //auto A_T = A.T();
    // auto B = random_matrix(k,n,COLUMN_MAJOR);
    // matrix<double> C({m, n}, COLUMN_MAJOR);
    // //auto C = random_matrix(m,m,ROW_MAJOR);
    // auto C_copy = C;

    // //for (auto p = 0; p < 256; p++)
    // //{
    // //for (auto i = 0; i < m; i++)
    // //    A[i][p] = 0.0;

    // //for (auto j = 0; j < n; j++)
    // //    B[p][j] = 0.0;
    // //}
    // //for (auto p = 512; p < 540; p++)
    // //{
    // //for (auto i = 0; i < m; i++)
    // //    A[i][p] = 0.0;

    // //for (auto j = 0; j < n; j++)
    // //    B[p][j] = 0.0;
    // //}
    // printf("Print X before gemm\n");
    // //matrixprint(C);
    // printf("Print C_COPY before gemm\n");
    // //matrixprint(C_copy);
    // blas::skew_tridiag_gemm(-1.0, A, t, B, 1.0, C);
    // gemm_sktri(-1.0, A, t, B, 1.0, C_copy);
    // printf("Print X after gemm\n");
    // //matrixprint(C);
    // printf("Print C_COPY after gemm\n");
    // //matrixprint(C_copy);

    // C -= C_copy;
    // printf("printf Error matrix\n");
    // //matrixprint(C);
    // printf("Norm of Error Matrix : %e\n", norm(C));
}
