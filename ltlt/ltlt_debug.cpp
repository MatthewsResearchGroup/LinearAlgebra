#include <cstdlib>
#include "test.hpp"

//std::mt19937_64 gen(time(nullptr));
std::mt19937_64 gen(5);
int main(int argc, char* argv[])
{

    auto n = 20;
    auto blocksize = 3;


    //test_bug(n, unblocked(ltlt_unblockRL));
    //test_bug(n, unblocked(ltlt_unblockLL));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
    //test_debug_piv(n, blocked(ltlt_pivot_blockRL, ltlt_pivot_unblockLL, blocksize));
    //test_debug_piv(n, unblocked(ltlt_pivot_unblockLL));
    //test_debug_piv(n, unblocked(ltlt_pivot_unblockRL));
    //timer::print_timers();

    // testing the multiple cores
 

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
    /*
    //auto n = 10;
    auto A = random_matrix(n,n,COLUMN_MAJOR);
    auto t = random_row(n-1);
    auto x = random_row(n);
    auto y = random_row(n);
    auto y_temp = y;

    gemv_sktri(1.0, A, t, x, 1.0, y);
    blas::skewtrigemv(1.0, A, t, x, 1.0, y_temp);
    y -= y_temp;
    printf("print error matrix\n");
    for (auto i : range(y.length()))
        std::cout << y[i] << ", ";
    std::cout << std::endl;
    */

    




    /******** SKR2 ***********/ 
    for (auto n = 100; n <= 1000; n+=100)
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
    auto C = random_matrix(n, n, ROW_MAJOR);
    auto C_copy = C;
    // printf("Print X before SKR2\n");
    // matrixprint(C);
    // printf("Print C_COPY before SKR2\n");
    //matrixprint(C_copy);

    skr2('L', 1.0, a, b, 1.0, C);
    // printf("Print X After SKR2\n");
    // matrixprint(C);

    blas::skr2('L', 1.0, a, b, 1.0, C_copy);
    // printf("Print C_COPY after SKR2\n");
    // matrixprint(C_copy);

    C -= C_copy;
    // printf("printf Error matrix\n");
    // matrixprint(C);
    printf("Norm of Error Matrix : %e\n", norm(C));
    }



    /********* GER2 *********/
    
    // auto m = 20;
    // auto n = 10;

    // auto a = random_row(m);
    // auto b = random_row(n);
    // auto c = random_row(m);
    // auto d = random_row(n);
    // auto E = random_matrix(m,n);

    // auto acopy = a;
    // auto bcopy = b;
    // auto ccopy = c;
    // auto dcopy = d;
    // auto Ecopy = E;

    // double alpha = 1.0;
    // double beta = -1.0;
    // double gamma = 1.0;

    // 
    // blas::ger(      alpha, a, b, gamma, E);
    // blas::ger(      beta,  c, d, gamma, E);

    // ger2(alpha, acopy, bcopy, beta, ccopy, dcopy, gamma, Ecopy);

    // E -= Ecopy;
    // printf("printf Error matrix\n");
    // matrixprint(E);
    // printf("Norm of Error Matrix : %e\n", norm(E));

}
