#include <cstdlib>
#include "test.hpp"

//std::mt19937_64 gen(time(nullptr));
std::mt19937_64 gen(5);
int main(int argc, char* argv[])
{

    auto n = 6;
    auto blocksize = 2;


    // test_bug(n, unblocked(ltlt_unblockRL));
    //test_bug(n, unblocked(ltlt_unblockLL));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
    // test_piv(n, blocked(ltlt_pivot_blockRL, ltlt_pivot_unblockLL, blocksize));
    test_debug_piv(n, blocked(ltlt_pivot_blockRL, ltlt_pivot_unblockLL, blocksize));
    //test_debug_piv(n, unblocked(ltlt_pivot_unblockLL));
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




    // // using orginal gemv as verification

    // blas::skewtrigemv(1.0, A, t, x, 1.0, y_temp);
    // printf("----------y_temp after gemv_sktri------------\n");
    // for (auto i : range(y_temp.length()))
    //     std::cout << y_temp[i] << ", ";
    // printf("\n------------------------------\n");
    // auto m = 10;
    // auto n = 10;
    // auto A = random_matrix(m, n);
    // //auto A = full_matrix_same_num(m,n,1.0);
    // //auto B = full_matrix_same_num(n,n,1.0);
    // auto B = random_matrix(n, n);
    // matrixprint(A);
    // auto T = make_T(B);
    // matrixprint(T);
    // auto x = random_row(n);
    // auto x_temp = x;
    // auto y = random_row(m);
    // auto y_temp = y;
    // std::cout << "Print x"<< std::endl;
    // for (auto i : range(x.length()))
    //     std::cout << x[i] << ", ";
    // std::cout << std::endl;
    // std::cout << "Print X-temp"<< std::endl;
    // for (auto i : range(x.length()))
    //     std::cout << x_temp[i] << ", ";
    // std::cout << std::endl;
    // // std::cout << "Print y"<< std::endl;
    // // for (auto i : range(y.length()))
    // //     std::cout << y[i] << ", ";
    // // std::cout << std::endl;
    // gemv_sktri(-1.0, A,
    //                  subdiag(T),
    //                  x,
    //             1.0, y);
    // blas::skewtrigemv(-1.0, A,
    //                  subdiag(T),
    //                  x_temp,
    //             1.0, y_temp);
    // std::cout << "Print x"<< std::endl;
    // for (auto i : range(x.length()))
    //     std::cout << x[i] << ", ";
    // std::cout << std::endl;
    // std::cout << "Print X-temp"<< std::endl;
    // for (auto i : range(x_temp.length()))
    //     std::cout << x_temp[i] << ", ";
    // std::cout << std::endl;
    // std::cout << "Print y"<< std::endl;
    // for (auto i : range(y.length()))
    //     std::cout << y[i] << ", ";
    // std::cout << std::endl;
    // std::cout << "Print y-temp"<< std::endl;
    // for (auto i : range(y_temp.length()))
    //     std::cout << y_temp[i] << ", ";
    // std::cout << std::endl;

    // std::cout << "Print x"<< std::endl;
    // for (auto i : range(x.length()))
    //     std::cout << x[i] << ", ";
    // std::cout << std::endl;
    // std::cout << "Print y"<< std::endl;
    // for (auto i : range(y.length()))
    //     std::cout << y[i] << ", ";
    // std::cout << std::endl;
    // auto A_T = A.T();
    // // auto A_T = random_matrix(m, n);
    // auto C = full_matrix_same_num(m,n,0.0);
    // auto D = full_matrix_same_num(m,n,0.0);
    // auto M = full_matrix_same_num(m,n,0.0);
    // //auto A = full_matrix_same_num(m,n,2.0);
    // //auto T = full_matrix_same_num(m,n,3.0);
    // //auto A_T = full_matrix_same_num(m,n,1.0);
    // //auto B = full_matrix_same_num(m,n,0.0);
    // //auto C = full_matrix_same_num(m,n,2.0);
    // //auto C_T = full_matrix_same_num(m,n,3.0);
    // //auto D = full_matrix_same_num(m,n,0.0);

    // std::cout << "Print A" << std::endl;
    // matrixprint(A);
    // std::cout << "Print T" << std::endl;
    // matrixprint(T);
    // std::cout << "Print A.T" << std::endl;
    // matrixprint(A_T);

    // 
    // auto t = subdiag(B);
    // 
    // gemmt_sktri('L',
    //             -1.0,         A,
    //                           t,
    //                         A_T,
    //             1.0,          C);

    // // blas::skew_tridiag_gemm(
    // //             -1.0,         A,
    // //                           t,
    // //                         A_T,
    // //             1.0,          C);
    // 
    // 
    // blas::skew_tridiag_rankk('L',
    //                          -1.0,      A,
    //                                     t,
    //                           1.0,      D);
    // 
    // // gemm_sktri(
    // //             -1.0,      A,
    // //                        t,
    // //                      A_T,
    // //               1.0,     D);


    // M -= MArray::blas::gemm(MArray::blas::gemm(A,T), A_T);


    // printf("\n\n Print C\n");
    // matrixprint(C);
    // printf("\n\n");
    // printf("\n\n Print D\n");
    // matrixprint(D);
    // printf("\n\n");
    // printf("\n\n Print M\n");
    // matrixprint(M);
    // printf("\n\n");
    // C -= D ;
    // // printf("\n\n Print Error\n");
    // matrixprint(C);
    // printf("Norm of Error Matrix : %e\n", norm(C));
    // //std::cout << "norm of Error matrix : " << norm(C) << std::endl;
    // // matrixprint(B);

}
