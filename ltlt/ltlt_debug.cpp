#include "test.hpp"
#include <cstdlib>

//std::mt19937_64 gen(time(nullptr));
std::mt19937_64 gen(5);
int main(int argc, char* argv[])
{

    auto n = 10;
    auto blocksize = 3;


    // test_bug(n, unblocked(ltlt_unblockRL));
    // test_bug(n, unblocked(ltlt_unblockLL));
    test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
    // test_bug(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
    //timer::print_timers();

    // testing the multiple cores
   

    // auto m = 1;
    // auto n = 1;
    // auto A = random_matrix(m, n);
    // auto B = random_matrix(m, n);
    // auto T = make_T(B);
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
