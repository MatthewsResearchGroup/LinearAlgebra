#include "test.hpp"
#include <cstdlib>

std::mt19937_64 gen(time(nullptr));
int main(int argc, char* argv[])
{

    auto n = 10000;
    auto blocksize = 1000;


    // test_bug(n, unblocked(ltlt_unblockRL));
    // test_bug(n, unblocked(ltlt_unblockLL));
    test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
    // test_bug(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
    // test_bug(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
    timer::print_timers();

    // testing the multiple cores
   
    // auto m = atoi(argv[1]);
    // auto n = atoi(argv[2]);
    // auto k = atoi(argv[3]);
    // // std::cout << "m ,n ,k = " << m << ", " << n << "," << k<< std::endl;
    // auto A = random_matrix(m, k);
    // auto B = random_matrix(k, n);
    // auto starting_point = bli_clock();
    // auto C = MArray::blas::gemm(A,B);
    // auto ending_point = bli_clock();

    // auto time = ending_point - starting_point;
    // //double GFLOPS = 2 * m * n * k / (time * 1e9);
    // double GFLOPS = 2 * double(m) * double(n) * double(k)/(time * 1e9);

    // std::cout << "m, n, k, time, GFLOPS =  " << m << "," << n << ", " << k << ", " << time << ", "<< GFLOPS << std::endl;

}
