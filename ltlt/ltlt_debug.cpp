#include "test.hpp"

std::mt19937_64 gen(time(nullptr));
int main()
{

    auto n = 10;
    auto blocksize = 3;


    //test_bug(n, unblocked(ltlt_unblockLL));
    test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    //test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);
    //test(n, blocksize, ltlt_blockLL, ltlt_unblockLL);
    //test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);
    
    
}
