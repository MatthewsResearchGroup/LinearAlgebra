#include "test.hpp"

std::mt19937_64 gen(time(nullptr));
int main()
{

    auto n = 6;
    auto blocksize = 2;


    test_bug(n, unblocked(ltlt_unblockRL));
    //test_bug(n, unblocked(ltlt_unblockLL));
    // test_bug(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
    // test_bug(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
    // test_bug(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
    //test_bug(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
    
    
}
