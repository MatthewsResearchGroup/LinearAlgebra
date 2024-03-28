#include "ltlt.hpp"
#include "test.hpp"
// #include <catch2/catch_test_macros.hpp>
#include <cstdlib>


int main()
{
    /*
     *  testing with different matrix size
     */

    auto n = 10; // square matrix size
    auto blocksize = 3;
                //
    // test(n, ltlt_unblockLL);
    test(n, ltlt_unblockRL);
    // test(n, ltlt_unblockTSRL);
    // test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);
    test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);
    // test(n, blocksize, ltlt_blockLL, ltlt_unblockLL);
    // test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);

    return 0;

    //
    // Poviting 
    //
    //

}
