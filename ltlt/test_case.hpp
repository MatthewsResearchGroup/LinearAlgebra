#define CATCH_CONFIG_MAIN
#include "test.hpp"
#include <cstdlib>
#include <time.h>


TEST_CASE("UBRL", "[RL], [UB]")
{
    // unblockedrightlooking algorithm 
    //
    // generate matrix size n
    
    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1); 

    test(n, ltlt_unblockRL);

    printf("Matrix size is %d, and test has passed!\n", n);

}

// UnBlockLeftLooking algorithm
TEST_CASE("UBLL", "[LL], [UB]")
{
    // unblockedrightlooking algorithm 
    //
    // generate matrix size n

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1); 
    // Call unblockedrightlooking algorithm
    //
    
    test(n, ltlt_unblockLL);
    printf("Matrix size is %d, and test has passed!\n", n);
}


// UnBlockTwoStepsRightLooking algorithm
TEST_CASE("UBTSRL", "[RL], [UB]")
{

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    // note that n should be odd
    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1) + 1; 
    // Call unblockedrightlooking algorithm
    //
    
    test(n, ltlt_unblockTSRL);
    printf("Matrix size is %d, and test has passed!\n", n);
}


// BlockLeftLooking with UnblockLeftLooking algorithm
TEST_CASE("BLL_UBLL", "[RL], [B]")
{

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    // note that n should be odd
    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1); 

    auto blocksize = (std::rand() % 20) + 1;
    // Call unblockedrightlooking algorithm
    //
    test(n, blocksize, ltlt_blockLL, ltlt_unblockLL);
    printf("Matrix size is %d, blocksize is %d, and test has passed!\n", n, blocksize);
}


// BlockLeftLooking with UnblockRightLooking algorithm
TEST_CASE("BLL_UBRL", "[LL], [B]")
{

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    // note that n should be odd
    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1) ; 

    auto blocksize = (std::rand() % 20) + 1;
    // Call blockedLeftLooking algorithm
    //
    test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);
    printf("Matrix size is %d, blocksize is %d, and test has passed!\n", n, blocksize);
}


// BlockRightLooking with UnblockLeftLookingalgorithm
TEST_CASE("BRL_UBLL", "[RL], [B]")
{

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    // note that n should be odd
    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1); 

    auto blocksize = (std::rand() % 20) + 1;
    // Call blockedRightlooking algorithm
    //
    test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);
    printf("Matrix size is %d, blocksize is %d, and test has passed!\n", n, blocksize);
}


// BlockRightLooking with UnblockRightLooking algorithm
TEST_CASE("BRL_UBRL", "[RL], [B]")
{

    std::srand( time( NULL ) );
    int32_t MAX_MATRIX_SIZE = 500;   

    // note that n should be odd
    auto n = 50 * (std::rand() % (MAX_MATRIX_SIZE/50) + 1); 

    auto blocksize = (std::rand() % 20) + 1;
    // Call blockedRightlooking algorithm
    //
    test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);
    printf("Matrix size is %d, blocksize is %d, and test has passed!\n", n, blocksize);
}
