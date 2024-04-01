#define CATCH_CONFIG_MAIN
#include "../catch2/catch.hpp"

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


// PivBlockRightLooking with UnblockRightLooking algorithm
TEST_CASE("Piv_BRL_UBLL", "[RL], [B], [Piv]")
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

// PivUnBlockRightLooking algorithm
TEST_CASE("Piv_UBRL", "[RL], [UB], [Piv]")
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

// PivUnBlockLeftLooking algorithm
TEST_CASE("Piv_UBLL", "[LL], [UB], [Piv]")
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


// Pivot Rows
TEST_CASE("Piv_Row", "[Piv]")
{
    int N = 10;
    std::mt19937_64 rng;

    std::uniform_real_distribution D;

    row<int> p ({N});
    for(auto i: range(N))
        p[i] = i;
    std::shuffle(p.begin(), p.end(), rng);

    //Matrix of doubles
    matrix<double> A {N, N};
    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            A[i][j] = D(rng);
        }
    }

    row<int> p2({N});
    row<int> idx({N}); 
    for(auto i: range(N))
        idx[i] = i;
    for(auto i: range(N))
    {
        for(auto j: range(i, N))
        {
            if (idx[j] == p[i])
            {
                p2[i] = j;
                std::swap(idx[i], idx[j]);
            }
        }

    }
    
    //create copy of A and permute
    auto Ap = A;
    
    pivot_rows(Ap, p2);

    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            if (std::abs(A[p[i]][j] - Ap[i][j]) > 1e-12)
            {
                printf("error");
            }
        }
    } 
}

//pivot Col
TEST_CASE("Piv_Col", "[Piv]")
{
    int N = 10;
    std::mt19937_64 rng;

    std::uniform_real_distribution D;

    row<int> p ({N});
    for(auto i: range(N))
        p[i] = i;
    std::shuffle(p.begin(), p.end(), rng);

    //Matrix of doubles
    matrix<double> A {N, N};
    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            A[i][j] = D(rng);
        }
    }



    row<int> p2({N});
    row<int> idx({N}); 
    for(auto i: range(N))
        idx[i] = i;
    for(auto i: range(N))
    {
        for(auto j: range(i, N))
        {
            if (idx[j] == p[i])
            {
                p2[i] = j;
                std::swap(idx[i], idx[j]);
            }
        }

    }
    
    //create copy of A and permute
    auto Ap = A;
    
    pivot_rows(Ap, p2);

    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            if (std::abs(A[[i]]p[j] - Ap[i][j]) > 1e-12)
            {
                printf("error");
            }
        }
    } 
}


// Pivot Both
TEST_CASE("Piv_Both", "[Piv]")
{
    int N = 10;
    std::mt19937_64 rng;

    std::uniform_real_distribution D;

    row<int> p ({N});
    for(auto i: range(N))
        p[i] = i;
    std::shuffle(p.begin(), p.end(), rng);

    //Matrix of doubles
    matrix<double> A {N, N};
    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            A[i][j] = D(rng);
        }
    }



    row<int> p2({N});
    row<int> idx({N}); 
    for(auto i: range(N))
        idx[i] = i;
    for(auto i: range(N))
    {
        for(auto j: range(i, N))
        {
            if (idx[j] == p[i])
            {
                p2[i] = j;
                std::swap(idx[i], idx[j]);
            }
        }

    }
    
    //create copy of A and permute
    auto Ap = A;
    
    pivot_rows(Ap, p2);

    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            if (std::abs(A[p[i]]p[j] - Ap[i][j]) > 1e-12)
            {
                printf("error");
            }
        }
    } 
}
