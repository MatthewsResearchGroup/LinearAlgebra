#define CATCH_CONFIG_MAIN
#include "../catch2/catch.hpp"

#include <random>

#include "test.hpp"

std::mt19937_64 gen(time(nullptr));

TEST_CASE("UBRL", "[RL], [UB]")
{
    test(rand_size(), unblocked(ltlt_unblockRL));
}

TEST_CASE("UBLL", "[LL], [UB]")
{
    test(rand_size(), unblocked(ltlt_unblockLL));
}

TEST_CASE("UBTSRL", "[RL], [UB]")
{
    // note that n should be odd
    auto n = rand_size();
    if (n%2 == 0) n++;

    test(n, unblocked(ltlt_unblockTSRL));
}

TEST_CASE("BLL_UBLL", "[RL], [B]")
{
    auto n = rand_size();
    auto blocksize = rand_size(10) + 10;

    INFO("blocksize = " << blocksize);

    test(n, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize));
}

TEST_CASE("BLL_UBRL", "[LL], [B]")
{
    auto n = rand_size();
    auto blocksize = rand_size(10) + 10;

    INFO("blocksize = " << blocksize);

    test(n, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize));
}

/*
TEST_CASE("BLL_UBTSRL", "[LL], [B]")
{
    auto n = rand_size();
    // note that blocksize should be odd
    auto blocksize = rand_size(10) + 10;
    if (blocksize%2 == 0) blocksize++;

    test(n, blocked(ltlt_blockLL, ltlt_unblockTSRL, blocksize));
}
*/

TEST_CASE("BRL_UBLL", "[RL], [B]")
{
    auto n = rand_size();
    auto blocksize = rand_size(10) + 10;

    INFO("blocksize = " << blocksize);

    test(n, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize));
}

TEST_CASE("BRL_UBRL", "[RL], [B]")
{
    auto n = rand_size();
    auto blocksize = rand_size(10) + 10;

    INFO("blocksize = " << blocksize);

    test(n, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize));
}

/*
TEST_CASE("BRL_UBTSRL", "[RL], [B]")
{
    auto n = rand_size();
    // note that blocksize should be odd
    auto blocksize = rand_size(10) + 10;
    if (blocksize%2 == 0) blocksize++;

    test(n, blocked(ltlt_blockRL, ltlt_unblockTSRL, blocksize));
}
*/

TEST_CASE("Piv_UBRL", "[RL], [UB], [Piv]")
{
    test_piv(rand_size(), unblocked(ltlt_pivot_unblockRL));
}

TEST_CASE("Piv_UBLL", "[LL], [UB], [Piv]")
{
    test_piv(rand_size(), unblocked(ltlt_pivot_unblockLL));
}

TEST_CASE("Piv_BRL_UBLL", "[RL], [B], [Piv]")
{
    auto n = rand_size();
    auto blocksize = rand_size(10) + 10;

    INFO("blocksize = " << blocksize);

    test_piv(n, blocked(ltlt_pivot_blockRL, ltlt_pivot_unblockLL, blocksize));
}

TEST_CASE("Piv_Row", "[Piv]")
{
    const int N = 10;

    auto [p, p2] = random_permutation(N);
    auto A = random_matrix(N, N);
    auto Ap = A;

    std::cout << p << std::endl;
    std::cout << p2 << std::endl;

    pivot_rows(Ap, p2);

    for(auto i : range(N))
    for(auto j : range(N))
        REQUIRE_THAT(std::abs(A[p[i]][j] - Ap[i][j]),  WithinAbs(0, 1e-12));
}

TEST_CASE("Piv_Col", "[Piv]")
{
    const int N = 10;

    auto [p, p2] = random_permutation(N);
    auto A = random_matrix(N, N);
    auto Ap = A;

    pivot_columns(Ap, p2);

    for(auto i : range(N))
    for(auto j : range(N))
        REQUIRE_THAT(std::abs(A[i][p[j]] - Ap[i][j]),  WithinAbs(0, 1e-12));
}

TEST_CASE("Piv_Both", "[Piv]")
{
    const int N = 100;

    auto [p, p2] = random_permutation(N);

    for (auto struc : {BLIS_GENERAL, BLIS_SYMMETRIC, BLIS_SKEW_SYMMETRIC, BLIS_HERMITIAN, BLIS_SKEW_HERMITIAN})
    for (auto uplo : {BLIS_LOWER, BLIS_UPPER})
    {
        INFO("struc = " << (struc == BLIS_GENERAL        ? "GENERAL" :
                            struc == BLIS_SYMMETRIC      ? "SYMMETRIC" :
                            struc == BLIS_SKEW_SYMMETRIC ? "SKEW_SYMMETRIC" :
                            struc == BLIS_HERMITIAN      ? "HERMITIAN" :
                                                           "SKEW_HERMITIAN"));
        INFO("uplo = " << (uplo == BLIS_LOWER ? "LOWER" : "UPPER"));

        auto A = random_matrix<dcomplex>(N, N);

        switch (struc)
        {
            case BLIS_SYMMETRIC:
                for(auto i : range(N))
                for(auto j : range(i+1,N))
                    A[i][j] = A[j][i];
                break;
            case BLIS_SKEW_SYMMETRIC:
                for(auto i : range(N))
                {
                    for(auto j : range(i+1,N))
                        A[i][j] = -A[j][i];
                    A[i][i] = 0;
                }
                break;
            case BLIS_HERMITIAN:
                for(auto i : range(N))
                {
                    for(auto j : range(i+1,N))
                        A[i][j] = std::conj(A[j][i]);
                    A[i][i].imag(0);
                }
                break;
            case BLIS_SKEW_HERMITIAN:
                for(auto i : range(N))
                {
                    for(auto j : range(i+1,N))
                        A[i][j] = -std::conj(A[j][i]);
                    A[i][i].real(0);
                }
                break;
            default: break;
        }

        auto Ap = A;

        pivot_both(Ap, p2, uplo, struc);

        auto Ap2 = A;
        for(auto i : range(N))
        for(auto j : range(N))
            Ap2[i][j] = A[p[i]][p[j]];

        for(auto i : range(N))
        for(auto j : range(N))
            Ap[i][j] -= A[p[i]][p[j]];

        check_zero(Ap, uplo, struc);
    }
}

TEST_CASE("SKR2", "[Level2]")
{
    const int N = 10;
    
}
