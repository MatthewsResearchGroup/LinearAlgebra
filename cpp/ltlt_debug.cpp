#include "ltlt.hpp"

#include <iostream>
#include <cstdlib>

#define PRINTMODE 0

std::mt19937_64 gen(5);

static void test_bug(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&)>& LTLT)
{
    auto A = random_matrix(n, n, COLUMN_MAJOR);
    matrix<double> B = A - A.T();
    row<double> t{n-1};
    matrix<double> B0 = B;

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix B before LTLT" << std::endl;
        matrixprint(B);
    }

    LTLT(B, t);

    auto Lm = make_L(B);
    auto Tm = make_T(t);
    auto LmT = Lm.T();

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix Lm " << std::endl;
        matrixprint(Lm);
        std::cout<< "Print Matrix Tm " << std::endl;
        matrixprint(Tm);
    }

    auto B_LTLT = MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix LTLT " << std::endl;
        matrixprint(B_LTLT);
    }

    B0 -= B_LTLT;
    double err = norm(B0) / (n * n);

    if (PRINTMODE)
    {
        std::cout<< "Print Error Matrix " << std::endl;
        matrixprint(B0);
    }
    std::cout << "Norm of Error Matrix : " << err << std::endl;
}

static void test_debug_piv(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&, const row_view<int>&)>& LTLT)
{
    auto A = random_matrix(n, n, COLUMN_MAJOR);
    matrix<double> B = A - A.T();
    row<double> t{n-1};
    row<int> p{n};
    matrix<double> B0 = B;

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix B before LTLT" << std::endl;
        matrixprint(B);
    }

    LTLT(B, t, p);
    pivot_both(B0, p);

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix B0 before pivot_both" << std::endl;
        matrixprint(B0);
    }

    auto Lm = make_L(B);
    auto Tm = make_T(t);
    auto LmT = Lm.T();

    if (PRINTMODE)
    {
        std::cout<< "Print Matrix Lm " << std::endl;
        matrixprint(Lm);
        std::cout<< "Print Matrix Tm " << std::endl;
        matrixprint(Tm);
    }

    auto B_LTLT = MArray::blas::gemm(MArray::blas::gemm(Lm,Tm), LmT);
    if (PRINTMODE)
    {
        std::cout<< "Print Matrix LTLT " << std::endl;
        matrixprint(B_LTLT);
    }

    B0 -= B_LTLT;
    double err = norm(B0) / (n * n);

    if (PRINTMODE)
    {
        std::cout<< "Print Error Matrix " << std::endl;
        matrixprint(B0);
    }
    std::cout << "Norm of Error Matrix : " << err << std::endl;
}

int main()
{
    auto test = [] <int Options>
    {
        for (auto n : {11,24,200})
        for (auto blocksize : {2,3,6,13})
        {
            printf("matrixsize, blocksize = %d, %d\n", n, blocksize);

            printf("Checking for error: unblockedRL\n"); test_bug(n, unblocked(ltlt_unblockRL<Options>));
            printf("Checking for error: unblockedLL\n"); test_bug(n, unblocked(ltlt_unblockLL<Options>));
            if ((n%2) == 1)
            {
                printf("Checking for error: unblockedTSRL\n"); test_bug(n, unblocked(ltlt_unblockTSRL<Options>));
            }
            printf("Checking for error: blockedRL+unbLL\n"); test_bug(n, blocked(ltlt_blockRL<Options>, ltlt_unblockLL<Options>, blocksize));
            printf("Checking for error: blockedRL+unbRL\n"); test_bug(n, blocked(ltlt_blockRL<Options>, ltlt_unblockRL<Options>, blocksize));
            printf("Checking for error: blockedLL+unbLL\n"); test_bug(n, blocked(ltlt_blockLL<Options>, ltlt_unblockLL<Options>, blocksize));
            printf("Checking for error: blockedLL+unbRL\n"); test_bug(n, blocked(ltlt_blockLL<Options>, ltlt_unblockRL<Options>, blocksize));
            printf("Checking for error: pivot+unbRL\n"); test_debug_piv(n, unblocked(ltlt_pivot_unblockRL<Options>));
            printf("Checking for error: pivot+unbLL\n"); test_debug_piv(n, unblocked(ltlt_pivot_unblockLL<Options>));
            printf("Checking for error: pivot+BRL+unbLL\n"); test_debug_piv(n, blocked(ltlt_pivot_blockRL<Options>, ltlt_pivot_unblockLL<Options>, blocksize));

            //timer::print_timers();
        }
    };

    printf("Testing step 0\n");
    test.operator()<STEP_0>();
    printf("Testing step 1\n");
    test.operator()<STEP_1>();
    printf("Testing step 2\n");
    test.operator()<STEP_2>();
    printf("Testing step 3\n");
    test.operator()<STEP_3>();
    printf("Testing step 4\n");
    test.operator()<STEP_4>();
    printf("Testing step 5\n");
    test.operator()<STEP_5>();
}
