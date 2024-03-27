#include "ltlt.hpp"

void ltlt_unblockRL(const matrix_view<double>& X, len_type k, bool first_column)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    auto n = X.length(0);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};

    if (first_column)
    {
        if (k == -1)
        {
            printf("\n\nUpdating B part with !!!!\n\n");
            // blas::skr2('L', 1.0, L[B][m], X[B][m], 1.0, X[B][B]);
            blas::ger( 1.0, L[B][m], X[B][m], 1.0, X[B][B]);
            blas::ger(-1.0, X[B][m], L[B][m], 1.0, X[B][B]);
        }
        else
        {
            auto [Btrunc, Brmd] = R3_trunc(T, B, k);
            // update the X[Btrunc]X[Btrunc] part
            blas::skr2('L', 1.0, L[Btrunc][m], X[Btrunc][m], 1.0, X[Btrunc][Btrunc]);
            // update X[Brmd][Btrunc] partion
            blas::ger( 1.0, L[Brmd][m], X[Btrunc][m], 1.0, X[Brmd][Btrunc]);
            blas::ger(-1.0, X[Brmd][m], L[Btrunc][m], 1.0, X[Brmd][Btrunc]);
            // blas::ger( 1.0, L[B][m], X[Btrunc][m], 1.0, X[B][Btrunc]);
            // blas::ger(-1.0, X[B][m], L[Btrunc][m], 1.0, X[B][Btrunc]);
        }
    }

    bool unblock = (k == -1) ? true : false;
    if (k == -1)
        k = n - 1;

    while(B.size() > n - k - 1)
    {
        // (T  || m  |   B    )
        // (R0 || r1 | r2 | R3) 4 * 4 partition

        auto [R0, r1, r2, R3] = repartition(T, m, B);

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        if (unblock)
        {
            // printf("Normal Unblock Rightlooking Algorithm..\n");
            blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);

        }
        else
        {
            // Since we only get [r1 | R2 | r3 | R4] from the blocked algorithm, 
            // therefore we need to start R0 here, where the begining of R0 here is the r1 in the blocked algorithmms
            auto [R3trunc, R3rmd] = R3_trunc(R0, R3, k);
            
            // update the X[R3trunc]X[R3trunc] part
            blas::skr2('L', 1.0, L[R3trunc][r2], X[R3trunc][r2], 1.0, X[R3trunc][R3trunc]);
            // update X[R3rmd][R3trunc] partion
            blas::ger( 1.0, L[R3rmd][r2], X[R3trunc][r2], 1.0, X[R3rmd][R3trunc]);
            blas::ger(-1.0, X[R3rmd][r2], L[R3trunc][r2], 1.0, X[R3rmd][R3trunc]);
        }    

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
    }
}
