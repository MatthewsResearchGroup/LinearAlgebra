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
            // printf("\n\nUpdating B part!!!!\n\n");
            // blas::skr2('L', 1.0, L[B][m], X[B][m], 1.0, X[B][B]);
            blas::ger( 1.0, L[B][m], X[B][m], 1.0, X[B][B]);
            blas::ger(-1.0, X[B][m], L[B][m], 1.0, X[B][B]);
        }
        else
        {
            auto Btrunc = R3_trunc(T, B, k);
            blas::ger( 1.0, L[B][m], X[Btrunc][m], 1.0, X[B][Btrunc]);
            blas::ger(-1.0, X[B][m], L[Btrunc][m], 1.0, X[B][Btrunc]);
        }
    }

    if (k == -1)
        k = n - 1;

    // printf("blocked matrix n, k: %d, %d \n", n, k);
    while(B.size() > n - k - 1)
    {
        // (T  || m  |   B    )
        // (R0 || r1 | r2 | R3) 4 * 4 partition

        auto [R0, r1, r2, R3] = repartition(T, m, B);
        printf("R3 = %d - %d\n", R3.from(), R3.to());

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        if ( k == -1)
        {
            // printf("Normal Unblock Rightlooking Algorithm..\n");
            //blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
            blas::ger( 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
            blas::ger(-1.0, X[R3][r2], L[R3][r2], 1.0, X[R3][R3]);

        }
        else
        {
            // printf("Truncated Unblock Rightlooking Algorithm..\n");
            // Since we only get [r1 | R2 | r3 | R4] from the blocked algorithm, 
            // therefore we need to start R0 here, where the begining of R0 here is the r1 in the blocked algorithmms
            auto R3trunc = R3_trunc(R0, R3, k);
            // printf("R3 range: %d, %d\n", R3.from(), R3.to());
            // printf("R3trunc range: (%d, %d)\n", R3trunc.from(), R3trunc.to());
            blas::ger( 1.0, L[R3][r2], X[R3trunc][r2], 1.0, X[R3][R3trunc]);
            blas::ger(-1.0, X[R3][r2], L[R3trunc][r2], 1.0, X[R3][R3trunc]);
        }    

        // printf("\nPrint X after updating:\n");
        // for (auto i : range(n))
        // {
        //     for (auto j : range(n))
        // {
        //     printf("%f, " , X[i][j]);
        // }
        //     printf("\n");
        // }
        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
    }
}
