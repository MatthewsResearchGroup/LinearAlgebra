#include "ltlt.hpp"

void ltlt_unblockTSRL(const matrix_view<double>& X, len_type k, bool first_column)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    auto n = X.length(0);
    MARRAY_ASSERT(n % 2 == 1);
    // MARRAY_ASSERT(n % 2 == 1, "n must be odd");

    if (k == -1)
        k = n - 1;

    // MARRAY_ASSERT(k % 2 == 1, "k must be odd");
    MARRAY_ASSERT(k % 2 == 0);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1,1);
    row<double> temp{X.length(0)};

    if (first_column)
    {
        blas::ger( 1.0, L[B][m], X[B][m], 1.0, X[B][B]);
        blas::ger(-1.0, X[B][m], L[B][m], 1.0, X[B][B]);
       
    }

    while(B.size() > n - k - 1)
    {
        // (T  || m  |     B        )
        // (R0 || r1 | r2 | r3 | R4 )
        //
        auto [R0, r1, r2, r3, R4] = repartition<1, 1>(T, m, B);


        L[r3|R4][r2] = X[r3|R4][r1] / X[r2][r1];

        L[R4][r3] = X[R4][r2] / X[r3][r2];

        X[R4][r3] = X[R4][r3] -   X[r3][r2] * L[r3][r2] * L[R4][r3];

        blas::ger( 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
        blas::ger(-1.0, X[R4][r3], L[R4][r3], 1.0, X[R4][R4]);

        X[R4][r3] = X[R4][r3] + X[r3][r2] * L[R4][r2];

        tie(T, m, B) = continue_with<2>(R0, r1, r2, r3, R4);


    }
}
