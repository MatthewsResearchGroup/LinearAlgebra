#include "ltlt.hpp"

void ltlt_unblockRL(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column)
{
    PROFILE_FUNCTION
    auto n = X.length(0);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);

    if (k == -1) k = n;
    auto [B0, B1] = split(B, k-B.front()-1);
    auto& R4 = B1;

    if (first_column)
    {
        skr2('L', 1.0, L[B0][m], X[B0][m], 1.0, X[B0][B0]);
        //blas::skr2('L', 1.0, L[B0][m], X[B0][m], 1.0, X[B0][B0]);
        ger2(1.0, L[B1][m], X[B0][m], -1.0, X[B1][m], L[B0][m], 1.0, X[B1][B0]);
        //blas::ger(      1.0, L[B1][m], X[B0][m], 1.0, X[B1][B0]);
        //blas::ger(     -1.0, X[B1][m], L[B0][m], 1.0, X[B1][B0]);

    }

    while (B0)
    {
        // ( T  || m  |    B0   | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];
        t[r1] = L[r2][r2];
        L[r2][r2] = 1;

        skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
        //blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
        ger2(1.0, L[R4][r2], X[R3][r2], -1.0, X[R4][r2], L[R3][r2], 1.0, X[R4][R3]);
        //blas::ger(      1.0, L[R4][r2], X[R3][r2], 1.0, X[R4][R3]);
        //blas::ger(     -1.0, X[R4][r2], L[R3][r2], 1.0, X[R4][R3]);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }

    // ( T  || m  |    B    )
    // ( R0 || r1 | r2 | R3 )
    B = B0|B1;
    auto [R0, r1, r2, R3] = repartition(T, m, B);

    L[R3][r2] = X[R3][r1] / X[r2][r1];
    t[r1] = L[r2][r2];
    L[r2][r2] = 1;
}
