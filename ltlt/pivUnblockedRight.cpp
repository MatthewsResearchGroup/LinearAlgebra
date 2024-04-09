#include "ltlt.hpp"

void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type k, bool first_column)
{
    auto n = X.length(0);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);

    if (!first_column)
    {
        // ( T  || m  |    B    )
        // ( R0 || r1 | r2 | R3 )
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        auto pi2 = blas::iamax(X[r2|R3][r1]);
        pi[r2] = pi2;

        pivot_rows(X[r2|R3][r1], pi2);

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        pivot_both(X[r2|R3][r2|R3], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);

        // ( R0 | r1 || r2 | R3 )
        // (    T    || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);

        // Drop the non-existent first column of L
        T = tail(T, -1);
    }

    if (k == -1) k = n;
    auto [B0, B1] = split(B, k-B.front());
    auto& R4 = B1;

    blas::skr2('L', 1.0, L[B0][m], X[B0][m], 1.0, X[B0][B0]);
    blas::ger(      1.0, L[B1][m], X[B0][m], 1.0, X[B1][B0]);
    blas::ger(     -1.0, X[B1][m], L[B0][m], 1.0, X[B1][B0]);

    while (B0)
    {
        // ( T  || m  |   B0    | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        auto pi2 = blas::iamax(X[r2|R3|R4][r1]);
        pi[r2] = pi2;

        pivot_rows(X[r2|R3|R4][r1], pi2);

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];

        pivot_rows(L[r2|R3|R4][R0|r1   ], pi2);
        pivot_both(X[r2|R3|R4][r2|R3|R4], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);

        blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
        blas::ger(      1.0, L[R4][r2], X[R3][r2], 1.0, X[R4][R3]);
        blas::ger(     -1.0, X[R4][r2], L[R3][r2], 1.0, X[R4][R3]);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }
}

