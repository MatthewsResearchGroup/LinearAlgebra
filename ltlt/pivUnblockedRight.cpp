#include "ltlt.hpp"

void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type k, bool first_column)
{
    auto n = X.length(0);
    if (k == -1) k = n-1;

    auto [f, T, m, B] = partition_rows<1, DYNAMIC,  1, DYNAMIC>(X);
    //auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    auto [B0, B1] = split(B, k);
    auto& R4 = B1;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    if (first_column)
    {
        T = f|T;          
    }
    else
    {
        auto pi2 = blas::iamax(X[m|B0|B1][f]);
        pi[m] = pi2;

        pivot_rows(X[m|B0|B1][m], pi2);

        L[B0|B1][m] = X[B0|B1][f] / X[m][f];

        pivot_both(X[m|B0|B1][m|B0|B1], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
    }

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

        pivot_rows(X[r2|R3|R4][r2], pi2);

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

