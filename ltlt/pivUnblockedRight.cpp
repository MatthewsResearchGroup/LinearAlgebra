#include "ltlt.hpp"

void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type k, bool first_column, bool first_row)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    auto n = X.length(0);
    
    if (k == -1)
        k = n;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};

    if (first_column)
        blas::skr2('L', 1.0, L[B][m], X[B][m], 1.0, X[B][B]);

    while(B.size() > n - k)
    {
        // (T  || m  |   B    )
        // (R0 || r1 | r2 | R3) 4 * 4 partition

        auto [R0, r1, r2, R3] = repartition(T, m, B);

        auto pi2 = blas::iamax(X[r2|R3][r1]);
        pi[r2] = pi2;

        pivot_rows(X[r2|R3][r2], pi2);

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        pivot_rows(L[r2|R3][R0|r1], pi2);
        pivot_both(X[r2|R3][r2|R3], pi2, BLIS_GENERAL);

        blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
    }
}
