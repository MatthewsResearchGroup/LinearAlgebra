#include "ltlt.hpp"

void ltlt_unblockTSRL(const matrix_view<double>& X, len_type k, bool first_column)
{
    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);

    auto n = X.length(0);
    MARRAY_ASSERT(n % 2 == 1);
    MARRAY_ASSERT(k == -1);
    MARRAY_ASSERT(first_column == false);

    matrix_view<double> L = X.rebased(1,1);
    row<double> temp{X.length(0)};

    while (B)
    {
        // ( T  || m  |      B       )
        // ( R0 || r1 | r2 | r3 | R4 )
        auto [R0, r1, r2, r3, R4] = repartition<1, 1>(T, m, B);

        L[r3|R4][r2] = X[r3|R4][r1] / X[r2][r1];
        L[   R4][r3] = X[   R4][r2] / X[r3][r2];

        X[R4][r3] -= X[r3][r2] * L[r3][r2] * L[R4][r3];

        //blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
        skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        X[R4][r3] += X[r3][r2] * L[R4][r2];

        // ( R0 | r1 | r2 || r3 | R4 )
        // (      T       || m  | B  )
        tie(T, m, B) = continue_with<2>(R0, r1, r2, r3, R4);
    }
}
