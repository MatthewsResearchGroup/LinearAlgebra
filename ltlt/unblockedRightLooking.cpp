#include "ltlt.hpp"

void ltlt_unblockRL(const matrix_view<double>& X, len_type k, bool first_column)
{
    auto n = X.length(0);
    if (k == -1) k = n - 1;
    //k = (k == -1) ? n-1 : k-1;

    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    auto [B0, B1] = split(B, k);
    auto& R4 = B1;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};
    printf("T, m, B, B0, B1 = %d-%d, %d, %d-%d, %d-%d, %d-%d\n\n", T.front(), T.end(), m, B.front(), B.end(), B0.front(), B0.end(), B1.front(), B1.end());

    if (first_column)
    {
        blas::skr2('L', 1.0, L[B0][m], X[B0][m], 1.0, X[B0][B0]);
        blas::ger(      1.0, L[B1][m], X[B0][m], 1.0, X[B1][B0]);
        blas::ger(     -1.0, X[B1][m], L[B0][m], 1.0, X[B1][B0]);
    }

    while (B0)
    {
        // ( T  || m  |    B0   | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);
        printf("r1, R3= %d, %d-%d\n", r1,  R3.front(), R3.end());

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];

        std::cout<< "print X matrix before SKR2 and GER" << std::endl;
        matrixprint(X);
        blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
        blas::ger(      1.0, L[R4][r2], X[R3][r2], 1.0, X[R4][R3]);
        blas::ger(     -1.0, X[R4][r2], L[R3][r2], 1.0, X[R4][R3]);
        std::cout<< "print X matrix after SKR2 and GER" << std::endl;
        matrixprint(X);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
        printf("B0 = %d-%d\n", B0.front(), B0.end());
    }
}
