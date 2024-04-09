#include "ltlt.hpp"

void ltlt_unblockLL(const matrix_view<double>& X, len_type k, bool first_column)
{
    std::cout << "Here we start the unblock" << std::endl;
    auto n = X.length(0);
    k = (k == -1) ? n-1 : k-1;
    //if (k == -1) k = n-1;

    auto [f, T, m, B] = partition_rows<1, DYNAMIC,  1, DYNAMIC>(X);
    auto [B0, B1] = split(B, k);
    auto& R4 = B1;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    matrix<double> temp_{n, 1};

    if (!first_column)
        L[B0|B1][m] = X[B0|B1][f] / X[m][f];
    else
        T = f|T;

    while (B0)
    {
        // ( T  || m  |   B0    | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        std::cout << "X before skewtirgemn (unblock)" << std::endl;
        matrixprint(X);
        auto [R0, r1, r2, R3] = repartition(T, m, B0);
        printf("T = %d-%d\n", T.front(),T.end());
        printf("r1 = %d\n", r1);

        auto temp = temp_.rebased(1, r1);
        temp[R0][r1] = L[r1][R0];
        temp[r1][r1] = 1; // L[r1][r1]

        blas::skewtrigemv(-1.0,         L   [r2|R3|R4][R0|r1],
                                subdiag(X   [R0|r1   ][R0|r1]),
                                        temp[R0|r1   ][   r1],
                           1.0,         X   [r2|R3|R4][   r1]);

        std::cout << "X after skewtirgemn (unblock)" << std::endl;
        matrixprint(X);
        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];
        std::cout << "X after X/L" << std::endl;
        matrixprint(X);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }
}
