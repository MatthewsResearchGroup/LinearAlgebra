#include "ltlt.hpp"

void ltlt_unblockLL(const matrix_view<double>& X, len_type k, bool first_column)
{
    PROFILE_FUNCTION
    auto n = X.length(0);
    matrix<double> temp_{n, 1};

    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    if (!first_column)
    {
        // ( T  || m  |    B    )
        // ( R0 || r1 | r2 | R3 )
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        // ( R0 | r1 || r2 | R3 )
        // (    T    || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);

        // Drop the non-existent first column of L
        T = tail(T, -1);
    }

    if (k == -1) k = n;
    auto [B0, B1] = split(B, k-B.front());
    auto& R4 = B1;

    int count = 2;

    while (B0)
    {
        // ( T  || m  |   B0    | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        auto temp = temp_.rebased(1, r1).T();
        temp[r1][R0] = L[r1][R0];
        temp[r1][r1] = 1; // L[r1][r1]
 
        blas::skewtrigemv(-1.0,         L       [     r2|R3|R4][R0|r1],
                                subdiag(X       [R0|r1        ][R0|r1]),
                                        temp.T()[R0|r1        ][   r1],
                           1.0,         X       [     r2|R3|R4][   r1]);



        // gemv_sktri(-1.0,         L       [     r2|R3|R4][R0|r1],
        //                         subdiag(X       [R0|r1        ][R0|r1]),
        //                                 temp.T()[R0|r1        ][   r1],
        //                    1.0,         X       [     r2|R3|R4][   r1]);

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
        count += 1;
    }
}
