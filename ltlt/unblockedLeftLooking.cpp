#include "ltlt.hpp"

void ltlt_unblockLL(const matrix_view<double>& X, len_type k = -1, bool first_column = false)
{
    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);
    auto n = X.length(0);

    if (k == -1)
        k = n;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};


    while(B.size() > n - k)
    {
        // (T  || m  ||   B    )
        // (R0 || r1 || r2 | R3) 4 * 4 partition
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        temp[R0] = L[r1][R0];
        temp[r1] = 1;
        skewtrigemv(-1.0, L[r2 | R3][R0 | r1], subdiag(X[R0 | r1 ][R0, r1]), temp[R0 | r1], 1, X[r2 | R3][r1]);

        L[R3, r2] = X[R3,r1] / X[r2, r1];

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);


    }
}

