#include "ltlt.hpp"

void ltlt_unblockLL(const matrix_view<double>& X, len_type k, bool first_column)
{
    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);
    auto n = X.length(0);

    if (k == -1)
        k = n - 1;

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);
    row<double> temp{X.length(0)};

    while(B.size() > n - k - 1)
    {
        // (T  || m  ||   B    )
        // (R0 || r1 || r2 | R3) 4 * 4 partition

        auto [R0, r1, r2, R3] = repartition(T, m, B);
        
        
        if (!R0.empty())
        {    
            auto R0p = first_column ? R0 : not_first(R0);
            temp[R0p] = L[r1][R0p];
            temp[r1] = 1; // L[r1][r1]

            blas::skewtrigemv(-1.0,         L   [r2|R3][R0p|r1],
                                    subdiag(X   [R0p|r1][R0p|r1]),
                                            temp[R0p|r1]/*[r1]*/,
                               1.0,         X   [r2|R3][r1   ]);
        }

        L[R3][r2] = X[R3][r1] / X[r2][r1];

        // (R0 | r1 || r2 | R3 )
        // (T       || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
    }
}

