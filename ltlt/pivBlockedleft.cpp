#include "ltlt.hpp"

void ltlt_pivot_blockLL(const matrix_view<double>& X, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    // auto n = X.length(0);
    // if (k == -1)
        // k = n;

    matrix_view<double> L = false ? X.shifted(1, -1) : X.rebased(1, 1);
    matrix<double> temp{X.length(0), X.length(0)};

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
        auto R0p = not_first(R0);

        /*left-looking*/
        temp[r1][R0p] = L[r1][R0p];
        temp[r1][r1 ] = 1; // L[r1][r1]
        temp[R2][R0p] = L[R2][R0p];
        temp[R2][r1 ] = L[R2][r1 ];
        blas::skew_tridiag_gemm(-1.0,         L   [R2 |r3|R4][R0p|r1],
                                      subdiag(X   [R0p|r1   ][R0p|r1]),
                                              temp[r1 |R2   ][R0p|r1].T(),
                                 1.0,         X   [R2 |r3|R4][r1 |R2]);

        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2 | r3 | R4],  (r1 | R2).size(), true);

        pi[R2 | r3] = X[R2 | r3];

        pivot_rows(L[R2 | r3 | R4][R0 | r1], pi[R2 | r3]);

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
