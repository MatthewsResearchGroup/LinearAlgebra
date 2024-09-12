#include "ltlt.hpp"

void ltlt_pivot_blockRL(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    matrix<double> temp_{X.length(0), block_size};

    pi[0] = 0;

    auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

    LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], pi[r1|R2|r3], (r1|R2|r3).size(), false);

    gemmt_sktri('L',
                -1.0,      L[r3|R4][R2|r3],
                                  t[R2|r3],
                       L.T()[R2|r3][r3|R4],
                  1.0,     X[r3|R4][r3|R4]);

    blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

    tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], pi[r1|R2|r3], (r1|R2|r3).size(), false);

        pivot_rows(X[R2|r3|R4][R0], pi[R2|r3|R4]);

        gemmt_sktri('L',
                    -1.0,      L[r3|R4][R2|r3],
                                      t[R2|r3],
                           L.T()[R2|r3][r3|R4],
                      1.0,     X[r3|R4][r3|R4]);

        blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
