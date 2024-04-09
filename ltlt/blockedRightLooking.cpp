#include "ltlt.hpp"

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    matrix<double> temp_{X.length(0), block_size};

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], (r1|R2|r3).size(), false);

        auto temp = temp_.rebased(1, R2.front());
        temp[r3][R2] = L[r3][R2];
        temp[r3][r3] = 1; // L[r3][r3]
        temp[R4][R2] = L[R4][R2];
        temp[R4][r3] = L[R4][r3];

        blas::skew_tridiag_rankk('L',
                                 -1.0,      temp[r3|R4][R2|r3],
                                       subdiag(X[R2|r3][R2|r3]),
                                  1.0,         X[r3|R4][r3|R4]);

        blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
