#include "ltlt.hpp"

void ltlt_pivot_blockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&, const row_view<int>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    matrix<double> temp_{X.length(0), block_size};

    pi[0] = 0;

    printf("print X at the begining\n\n");
    matrixprint(X);

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
        auto R0p = tail(R0, -1);

        printf("print X before unblock\n");
        matrixprint(X);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], pi[r1|R2|r3], (r1|R2|r3).size(), false);
        
        printf("print X after unblock\n");
        matrixprint(X);

        printf("print L before pivot\n");
        matrixprint(L);

        pivot_rows(L[R2|r3|R4][R0|r1], pi[R2|r3]);

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
