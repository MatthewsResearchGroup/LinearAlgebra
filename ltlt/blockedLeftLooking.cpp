#include "ltlt.hpp"

void ltlt_blockLL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    printf("Is the code running here 1?\n");
    // auto n = X.length(0);
    // if (k == -1)
        // k = n;

    matrix_view<double> L = false ? X.shifted(1, -1) : X.rebased(1, 1);
    matrix<double> temp{X.length(0), X.length(0)};

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        printf("Before the reparititiion\n");
        printf("T = %d - %d, m = %d, B = %d - %d\n", T.from(), T.to(), m, B.from(), B.to());
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
        printf("After the reparititiion\n");
        printf("R0 = %d - %d, r1 = %d, R2 = %d - %d, r3 = %d, R4 = %d - %d\n\n", R0.from(), R0.to(), r1, R2.from(), R2.to(), r3, R4.from(), R4.to());

        if (!R0.empty())
        {
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
        }
        printf("R0.empty = %d\n", R0.empty());
        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2 | r3 | R4],  (r1 | R2).size(), !R0.empty());

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
