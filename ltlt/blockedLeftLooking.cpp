#include "ltlt.hpp"

void ltlt_blockLL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    PROFILE_FUNCTION
    auto [f, T, m, B] = partition_rows<1, DYNAMIC, 1, DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    matrix<double> temp_{block_size, X.length(0)};

    L[B][m] = X[B][f] / X[m][f];

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        auto temp = temp_.rebased(0, r1);
        temp[r1][R0] = L[r1][R0];
        temp[r1][r1] = 1; // L[r1][r1]
        temp[R2][R0] = L[R2][R0];
        temp[R2][r1] = L[R2][r1];

        printf("R0 = %d - %d , r1 = %d, R2 = %d - %d\n", R0.from(), R0.to(), r1, R2.from(), R2.to());
        printf("\nprint X before gemm_sktri\n");
        matrixprint(X);
        printf("L        [     R2|r3|R4][R0|r1   ]\n");
        matrixprint(L        [     R2|r3|R4][R0|r1   ]);
        //printf("t\n");
        //rowprint(subdiag(X       [R0|r1        ][R0|r1   ]));
        printf("temp.T()[R0|r1        ][   r1|R2]\n");
        matrixprint(temp.T()[R0|r1        ][   r1|R2]);
        //blas::skew_tridiag_gemm(-1.0,         L       [     R2|r3|R4][R0|r1   ],
        //                              subdiag(X       [R0|r1        ][R0|r1   ]),
        //                                      temp.T()[R0|r1        ][   r1|R2],
        //                       1.0,         X       [     R2|r3|R4][   r1|R2]);
        
        gemm_sktri(-1.0,   L        [     R2|r3|R4][R0|r1   ],
                    subdiag(X       [R0|r1        ][R0|r1   ]),
                            temp.T()[R0|r1        ][   r1|R2], 
                    1.0,   X        [     R2|r3|R4][   r1|R2]);
        printf("print X after gemm_sktri\n");
        matrixprint(X);
                

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], (r1|R2|r3).size(), true);

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
