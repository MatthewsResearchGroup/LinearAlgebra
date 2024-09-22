#include "ltlt.hpp"

void ltlt_blockLL(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB)
{
    PROFILE_FUNCTION
    auto [f, T, m, B] = partition_rows<1, DYNAMIC, 1, DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    matrix<double> temp_{block_size, X.length(0)};

    L[B][m] = X[B][f] / X[m][f];
    t[f] = L[m][m];
    L[m][m] = 1;

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        //blas::skew_tridiag_gemm(-1.0,       L    [     R2|r3|R4][R0|r1   ],
        //                       t                   [R0|r1  ],
        //                       L.T()[R0|r1        ][   r1|R2], 
        //            1.0,   X        [     R2|r3|R4][   r1|R2]);

        gemm_sktri(-1.0,       L    [     R2|r3|R4][R0|r1   ],
                               t                   [R0|r1  ],
                               L.T()[R0|r1        ][   r1|R2], 
                    1.0,   X        [     R2|r3|R4][   r1|R2]);
                
        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], (r1|R2|r3).size(), true);

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
