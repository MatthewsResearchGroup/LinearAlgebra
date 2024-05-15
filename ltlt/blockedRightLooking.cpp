#include "ltlt.hpp"

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    PROFILE_FUNCTION
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    // matrix<double> temp_{X.length(0), block_size};
    matrix<double> temp_{X.length(0), X.length(1)};

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], (r1|R2|r3).size(), false);

        printf("\nPrint X after LTLT_UNB\n");
        matrixprint(X);
        auto temp = temp_.rebased(1, R2.front());
        temp[r3][R2] = L[r3][R2];
        temp[r3][r3] = 1; // L[r3][r3]
        temp[R4][R2] = L[R4][R2];
        temp[R4][r3] = L[R4][r3];

        auto temp_T = temp.T();
    
        //blas::skew_tridiag_rankk('L',
        //                         -1.0,      temp[r3|R4][R2|r3],
        //                               subdiag(X[R2|r3][R2|r3]),
        //                          1.0,         X[r3|R4][r3|R4]);
        
        gemmt_sktri('L',
                    -1.0,      temp[r3|R4][R2|r3],
                          subdiag(X[R2|r3][R2|r3]),
                             temp_T[R2|r3][r3|R4],
                      1.0,        X[r3|R4][r3|R4]);
        printf("Print temp[r3|R4][R2|r3]\n");
        matrixprint(temp[r3|R4][R2|r3]);
        printf("subdiag X[R2|r3][R2|r3]\n");
        matrixprint(X[R2|r3][R2|r3]);
        printf("Print temp.T[R2|r3][r3|R4]\n");
        matrixprint(temp_T[R2|r3][r3|R4]);
        printf("\nPrint X after gemmt_sktri\n");
        matrixprint(X);
        PROFILE_SECTION("skr2")
        blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
        PROFILE_FLOPS(2*(r3|R4).size()*(r3|R4).size());
        PROFILE_STOP

        printf("\nPrint X after skr2\n");
        matrixprint(X);

        printf("\n\n\n\n");
        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}
