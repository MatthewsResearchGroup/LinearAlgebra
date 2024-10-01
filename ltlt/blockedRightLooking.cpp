#include "ltlt.hpp"

void ltlt_blockRL_var0(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB)
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

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], (r1|R2|r3).size(), false);

        //blas::skew_tridiag_rankk('L',
        //                         -1.0,      temp[r3|R4][R2|r3],
        //                               subdiag(X[R2|r3][R2|r3]),
        //                          1.0,         X[r3|R4][r3|R4]);
        
        gemmt_sktri('L',
                    -1.0,      L[r3|R4][R2|r3],
                                         t[R2],
                           L.T()[R2|r3][r3|R4],
                      1.0,     X[r3|R4][r3|R4]);
        //blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
        skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);
        //printf("print X[R4][R4] after skr2\n");
        //matrixprint(X[R4][R4]);
        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}

void ltlt_blockRL_var1(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB)
{
    PROFILE_FUNCTION
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    int first_iter = true;

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        if (!R4) {
            auto tau1 = t[r1];
            LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], (r1|R2|r3).size(), false);
            if (!first_iter) {
                t[r1] = tau1;
            }
        } else {
            auto tau1 = t[r1];
            LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2|r3], (r1|R2|r3|R4.front()).size(), false);
            if (!first_iter) {
                t[r1] = tau1;
            }
            first_iter = false;

            gemmt_sktri('L',
                        -1.0,  L[R4][R2|r3|R4.front()],
                                   t[R2|r3],
                               L.T()[R2|r3|R4.front()][R4],
                          1.0, X[R4][R4]);
        }
        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}

void ltlt_blockRL(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB) {
    return ltlt_blockRL_var1(X, t, block_size, LTLT_UNB);
}
