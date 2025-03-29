#include "ltlt.hpp"

template <int Options>
void ltlt_blockRL(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB)
{
    if (Options & BLOCK_RL_VAR1)
    {
        ltlt_blockRL_var1(X, t, block_size, LTLT_UNB);
        return;
    }

    PROFILE_FUNCTION
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], (r1|R2|r3).size(), false);

        if (Options & SEPARATE_T)
        {
            gemmt_sktri<Options>('L', -1.0, L    [r3|R4][R2|r3],
                                            t           [R2],
                                            L.T()       [R2|r3][r3|R4],
                                       1.0, X    [r3|R4]       [r3|R4]);
        }
        else
        {
            X[R4][r3] -= t[R2.back()]*L[r3][R2.back()]*L[R4][r3]
                       - t[R2.back()]/*L[r3][r3] = 1*/*L[R4][R2.back()];

            if (R2)
            gemv_sktri<Options>(-1.0, L    [R4][R2],
                                      t        [head(R2, -1)],
                                      L.T()    [R2][r3],
                                 1.0, X    [R4]    [r3]);

            gemmt_sktri<Options>('L', -1.0, L    [R4][R2|r3],
                                            t        [R2],
                                            L.T()    [R2|r3][R4],
                                       1.0, X    [R4]       [R4]);
        }

        skr2<Options>('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}

/*template <int Options = STEP_5>*/
void ltlt_blockRL_var1(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB)
{
    PROFILE_FUNCTION
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);

    // ( T  || m  |    B    )
    // ( R0 || r1 | r2 | R3 )
    auto [R0, r1, r2, R3] = repartition(T, m, B);
    PROFILE_SECTION("divide")
    L[R3][r2] = X[R3][r1] / X[r2][r1];
    PROFILE_STOP
    t[r1] = X[r2][r1];
    L[r2][r2] = 1;

    while (B.size() > 1)
    {
        // (  T ||  m |         B         )
        // ( R0 || r1 | R2 | r3 | r4 | R5 )
        auto [R0, r1, R2, r3, r4, R5] = repartition<DYNAMIC,1,1>(T, m, B, block_size);

        LTLT_UNB(X[R2|r3|r4|R5][R2|r3|r4|R5], t[R2|r3], (R2|r3|r4).size(), true);

        gemmt_sktri<STEP_5>('L',
                            -1.0,     L[r4|R5][R2|r3|r4],
                                      t       [R2|r3   ],
                                  L.T()       [R2|r3|r4][r4|R5],
                             1.0,     X[r4|R5]          [r4|R5]);

        // ( R0 | r1 | R2 || r3 | r4 | R4 )
        // (      T       ||  m |    B    )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, r4|R5);
    }
}

template void ltlt_blockRL<STEP_0>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
template void ltlt_blockRL<STEP_1>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
template void ltlt_blockRL<STEP_2>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
template void ltlt_blockRL<STEP_3>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
template void ltlt_blockRL<STEP_4>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
template void ltlt_blockRL<STEP_5>(const matrix_view<double>& X, const row_view<double>& t, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,len_type,bool)>& LTLT_UNB);
