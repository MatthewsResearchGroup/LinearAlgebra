#include "ltlt.hpp"

void ltlt_pivot_blockRL_var0(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& LTLT_UNB)
{
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);

    pi[0] = 0;

    auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

    LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], pi[r1|R2|r3], (r1|R2|r3).size(), false);

    gemmt_sktri('L',
                -1.0,      L[r3|R4][R2|r3],
                                     t[R2],
                       L.T()[R2|r3][r3|R4],
                  1.0,     X[r3|R4][r3|R4]);

    skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

    tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);

    while (B)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);

        LTLT_UNB(X[r1|R2|r3|R4][r1|R2|r3|R4], t[r1|R2], pi[r1|R2|r3], (r1|R2|r3).size(), false);

        PROFILE_SECTION("pivot_rows_BRL")
        pivot_rows(X[R2|r3|R4][R0], pi[R2|r3|R4]);
        PROFILE_STOP

        gemmt_sktri('L',
                    -1.0,      L[r3|R4][R2|r3],
                                         t[R2],
                           L.T()[R2|r3][r3|R4],
                      1.0,     X[r3|R4][r3|R4]);

        skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
    }
}



void ltlt_pivot_blockRL_var1(const matrix_view<double>& X, const row_view<double>& t, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&,const row_view<double>&,const row_view<int>&,len_type,bool)>& LTLT_UNB)
{

    PROFILE_FUNCTION
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = X.rebased(1, 1);
    int first_iter = false;

    pi[0] = 0;
    // ( T  || m  |    B0   | B1 )
    // ( R0 || r1 | r2 | R3 | R4 )
    auto [R0, r1, r2, R3] = repartition(T, m, B);

    auto pi2 = blas::iamax(X[r2|R3][r1]);
    pi[r2] = pi2;

    PROFILE_SECTION("pivot_row_UBLL")
    pivot_rows(X[r2|R3][r1], pi2);
    PROFILE_STOP
    
    PROFILE_SECTION("divide")
    L[R3][r2] = X[R3][r1] / X[r2][r1];
    PROFILE_STOP
    t[r1] = X[r2][r1];
    L[r2][r2] = 1;

    PROFILE_SECTION("pivot_both_UBLL")
    pivot_both(X[r2|R3][r2|R3], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
    PROFILE_STOP

    while (B.size() > 1)
    {
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, r4, R5] = repartition<DYNAMIC,1,1>(T, m, B, block_size);

        //LTLT_UNB(X[R2|r3|r4|R5][R2|r3|r4|R5], t[R2|r3], (R2|r3|r4).size(), true);
        LTLT_UNB(X[R2|r3|r4|R5][R2|r3|r4|R5], t[R2|r3], pi[R2|r3|r4], (R2|r3|r4).size(), true);


        auto [R2f, R2l] = split(R2, 1);
        PROFILE_SECTION("pivot_rows_BRL")
        if (R2.size() > 0)
        {
            
            pivot_rows(X[R2l|r3|r4|R5][R0], pi[R2l|r3|r4|R5]);  // add r1 if it doesn't work
        }
        else
        {
            pivot_rows(X[r4|R5][R0], pi[r4|R5]);  // add r1 if it doesn't work
        }
        PROFILE_STOP

        gemmt_sktri('L',
                    -1.0,      L[r4|R5][R2|r3|r4],
                                      t[R2|r3   ],
                           L.T()       [R2|r3|r4][r4|R5],
                      1.0,     X[r4|R5]          [r4|R5]);
        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, r4|R5);
    }

}
