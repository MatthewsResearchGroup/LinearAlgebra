#include "ltlt.hpp"

template <int Options>
void ltlt_unblockLL(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column)
{
    PROFILE_FUNCTION
    auto n = X.length(0);
    matrix<double> temp_{n, 1};

    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    if (!first_column)
    {
        // ( T  || m  |    B    )
        // ( R0 || r1 | r2 | R3 )
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        L[R3][r2] = X[R3][r1] / X[r2][r1];
        t[r1] = L[r2][r2];
        if (Options & SEPARATE_T)
            L[r2][r2] = 1;

        // ( R0 | r1 || r2 | R3 )
        // (    T    || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);

        // Drop the non-existent first column of L
        T = tail(T, -1);
    }

    if (k == -1) k = n;
    auto [B0, B1] = split(B, k-B.front());
    auto& R4 = B1;

    while (B0)
    {
        // ( T  || m  |   B0    | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        if (R0)
        {
            if (Options & SEPARATE_T)
            {
                gemv_sktri<Options>(-1.0, L    [r2|R3|R4][R0|r1],
                                          t              [R0],
                                          L.T()          [R0|r1][r1],
                                     1.0, X    [r2|R3|R4]       [r1]);
            }
            else
            {
                X[r2|R3|R4][r1] -= t[R0.back()]*L[r1][R0.back()]*L[r2|R3|R4][r1]
                                 - t[R0.back()]/*L[r1][r1] = 1*/*L[r2|R3|R4][R0.back()];

                if (R0)
                gemv_sktri<Options>(-1.0, L    [r2|R3|R4][R0],
                                          t              [head(R0,-1)],
                                          L.T()          [R0][r1],
                                     1.0, X    [r2|R3|R4]    [r1]);
            }
        }

        PROFILE_SECTION("divide")
        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];
        PROFILE_STOP
        t[r1] = X[r2][r1];
        if (Options & SEPARATE_T)
            L[r2][r2] = 1;

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }
}

template void ltlt_unblockLL<STEP_0>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
template void ltlt_unblockLL<STEP_1>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
template void ltlt_unblockLL<STEP_2>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
template void ltlt_unblockLL<STEP_3>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
template void ltlt_unblockLL<STEP_4>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
template void ltlt_unblockLL<STEP_5>(const matrix_view<double>& X, const row_view<double>& t, len_type k, bool first_column);
