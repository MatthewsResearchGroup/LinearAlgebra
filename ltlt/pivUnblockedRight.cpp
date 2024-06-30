#include "ltlt.hpp"

void ltlt_pivot_unblockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type k, bool first_column)
{
    auto n = X.length(0);

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    auto [T, m, B] = partition_rows<DYNAMIC, 1, DYNAMIC>(X);
    printf("T = %d - %d, m = %d, B = %d - %d\n", T.from(), T.to(), m, B.from(), B.to());

    if (!first_column)
    {
        // ( T  || m  |    B    )
        // ( R0 || r1 | r2 | R3 )
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        auto pi2 = blas::iamax(X[r2|R3][r1]);
        pi[r2] = pi2;
        printf("pi2 is %d\n", pi2);

        pivot_rows(X[r2|R3][r1], pi2);
        std::cout << "----------------Here is the first intetation before pivot_rows------------------" << std::endl;
        printf("r2|R3 = %d - %d, r1 = %d\n", (r2|R3).front(), (r2|R3).end(), r1);
        matrixprint(X);

        L[R3][r2] = X[R3][r1] / X[r2][r1];
        std::cout << "----------------Here is the first intetation after X[R3][r1] / X[r2][r1]------------------" << std::endl;
        matrixprint(X);

        pivot_both(X[r2|R3][r2|R3], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
        std::cout << "----------------Here is the first intetation pivot both--------------------" << std::endl;
        matrixprint(X);

        // ( R0 | r1 || r2 | R3 )
        // (    T    || m  | B  )
        tie(T, m, B) = continue_with(R0, r1, r2, R3);
        printf("T = %d - %d, m = %d, B = %d - %d\n", T.from(), T.to(), m, B.from(), B.to());

        // Drop the non-existent first column of L
        T = tail(T, -1);
        printf("T = %d - %d, m = %d, B = %d - %d\n", T.from(), T.to(), m, B.from(), B.to());
    }

    if (k == -1) k = n;
    auto [B0, B1] = split(B, k-B.front()-1);
    printf("k-B.front() -1 = %d \n", k-B.front() -1);
    auto& R4 = B1;
    printf("k = %d,B = %d - %d,  B0 = %d - %d, B1 = %d - %d\n", k, B.from(), B.to(),  B0.from(), B0.to(), B1.from(), B1.to());

    blas::skr2('L', 1.0, L[B0][m], X[B0][m], 1.0, X[B0][B0]);
    blas::ger(      1.0, L[B1][m], X[B0][m], 1.0, X[B1][B0]);
    blas::ger(     -1.0, X[B1][m], L[B0][m], 1.0, X[B1][B0]);
    std::cout << "----------------Here is the first intetation after skr2 and ger--------------------" << std::endl;
    matrixprint(X);

    while (B0)
    {
        // ( T  || m  |   B0    | B1 )
        // ( R0 || r1 | r2 | R3 | R4 )
        std::cout << "----------------UNBLOCK not first interation, before interation------------------" << std::endl;
        matrixprint(X);
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        auto pi2 = blas::iamax(X[r2|R3|R4][r1]);
        pi[r2] = pi2;
        printf("pi2 = %d\n", pi2);

        pivot_rows(X[r2|R3|R4][r1], pi2);
        std::cout << "----------------UNBLOCK not first interation, afte pivot_row X[r2|R3|R4][r1]------------------" << std::endl;
        matrixprint(X);

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];
        std::cout << "----------------UNBLOCK not first interation, after X[R3|R4][r2] updating------------------" << std::endl;
        matrixprint(X);

        pivot_rows(L[r2|R3|R4][R0|r1   ], pi2);
        std::cout << "----------------UNBLOCK not first interation, after L part pivoting------------------" << std::endl;
        matrixprint(X);
        printf("r2|R3|R4 = %d - %d\n", (r2|R3|R4).from(), (r2|R3|R4).to());
        std::cout << "000000000 Print X[r2|R3|R4][r2|R3|R4] 0000000000 " << std::endl;
        matrixprint(X[r2|R3|R4][r2|R3|R4]);
        pivot_both(X[r2|R3|R4][r2|R3|R4], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
        std::cout << "----------------UNBLOCK not first interation, after X part both pivoting------------------" << std::endl;
        matrixprint(X);

        printf("r2 = %d, R3 = %d - %d , R4 = %d - %d", r2, R3.from(), R3.to(), R4.from(), R4.to());
        blas::skr2('L', 1.0, L[R3][r2], X[R3][r2], 1.0, X[R3][R3]);
        std::cout << "----------------Here is the first intetation after skr2 --------------------" << std::endl;
        matrixprint(X);
        blas::ger(      1.0, L[R4][r2], X[R3][r2], 1.0, X[R4][R3]);
        blas::ger(     -1.0, X[R4][r2], L[R3][r2], 1.0, X[R4][R3]);
        std::cout << "----------------Here is the first intetation after ger--------------------" << std::endl;
        matrixprint(X);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }
    // B = B0|B1;
    // printf("T = %d - %d, m = %d, B = %d - %d\n", T.from(), T.to(), m, B.from(), B.to());
    // auto [R0, r1, r2, R3] = repartition(T, m, B);
    // printf("R0 = %d - %d, r1 = %d, r2 = %d,  R3 = %d - %d\n", R0.from(), R0.to(), r1, r2, R3.from(), R3.to());

    // std::cout << "***************** LAST ITERATION OF RIGHT UNBLOCK ALGORTHM ****************"<< std::endl;
    // matrixprint(X);

    // auto pi2 = blas::iamax(X[r2|R3][r1]);
    // pi[r2] = pi2;
    // printf("pi2 = %d\n", pi2);

    // pivot_rows(X[r2|R3][r1], pi2);
    // std::cout << "----------------UNBLOCK not first interation, afte pivot_row X[r2|R3|R4][r1]------------------" << std::endl;
    // std::cout << "&&&&&&&&&&&&&&&&& LAST ITERATION AFTER PIVOT_ROW OF X &&&&&&&&&&&&&&&" << std::endl;
    // matrixprint(X);

    // L[R3][r2] = X[R3][r1] / X[r2][r1];
    // std::cout << "$$$$$$$$$$$$$$$$$  UNBLOCK not first interation, after X[R3|R4][r2] updating $$$$$$$$$$$$" << std::endl;
    // matrixprint(X);

    // pivot_rows(L[r2|R3][R0|r1   ], pi2);
    // std::cout << "@@@@@@@@@@@@@@@ UNBLOCK not first interation, after L part pivoting @@@@@@@@@@@@@@@" << std::endl;
    // matrixprint(X);
    // printf("R0 = %d - %d, r1 = %d, r2 = %d,  R3 = %d - %d\n", R0.from(), R0.to(), r1, r2, R3.from(), R3.to());
    // pivot_both(X[r2|R3][r2|R3], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
    // std::cout << "----------------UNBLOCK not first interation, after X part both pivoting------------------" << std::endl;
    //  matrixprint(X);

}

