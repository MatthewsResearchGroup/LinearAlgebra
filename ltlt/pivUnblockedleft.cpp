#include "ltlt.hpp"


void ltlt_pivot_unblockLL(const matrix_view<double>& X, const row_view<int>& pi, len_type k, bool first_column)
{
    auto n = X.length(0);
    matrix<double> temp_{n, 1};

    matrix_view<double> L = first_column ? X.shifted(1, -1) : X.rebased(1, 1);

    auto [T, m, B] = partition_rows<DYNAMIC,  1, DYNAMIC>(X);

    if (!first_column)
    {
        // ( T  || m  |    B    )
        // ( R0 || r1 | r2 | R3 )
        auto [R0, r1, r2, R3] = repartition(T, m, B);

        auto pi2 = blas::iamax(X[r2|R3][r1]);
        pi[r2] = pi2;
        printf("pi2 is %d\n", pi2);

        std::cout << "----------------Here is the first intetation before pivot_rows------------------" << std::endl;
        matrixprint(X);
        pivot_rows(X[r2|R3][r1], pi2);
        std::cout << "----------------Here is the first intetation after pivot_rows------------------" << std::endl;
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

        // Drop the non-existent first column of L
        T = tail(T, -1);
    }

    if (k == -1) k = n;
    MARRAY_ASSERT(pi.length() == k);
    auto [B0, B1] = split(B, k-B.front());
    auto& R4 = B1;

    while (B0)
    {
        std::cout << "----------------UNBLOCK not first interation, before interation------------------" << std::endl;
        matrixprint(X);
        // ( T  || m  ||   B0    | B1 )
        // ( R0 || r1 || r2 | R3 | R4 )
        auto [R0, r1, r2, R3] = repartition(T, m, B0);

        auto temp = temp_.rebased(1, r1);
        temp[R0][r1] = L[r1][R0];
        temp[r1][r1] = 1; // L[r1][r1]

        blas::skewtrigemv(-1.0,         L   [r2|R3|R4][R0|r1],
                                subdiag(X   [R0|r1   ][R0|r1]),
                                        temp[R0|r1   ][   r1],
                           1.0,         X   [r2|R3|R4][   r1]);

        std::cout << "----------------UNBLOCK not first interation, after skewtrigemv------------------" << std::endl;
        matrixprint(X);
        auto pi2 = blas::iamax(X[r2|R3|R4][r1]);
        pi[r2] = pi2;

        printf("pi2 = %d\n", pi2);
            
        pivot_rows(X[r2|R3|R4][r1], pi2);
        std::cout << "----------------UNBLOCK not first interation, after pivot_rows------------------" << std::endl;
        matrixprint(X);

        L[R3|R4][r2] = X[R3|R4][r1] / X[r2][r1];
        std::cout << "----------------UNBLOCK not first interation, after X[R3|R4][r2] updating------------------" << std::endl;
        matrixprint(X);

        pivot_rows(L[r2|R3|R4][R0|r1   ], pi2);
        std::cout << "----------------UNBLOCK not first interation, after L part pivoting------------------" << std::endl;
        matrixprint(X);
        pivot_both(X[r2|R3|R4][r2|R3|R4], pi2, BLIS_LOWER, BLIS_SKEW_SYMMETRIC);
        std::cout << "----------------UNBLOCK not first interation, after X part both pivoting------------------" << std::endl;
        matrixprint(X);

        // ( R0 | r1 || r2 | R3 | R4 )
        // (    T    || m  | B0 | B1 )
        tie(T, m, B0) = continue_with(R0, r1, r2, R3);
    }
}


