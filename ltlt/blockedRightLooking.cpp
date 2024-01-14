#include "ltlt.hpp"

void ltlt_blockRL(const matrix_view<double>& X, len_type block_size, const std::function<void(const matrix_view<double>&,len_type,bool)>& LTLT_UNB)
{
    auto n = X.length(0);
    auto [T, m, B] = partition_rows<DYNAMIC,1,DYNAMIC>(X);

    matrix_view<double> L = false ? X.shifted(1, -1) : X.rebased(1, 1);

    matrix<double> temp{X.length(0), X.length(0)};

    //if (first_column)
    //    blas::skr2('L', 1.0, L[B, m], X[B, m], 1.0, X[B, B]);
    int count  = 0;
    while (B)
    {
        count += 1;
        printf("interation: %d\n", count);
        for (auto i : range(n))
        {
            for (auto j : range(n))
        {
            printf("%f, " , X[i][j]);
        }
            printf("\n");
        }
        // (  T ||  m |       B      )
        // ( R0 || r1 | R2 | r3 | R4 )
        auto [R0, r1, R2, r3, R4] = repartition<DYNAMIC,1>(T, m, B, block_size);
        // printf("r1, r3: %d, %d\n", r1, r3);
        // printf("%d-%d\n", R4.front(), R4.back());
        // printf("R4.first: %d, R4.last: %d\n", R4.first(), R4.last());
        //for (auto i : R4)
        //{
        //    printf("%d,", i);
        //}
        //printf("\n");
        /*right-looking*/

        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2 | r3 | R4],  (r1 | R2).size() + 1, false);
        
        printf("X after unblokced algorithm \n");
        for (auto i : range(n))
        {
            for (auto j : range(n))
        {
            printf("%f, " , X[i][j]);
        }
            printf("\n");
        }
        // print L[r3|R4][R2|r3]
        printf("print L[r3|R4][R2|r3]\n");
        printf("r3|R4 : %d - %d \n", (r3|R4).front(), (r3|R4).back());
        printf("R2|r3 : %d - %d \n", (R2|r3).front(), (R2|r3).back());
        for (auto i : (r3|R4))
        {
        for (auto j : (R2|r3))
        {
            printf("%f, ", L[i][j]);
        }
            printf("\n");
        }
        
 
        temp[r3][R2] = L[r3][R2];
        temp[r3][r3] = 1;
        temp[R4][R2] = L[R4][R2];
        temp[R4][r3] = L[R4][r3];

        printf("print temp[r3|R4][R2|r3]\n");
        for (auto i : (r3|R4))
        {
        for (auto j : (R2|r3))
        {
            printf("%f, ", temp[i][j]);
        }
            printf("\n");
        }

        printf("\n");

        printf("T .......\n");
        for (auto i : (R2|r3))
        {
        for (auto j : (R2|r3))
        {
            printf("%f, ", X[i][j]);
        }
        printf("\n");
        }
        printf("\n");
        printf("print X33 X44 before updating\n");
        for (auto i : (r3|R4))
        {
            for (auto j : (r3|R4))
        {
            printf("%f, " , X[i][j]);
        }
            printf("\n");
        }

        blas::skew_tridiag_rankk('L',
                                 -1.0,      temp[r3|R4][R2|r3],
                                       subdiag(X[R2|r3][R2|r3]),
                                  1.0,         X[r3|R4][r3|R4]);

        printf("X after rankk\n");
        for (auto i : (r3|R4))
        {
            for (auto j : (r3|R4))
        {
            printf("%f, " , X[i][j]);
        }
            printf("\n");
        }
        // X44 += l43 x43^T - x43 l43^T
        // print something to debug
        // printf("print L43\n");
        // printf("r3 : %d, R4: %d, %d\n", r3, R4[0], R4[1]);
        for (auto i = R4.front(); i < R4.back(); i++)
        {
                printf("%f,", L[i][r3]);
        }
        printf("\n");
        blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);

        printf("X44 after updataing\n");
        for (auto i : R4)
        {
            for (auto j : R4)
        {
            printf("%f, " , X[i][j]);
        }
            printf("\n");
        }
        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
        // printf("size of B: %d\n", B[1]);
    }
}
