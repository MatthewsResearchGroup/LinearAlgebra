#include "flame.hpp"
#include "fwd/marray_fwd.hpp"
#include "ltlt.hpp"

void ltlt_pivot_blockRL(const matrix_view<double>& X, const row_view<int>& pi, len_type block_size, const std::function<void(const matrix_view<double>&, const row_view<int>&,len_type,bool)>& LTLT_UNB)
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

        /*right-looking*/

        LTLT_UNB(X[r1 | R2 | r3 | R4][r1 | R2 | r3 | R4], pi[R2 | r3], (r1 | R2 | r3 | R4).size() + 1, false);

        // pi[R2 | r3] = X[R2 | r3];

        pivot_rows(L[R2 | r3 | R4][R0 | r1], pi[R2 | r3]);
        

        temp[r3][R2] = L[r3][R2];
        temp[r3][r3] = 1;
        temp[R4][R2] = L[R4][R2];
        temp[R4][r3] = L[R4][r3];


        blas::skew_tridiag_rankk('L',
                                 -1.0,      temp[r3|R4][R2|r3],
                                       subdiag(X[R2|r3][R2|r3]),
                                  1.0,         X[r3|R4][r3|R4]);


        blas::skr2('L', 1.0, L[R4][r3], X[R4][r3], 1.0, X[R4][R4]);


        // ( R0 | r1 | R2 || r3 | R4 )
        // (      T       ||  m |  B )
        tie(T, m, B) = continue_with<2>(R0, r1, R2, r3, R4);
        // printf("size of B: %d\n", B[1]);
    }
}
