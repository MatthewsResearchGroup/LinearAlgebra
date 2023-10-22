#include "ltlt.hpp"
#include "unblockLeftLooking.hpp"
#include "unblockRightLooking.hpp"
#include "blockLeftLooking.hpp"
#include "blockRightLooking.hpp"

int main()
{
    /*
     *  testing unblockLeftLooking first. 
     *  
     */

    // build a skew symmtri matrix first.
    // B = A - A.T 
 
    int n; // square matrix size
  
    matrix_view<double> A{n,n};

    // initialize matrix A
    A.for_each_element([&](auto& Aij){ Aij = rng.uniform(0,1); });

    matrix_view<double> B = A - A.T();


    for i : range(n):
        for j : range(n):
            printf("%f", B[i][j]);

        
    return 0;
}
