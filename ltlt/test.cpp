#include <array>
# include <cstdio>
#include <algorithm>
#include "../marray/marray/flame.hpp"
#include "fwd/marray_fwd.hpp"
#include <random>

using namespace MArray;
int main()
{
    int N = 10;
    std::mt19937_64 rng;

    std::uniform_real_distribution D;

    row<int> p ({N});
    for(auto i: range(N))
        p[i] = i;
    std::shuffle(p.begin(), p.end(), rng);

    //Matrix of doubles
    matrix<double> A {N, N};
    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            A[i][j] = D(rng);
        }
    }



    row<int> p2({N});
    row<int> idx({N}); 
    for(auto i: range(N))
        idx[i] = i;
    for(auto i: range(N))
    {
        for(auto j: range(i, N))
        {
            if (idx[j] == p[i])
            {
                p2[i] = j;
                std::swap(idx[i], idx[j]);
            }
        }

    }
    
    //create copy of A and permute
    auto Ap = A;
    
    pivot_rows(Ap, p2);

    for(auto i: range(N))
    {
        for(auto j: range(N))
        {
            if (std::abs(A[p[i]][j] - Ap[i][j]) > 1e-12)
            {
                printf("error");
            }
        }
    }  

}