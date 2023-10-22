#include "ltlt.hpp"

#include <random>

int main()
{
    /*
     *  testing unblockLeftLooking first.
     *
     */

    // build a skew symmtri matrix first.
    // B = A - A.T

    auto n = 100; // square matrix size

    matrix<double> A{n, n};

    // initialize matrix A
    std::mt19937_64 rng;
    std::uniform_real_distribution<> uniform;
    A.for_each_element([&](auto& Aij){ Aij = uniform(rng); });

    matrix<double> B = A - A.T();

    for (auto i : range(n))
        for (auto j : range(n))
            printf("%f\n", B[i][j]);

    return 0;
}
