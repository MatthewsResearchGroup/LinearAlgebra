#include <cstdlib>
#include <map>
#include <math.h>

#include "test.hpp"
#include "docopt.h"
#include "io.hpp"

// Benchmark for different functions
std::mt19937_64 gen(time(nullptr));

int main()
{
   // benchmark for gemv-sktri function with different BS
   for (auto matrixsize = 100; matrixsize <= 5100; matrixsize+=200)
   {
     //auto matrixsize = 2100;
     auto A = random_matrix(matrixsize,matrixsize);
     auto B = random_matrix(matrixsize,matrixsize);
     auto T = make_T(B);
     auto t = subdiag(T);
     auto x = random_row(matrixsize);
     auto y = random_row(matrixsize);

     auto starting_point =  bli_clock();
     gemv_sktri(-1.0, A,
                      t,
                      x,
                1.0,  y);
     auto ending_point = bli_clock();
     auto GFLOPS = 2*pow(matrixsize,2)/((ending_point-starting_point)*3e9);
     printf("matrixsize, GFLOPS = %ld,  %f\n", matrixsize, GFLOPS);

      timer::print_timers();
   }


   // benchmark for skr2 function with different BS
   /*
   for (auto matrixsize = 100; matrixsize <= 5100; matrixsize+=200)
   {
     auto C = random_matrix(matrixsize,matrixsize);
     auto x = random_row(matrixsize);
     auto y = random_row(matrixsize);

     auto starting_point =  bli_clock();
     skr2('L', 1.0, x, y, 1.0, C);
     auto ending_point = bli_clock();
     auto GFLOPS = 2*pow(matrixsize,2)/((ending_point-starting_point)*3e9);
     printf("matrixsize, GFLOPS = %ld,  %f\n", matrixsize, GFLOPS);

     timer::print_timers();
   }
   */

   // benchmark for ger2 function with different BS
   /*
   for (auto matrixsize = 100; matrixsize <= 5100; matrixsize+=200)
   {
     auto E = random_matrix(matrixsize,matrixsize);
     auto a = random_row(matrixsize);
     auto b = random_row(matrixsize);
     auto c = random_row(matrixsize);
     auto d = random_row(matrixsize);

     auto starting_point =  bli_clock();
     ger2(1.0, a, b, -1.0, c, d, 1.0, E);
     auto ending_point = bli_clock();
     auto GFLOPS = 4*pow(matrixsize,2)/((ending_point-starting_point)*3e9);
     printf("matrixsize, GFLOPS = %ld,  %f\n", matrixsize, GFLOPS);

     timer::print_timers();
   }
   */
}
