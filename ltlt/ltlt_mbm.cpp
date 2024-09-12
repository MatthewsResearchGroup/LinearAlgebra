#if 0
#include <cstdlib>
#include <map>
#include <math.h>
#include <limits>

#include "test.hpp"
#include "docopt.h"
#include "io.hpp"


// Benchmark for different functions
std::mt19937_64 gen(time(nullptr));

int main(int argc, char* argv[])
{
    if (argc > 2)
    {
        printf("Worng number of arg %d\n", argc);
        exit(1);
    }


    std::string funcname = std::string(argv[1]);

    int repeated_times = 3;


    if (funcname == "gemv-sktri")
    {
        // benchmark for gemv-sktri function with different BS
        for ( auto matrixsize = 100; matrixsize <= 20100; matrixsize += 200 )
        {
          //int n = 1000;
          //auto A = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          auto A = random_matrix(matrixsize, matrixsize, COLUMN_MAJOR);
          // auto B = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          // auto T = make_T(B);
          // auto t = subdiag(T);
          auto t = random_row(matrixsize-1);
          auto x = random_row(matrixsize);
          auto y = random_row(matrixsize);
          //auto A = random_matrix(matrixsize, n, ROW_MAJOR);
          //auto t = random_row(n-1);
          //auto x = random_row(n);
          //auto y = random_row(matrixsize);


          auto min_time = std::numeric_limits<float>::max();
          for (auto i = 0; i < repeated_times; i++)
          {

            auto starting_point =  bli_clock();
            gemv_sktri(-1.0, A,
                             t,
                             x,
                       1.0,  y);
            auto ending_point = bli_clock();
            auto time = ending_point - starting_point;
            if (time < min_time)
                min_time = time;
          }
          auto GFLOPS = 2*(matrixsize* matrixsize)/(min_time *1e9);
          printf("gemv-sktri: matrixsize, time,  GFLOPS = %ld, %f s, %f gflops/sec\n", matrixsize, min_time, GFLOPS);

          // timer::print_timers();
        }
    }
    else if (funcname == "skr2")
    {
        // benchmark for skr2 function with different BS
        for (auto matrixsize = 100; matrixsize <= 20100; matrixsize+=200)
        {
          //auto C = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          auto C = random_matrix(matrixsize, matrixsize, COLUMN_MAJOR);
          auto x = random_row(matrixsize);
          auto y = random_row(matrixsize);
          auto min_time = std::numeric_limits<float>::max();
          
          for (auto i = 0; i < repeated_times; i++)
          {
            auto starting_point =  bli_clock();
            skr2('L', 1.0, x, y, 1.0, C);
            auto ending_point = bli_clock();
            auto time = ending_point - starting_point;
            if (time < min_time)
                min_time = time;
          }
          auto GFLOPS = 2*pow(matrixsize,2)/((min_time)*1e9);
          //auto GFLOPS = 2*pow(matrixsize,2)/((ending_point-starting_point)*1e9);
          printf("skr2: matrixsize, time, GFLOPS = %ld, %f s, %f gflops/sec\n", matrixsize, min_time, GFLOPS);

          //timer::print_timers();
        }
    }
    else if (funcname == "skr2-nonunit")
    {
        for (auto matrixsize = 100; matrixsize <= 20100; matrixsize+=200)
        {
          auto stride = 2;
          //auto C = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          auto C = random_matrix(matrixsize, matrixsize, COLUMN_MAJOR);
          auto X = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto Y = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto x = X[0][all];
          auto y = Y[0][all];
          
          auto min_time = std::numeric_limits<float>::max();

          for (auto i = 0; i < repeated_times; i++)
          {
            auto starting_point =  bli_clock();
            skr2('L', 1.0, x, y, 1.0, C);
            auto ending_point = bli_clock();
            auto time = ending_point - starting_point;
            if (time < min_time)
                min_time = time;
          }
          auto GFLOPS = 2*pow(matrixsize,2)/((min_time)*1e9);
          //auto GFLOPS = 2*pow(matrixsize,2)/((ending_point-starting_point)*1e9);
          printf("skr2: matrixsize, time, GFLOPS = %ld, %f s, %f gflops/sec\n", matrixsize, min_time, GFLOPS);

          //timer::print_timers();
        }
    }
    else if (funcname == "ger2")
    {
        // benchmark for ger2 function with different BS
        for (auto matrixsize = 100; matrixsize <= 20100; matrixsize+=200)
        {
          //auto E = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          auto E = random_matrix(matrixsize, matrixsize, COLUMN_MAJOR);
          auto a = random_row(matrixsize);
          auto b = random_row(matrixsize);
          auto c = random_row(matrixsize);
          auto d = random_row(matrixsize);

          auto se0 = E.stride(0);
          auto se1 = E.stride(1);
          //printf("se0 = %d, se1 = %d\n", se0, se1);
          
          auto min_time = std::numeric_limits<float>::max();

          for (auto i = 0; i < repeated_times; i++)
          {
            auto starting_point =  bli_clock();
            ger2(1.0, a, b, -1.0, c, d, 1.0, E);
            auto ending_point = bli_clock();
            auto time = ending_point - starting_point;
            if (time < min_time)
                min_time = time;
          }
          auto GFLOPS = 4*pow(matrixsize,2)/(min_time * 1e9);
          printf("ger: matrixsize, time, GFLOPS = %ld, %f s,  %f gflops/sec\n", matrixsize, min_time, GFLOPS);

          //timer::print_timers();
        }
    }
    else if (funcname == "ger2-nonunit")
    {
        for (auto matrixsize = 100; matrixsize <= 20100; matrixsize+=200)
        {
          auto stride = 2;
          //auto E = random_matrix(matrixsize, matrixsize, ROW_MAJOR);
          auto E = random_matrix(matrixsize, matrixsize, COLUMN_MAJOR);
          auto A = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto B = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto C = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto D = random_matrix(stride, matrixsize, COLUMN_MAJOR);
          auto a = A[0][all];
          auto b = B[0][all];
          auto c = C[0][all];
          auto d = D[0][all];

          auto min_time = std::numeric_limits<float>::max();
          for (auto i = 0; i < repeated_times; i++)
          {
            auto starting_point =  bli_clock();
            ger2(1.0, a, b, -1.0, c, d, 1.0, E);
            auto ending_point = bli_clock();
            auto time = ending_point - starting_point;
            if (time < min_time)
                min_time = time;
          }
          auto GFLOPS = 4*pow(matrixsize,2)/(min_time * 1e9);
          printf("ger: matrixsize, time, GFLOPS = %ld, %f s,  %f gflops/sec\n", matrixsize, min_time, GFLOPS);

        }
    }
    //timer::print_timers();
    return 0;
}
#endif
