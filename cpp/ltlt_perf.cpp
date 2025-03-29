#include <cstdlib>
#include <map>
#include <math.h>
#include <filesystem>

#include "ltlt.hpp"
#include "docopt.h"

std::mt19937_64 gen(time(nullptr));

static void output_to_csv(const int& nt,
                          const int& MatrixSize,
                          const std::string& MajorAlgo,
                          const std::string& MinorAlgo,
                          const int& BlockSize,
                          const double& time,
                          const double& GFLOPS)
{
    auto filename = MinorAlgo.empty() ? "./time.csv" : "./time_" + std::to_string(BlockSize) + ".csv";
    auto out_csv = fopen(filename.c_str(),"a+");

    if (!out_csv)
    {
        printf("\nERROR: could not open %s for output\n",filename.c_str());
        exit(1);
    }

    if (std::filesystem::is_empty(filename))
        fprintf(out_csv,"%s\n", "NUM_THREADS, MatrixSize, MajorAlgo, MinorAlgo, BlockSize, Time, GFLOPS");

    fprintf(out_csv, "%6d, %6d, %s, %s, %6d, %E, %E\n",  nt,
    MatrixSize,         MajorAlgo.c_str(),
    MinorAlgo.c_str(),  BlockSize,
    time,               GFLOPS);

    fclose(out_csv);
}

static double performance(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&)>& LTLT, int repitation = 3)
{
    auto MinTime = std::numeric_limits<double>::max();
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();
    matrix<double> B0 = B;

    for (auto i : range(repitation))
    {
        row<double> t{n-1};
        auto B = B0;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT(B,t);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;
        printf("Rep and time: %d, %f\n", i, time);

        MinTime = (time < MinTime)? time : MinTime;

    }

    return MinTime;
}

static double pivperformance(int n, const std::function<void(const matrix_view<double>&, const row_view<double>&, const row_view<int>&)>& LTLT, int repitation = 3)
{
    auto MinTime = std::numeric_limits<double>::max();
    auto A = random_matrix(n, n);
    matrix<double> B = A - A.T();
    matrix<double> B0 = B;

    for (auto i : range(repitation))
    {
        row<double> t{n-1};
        row<int> p{n};
        auto B = B0;
        auto B_deepcopy = B;

        auto starting_point =  bli_clock();
        LTLT(B,t,p);
        auto ending_point = bli_clock();

        auto time = ending_point - starting_point;
        printf("Rep and time: %d, %f\n", i, time);

        MinTime = (time < MinTime)? time : MinTime;
    }

    return MinTime;

}

int main(int argc, const char** argv)
{
    /*
     *  testing with different matrix size
     */

    static const char USAGE[] =
    R"(ltlt.
      Usage:
        ltlt <majoralgo> <matrixsize_min> <matrixsize_max> <stepsize> <repetition> [--minoralgo=<minoralgo>] [--bs=<bs>] [--step=<step>]
        ltlt (-h | --help)
        ltlt --version

      Options:
        <majoralgo>       The major algorithm for ltlt, including UnBlock algorithms and Block algorithms
                          Block algorithms:
                                            Block Right Looking (ltlt_blockRL)
                                            Block Left Looking (ltlt_blockLL)
                          UnBlock algorithms:
                                            UnBlock Right Looking (ltlt_unblockRL)
                                            Unblock Left Looking (ltlt_unblockLL)
                                            UnBlock Two Step Right Looking (ltlt_unblockTSRL)
                                            Pivot Unblock Left Looking (ltlt_pivot_unblockLL)
                                            Pivot Unblock Right Looking (ltlt_pivot_unblockRL)
                          piv block algorithms:
                                            pivot Block Right Looking var0 (ltlt_pivot_blockRL (it only applies to unblock left looking))

        <matrixsize_min>  The min size of skew matirx we plan to decompose

        <matrixsize_max>  The max size of skew matirx we plan to decompose

        <stepsize>        The step of range of matrix size

        <repetition>      Times of repetition

        [--minoralgo]     If the majoralgo is Block algorithms, then a minoralgo is needed.
                          There are two options:
                                                UnBlock Right Loooking (ltlt_unblockRL),
                                                Unblock Left Loooking (ltlt_unblockLL).
                                                Pivot Unblock Left Looking (ltlt_pivot_unblockLL) only work for piv right looking.

        [--bs]            The block size for block algorithm [default: 256]

        [--step]          Step to use [default: 5]

        -h --help         Show this screen.
        --version        Show version.
    )";

    std::vector<std::string> raw_args;
    for (auto i : range(1,argc))
    raw_args.push_back(argv[i]);

    std::map<std::string, docopt::value> args = docopt::docopt(USAGE,
                                             raw_args,
                                             true,
                                             "LTLT 1.0");

    auto majoralgo = args["<majoralgo>"].asString();
    auto matrixsize_min = args["<matrixsize_min>"].asLong();
    auto matrixsize_max = args["<matrixsize_max>"].asLong();
    auto step = args["<stepsize>"].asLong();
    auto repetition = args["<repetition>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong(): 256;
    auto algo_step = args["--step"] ? args["--step"].asLong(): 5;

    auto perf = [&] <int Options>
    {
        for (auto matrixsize = matrixsize_min; matrixsize <=  matrixsize_max; matrixsize += step)
        {
            double time;
            if (minoralgo.empty())
            {
                if (majoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, unblocked(ltlt_unblockLL<Options>), repetition);

                else if (majoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, unblocked(ltlt_unblockRL<Options>), repetition);

                else if (majoralgo == "ltlt_pivot_unblockLL")
                    time = pivperformance(matrixsize, unblocked(ltlt_pivot_unblockLL<Options>), repetition);

                else if (majoralgo == "ltlt_pivot_unblockRL")
                    time = pivperformance(matrixsize, unblocked(ltlt_pivot_unblockRL<Options>), repetition);

                else if (majoralgo == "ltlt_unblockTSRL")
                    time = performance(matrixsize, unblocked(ltlt_unblockTSRL<Options>), repetition);

                else
                {
                    std::cerr << "The Algorithm is not suppotted" << std::endl;
                    exit(1);
                }
            }
            else
            {
                if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, blocked(ltlt_blockRL<Options>, ltlt_unblockRL<Options>, blocksize), repetition);

                else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, blocked(ltlt_blockRL<Options>, ltlt_unblockLL<Options>, blocksize), repetition);

                else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, blocked(ltlt_blockLL<Options>, ltlt_unblockRL<Options>, blocksize), repetition);

                else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, blocked(ltlt_blockLL<Options>, ltlt_unblockLL<Options>, blocksize), repetition);

                else if (majoralgo == "ltlt_pivot_blockRL" && minoralgo == "ltlt_pivot_unblockLL")
                    time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL<Options>, ltlt_pivot_unblockLL<Options>, blocksize), repetition);

                else
                {
                    std::cerr << "The Algorithm is not suppotted" << std::endl;
                    exit(1);
                }

            }

            auto GFLOPS = pow(matrixsize,3)/(time*3e9);
            printf("matrixsize, blocksize, time, GFLOPS = %d, %d, %f, %f\n", (int)matrixsize, (int)blocksize, time, GFLOPS);

            int nt = 0;
            #pragma omp parallel
            {
                nt = omp_get_num_threads();
            }
            output_to_csv(nt, matrixsize, majoralgo, minoralgo, blocksize, time, GFLOPS);
        }
    };

    switch (algo_step)
    {
        case 0: perf.operator()<STEP_0>(); break;
        case 1: perf.operator()<STEP_1>(); break;
        case 2: perf.operator()<STEP_2>(); break;
        case 3: perf.operator()<STEP_3>(); break;
        case 4: perf.operator()<STEP_4>(); break;
        case 5: perf.operator()<STEP_5>(); break;
    }

    timer::print_timers();

    return 0;
}
