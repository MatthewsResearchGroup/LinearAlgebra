#include <cstdlib>
#include <map>
#include <math.h>

#include "test.hpp"
#include "docopt.h"
#include "io.hpp"

std::mt19937_64 gen(time(nullptr));

int main(int argc, const char** argv)
{
    /*
     *  testing with different matrix size
     */

    static const char USAGE[] =
    R"(ltlt.
      Usage:
        ltlt <majoralgo> <matrixsize_min> <matrixsize_max> <stepsize> <repitation> [--minoralgo=<minoralgo>] [--bs=<bs>] [--step=<step>]
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

        <repitation>      Times of repitation [default: 3]

        [--minoralgo]     If the majoralgo is Block algorithms, then a minoralgo is needed.
                          There are two options:
                                                UnBlock Right Loooking (ltlt_unblockRL),
                                                Unblock Left Loooking (ltlt_unblockLL).
                                                Pivot Unblock Left Looking (ltlt_pivot_unblockLL) only work for piv right looking.

        [--bs]            The block size for block algorithm

        [--step]          Step to use [default: 5]

        -h --help         Show this screen.
         --version        Show version.
    )";

    std::map<std::string, docopt::value> args = docopt::docopt(USAGE,
                                             { argv+1, argv+argc },
                                             true,
                                             "LTLT 1.0");

    // int n {0}, blocksize {0};
    // std::string majoralgo {}, minoralgo {};
    // double error, time;
    // int repitation;

    auto majoralgo = args["<majoralgo>"].asString();
    auto matrixsize_min = args["<matrixsize_min>"].asLong();
    auto matrixsize_max = args["<matrixsize_max>"].asLong();
    auto step = args["<stepsize>"].asLong();
    auto repitation = args["<repitation>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong(): 0;
    auto algo_step = args["--step"] ? args["--step"].asLong(): 0;
    //PROFILE_SECTION("main function")

    auto perf = [&] <int Options>
    {
        for (auto matrixsize = matrixsize_min; matrixsize <=  matrixsize_max; matrixsize += step)
        {
            double time;
            if (minoralgo.empty())
            {
                if (majoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, unblocked(ltlt_unblockLL<Options>), repitation);

                else if (majoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, unblocked(ltlt_unblockRL<Options>), repitation);

                else if (majoralgo == "ltlt_pivot_unblockLL")
                    time = pivperformance(matrixsize, unblocked(ltlt_pivot_unblockLL<Options>), repitation);

                else if (majoralgo == "ltlt_pivot_unblockRL")
                    time = pivperformance(matrixsize, unblocked(ltlt_pivot_unblockRL<Options>), repitation);

                else if (majoralgo == "ltlt_unblockTSRL")
                    time = performance(matrixsize, unblocked(ltlt_unblockTSRL<Options>), repitation);

                else
                {
                    std::cerr << "The Algorithm is not suppotted" << std::endl;
                    exit(1);
                }
            }
            else
            {
                if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, blocked(ltlt_blockRL<Options>, ltlt_unblockRL<Options>, blocksize), repitation);

                else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, blocked(ltlt_blockRL<Options>, ltlt_unblockLL<Options>, blocksize), repitation);

                else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
                    time = performance(matrixsize, blocked(ltlt_blockLL<Options>, ltlt_unblockRL<Options>, blocksize), repitation);

                else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockLL")
                    time = performance(matrixsize, blocked(ltlt_blockLL<Options>, ltlt_unblockLL<Options>, blocksize), repitation);

                else if (majoralgo == "ltlt_pivot_blockRL" && minoralgo == "ltlt_pivot_unblockLL")
                    time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL<Options>, ltlt_pivot_unblockLL<Options>, blocksize), repitation);

                else
                {
                    std::cerr << "The Algorithm is not suppotted" << std::endl;
                    exit(1);
                }

            }
            //auto GFLOPS = check_RL(majoralgo)? 3*pow(matrixsize,3)/(time*3e9) : pow(matrixsize,3)/(time*3e9);
            auto GFLOPS = pow(matrixsize,3)/(time*3e9);
            printf("matrixsize, blocksize, time, GFLOPS = %d, %d, %f, %f\n", (int)matrixsize, (int)blocksize, time, GFLOPS);
             // for (auto i : range(repitation))
            int nt = 0;
            #pragma omp parallel
            {
                nt = omp_get_num_threads();
            }
            //printf("We are using %d threads\n", nt);
            output_to_csv(nt, matrixsize, majoralgo, minoralgo, blocksize, time, GFLOPS);
        }
    };
    //PROFILE_STOP

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
