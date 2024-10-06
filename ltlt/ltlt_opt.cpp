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
        ltlt <majoralgo> <opt_step> <matrixsize_min> <matrixsize_max> <step> <repitation> [--bs=<bs>]
        ltlt (-h | --help)
        ltlt --version
    
      Options:
        <majoralgo>       The major algorithm for ltlt, including UnBlock algorithms and Block algorithms                                                     
                          Block algorithms: 
                                            Block Right Looking var0 (ltlt_blockRL_var0) 
                                            Block Right Looking var1 (ltlt_blockRL_var1) 
                          piv block algorithms:
                                            pivot Block Right Looking var0 (ltlt_pivot_blockRL_var0 (it only applies to unblock left looking))
                                            pivot Block Right Looking var1 (ltlt_pivot_blockRL_var1 (it only applies to unblock left looking))

        <opt_step>        Optimization step for block right looking algorithm with unblock Left Looking algorithm, including BRL_VAR0+UBLL, BRL_VAR1+UBLL, PIV_BRL_VAR0+UBLL, PIV_BRL_VAR1+UBLL
                          BRL_VAR0+UBLL:
                                           Step 0 : no optimization
                                           Step 1 : gemv-sktri optimization on UBLL
                                           Step 2 : gemv-sktri optimization on UBLL and SKR2 optimization on BRL_VAR0
                                           Step 3 : gemv-sktri on UBLL and SKR2, GEMMT-sktri optimization on BRL_VAR0

                          BRL_VAR1+UBLL: 
                                           Step 0 : no optimization
                                           Step 1 : gemv-sktri optimization on UBLL
                                           Step 2 : gem-stri optimization on UBLL and GEMMT-sktri optimization on BRL_VAR1
                    
                          PIV+BRL_VAR0+UBLL:
                                           Step 0 : no optimization
                                           Step 1 : gemv-sktri optimization on PIV_UBLL
                                           Step 2 : gemv-sktri optimization on PIV_UBLL and SKR2 optimization on PIV_BRL_VAR0
                                           Step 3 : gemv-sktri on PIV_UBLL and SKR2, GEMMT-sktri optimization on PIV_BRL_VAR0
                                           
                          PIV+BRL_VAR1+UBLL:
                                           Step 0 : no optimization
                                           Step 1 : gemv-sktri optimization on PIV_UBLL
                                           Step 2 : gem-stri optimization on PIV_UBLL and GEMMT-sktri optimization on PIV_BRL_VAR1

        <matrixsize_min>  The min size of skew matirx we plan to decompose

        <matrixsize_max>  The max size of skew matirx we plan to decompose

        <step>            The step of range of matrix size

        <repitation>      Times of repitation [default: 3]

        [--bs]            The block size for block algorithm

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
    auto opt_step = args["<opt_step>"].asLong();
    auto matrixsize_min = args["<matrixsize_min>"].asLong();
    auto matrixsize_max = args["<matrixsize_max>"].asLong();
    auto step = args["<step>"].asLong();
    auto repitation = args["<repitation>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong(): 0;
    //PROFILE_SECTION("main function")
        
    for (auto matrixsize = matrixsize_min; matrixsize <=  matrixsize_max; matrixsize += step)
    {   
        double time;
        if (majoralgo == "ltlt_blockRL_var0" && opt_step == 0)
            time = performance(matrixsize, blocked(ltlt_blockRL_var0_s0, ltlt_unblockLL_s0, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var0" && opt_step == 1)
            time = performance(matrixsize, blocked(ltlt_blockRL_var0_s0, ltlt_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var0" && opt_step == 2)
            time = performance(matrixsize, blocked(ltlt_blockRL_var0_s1, ltlt_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var0" && opt_step == 3)
            time = performance(matrixsize, blocked(ltlt_blockRL_var0_s2, ltlt_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var1" && opt_step == 0)
            time = performance(matrixsize, blocked(ltlt_blockRL_var1_s0, ltlt_unblockLL_s0, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var1" && opt_step == 1)
            time = performance(matrixsize, blocked(ltlt_blockRL_var1_s0, ltlt_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_blockRL_var1" && opt_step == 2)
            time = performance(matrixsize, blocked(ltlt_blockRL_var1_s1, ltlt_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var0" && opt_step == 0)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var0_s0, ltlt_pivot_unblockLL_s0, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var0" && opt_step == 1)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var0_s0, ltlt_pivot_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var0" && opt_step == 2)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var0_s1, ltlt_pivot_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var0" && opt_step == 3)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var0_s2, ltlt_pivot_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var1" && opt_step == 0)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var1_s0, ltlt_pivot_unblockLL_s0, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var1" && opt_step == 1)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var1_s0, ltlt_pivot_unblockLL_s1, blocksize), repitation);

        else if (majoralgo == "ltlt_pivot_blockRL_var1" && opt_step == 2)
            time = pivperformance(matrixsize, blocked(ltlt_pivot_blockRL_var1_s1, ltlt_pivot_unblockLL_s1, blocksize), repitation);

        else
        {
            std::cerr << "The Algorithm is not suppotted" << std::endl;
            exit(1);
        }
        //auto GFLOPS = check_RL(majoralgo)? 3*pow(matrixsize,3)/(time*3e9) : pow(matrixsize,3)/(time*3e9);
        auto GFLOPS = pow(matrixsize,3)/(time*3e9);
        printf("matrixsize, blocksize, time, GFLOPS = %d, %d, %f, %f\n", matrixsize, blocksize, time, GFLOPS);
         // for (auto i : range(repitation))
        int nt = 0;
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
        //printf("We are using %d threads\n", nt);
        output_to_csv(nt, matrixsize, majoralgo, opt_step, blocksize, time, GFLOPS);
    }
    //PROFILE_STOP

    timer::print_timers();

    return 0;

}
