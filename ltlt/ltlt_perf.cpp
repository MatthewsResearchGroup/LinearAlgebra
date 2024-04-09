#include <cstdlib>
#include <string>
#include <map>

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
        ltlt <majoralgo> <matrixsize_min> <matrixsize_max> <step> <repitation> [--minoralgo=<minoralgo>] [--bs=<bs>]
        ltlt (-h | --help)
        ltlt --version
    
      Options:
        <majoralgo>       The major algorithm for ltlt, including UnBlock algorithms and Block algorithms                                                     
                          Block algorithms: 
                                            Block Right Loooking (ltlt_blockRL), 
                                            Block Left Loooking (ltlt_blockLL).
                          UnBlock algorithms: 
                                            UnBlock Right Loooking (ltlt_unblockRL), 
                                            Unblock Left Loooking (ltlt_unblockLL)
                                            UnBlock Two Step Right Looking (ltlt_unblockTSRL)

        <matrixsize_min>  The min size of skew matirx we plan to decompose

        <matrixsize_max>  The max size of skew matirx we plan to decompose

        <step>            The step of range of matrix size

        <repitation>      Times of repitation [default: 3]

        [--minoralgo]     If the majoralgo is Block algorithms, then a minoralgo is needed. 
                          There are two options: 
                                                UnBlock Right Loooking (ltlt_unblockRL),
                                                Unblock Left Loooking (ltlt_unblockLL).

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
    auto matrixsize_min = args["<matrixsize_min>"].asLong();
    auto matrixsize_max = args["<matrixsize_max>"].asLong();
    auto step = args["<step>"].asLong();
    auto repitation = args["<repitation>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong(): 0;

    for (auto matrixsize = matrixsize_min; matrixsize <=  matrixsize_max; matrixsize += step)
    {   
        std::vector<double> error_vec {}, time_vec{};
        if (minoralgo.empty())
        { 
            if (majoralgo == "ltlt_unblockLL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, unblocked(ltlt_unblockLL), repitation);

            else if (majoralgo == "ltlt_unblockRL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, unblocked(ltlt_unblockRL), repitation);

            else if (majoralgo == "ltlt_unblockTSRL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, unblocked(ltlt_unblockTSRL), repitation);

            else
            {
                std::cerr << "The Algorithms is not suppotted" << std::endl;
                exit(1);
            }
        }
        else
        {
            if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, blocked(ltlt_blockRL, ltlt_unblockRL, blocksize), repitation);

            else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, blocked(ltlt_blockRL, ltlt_unblockLL, blocksize), repitation);

            else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, blocked(ltlt_blockLL, ltlt_unblockRL, blocksize), repitation);

            else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockLL")
                std::tie(error_vec, time_vec) = test_perf(matrixsize, blocked(ltlt_blockLL, ltlt_unblockLL, blocksize), repitation);

            else
            {
                std::cerr << "The Algorithms is not suppotted" << std::endl;
                exit(1);
            }
    
        }

        for (auto i : range(repitation))
            output_to_csv(matrixsize, majoralgo, minoralgo, blocksize, error_vec[i], time_vec[i]);
    }


    return 0;

}
