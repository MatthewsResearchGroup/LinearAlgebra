#include "ltlt.hpp"
#include "test.hpp"
#include "docopt.h"
#include "io.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

using namespace performance;

int main(int argc, const char** argv)
{
    /*
     *  testing with different matrix size
     */

    // auto n = 10; // square matrix size
    // auto blocksize = 3;
    //
    static const char USAGE[] =
    R"(ltlt.
      Usage:
        ltlt <majoralgo> <matrixsize> <repitation> [--minoralgo=<minoralgo>]  [--bs=<blocksize>] 
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

        <matrixsize>      The size of skew matirx we plan to decompose

        <repitation>       Times of repitation

        [--minoralgo]     If the majoralgo is Block algorithms, then a minoralgo is needed. 
                          There are two options: 
                                                UnBlock Right Loooking (ltlt_unblockRL),
                                                Unblock Left Loooking (ltlt_unblockLL).

        [--bs]            The size of partition matrix unblock algorithm need to decompose.

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
    std::vector<double> error_vec {}, time_vec{};

    auto majoralgo = args["<majoralgo>"].asString();
    auto n = args["<matrixsize>"].asLong();
    auto repitation = args["<repitation>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong() : 0; 
    
    //std::cout << majoralgo << n << repitation << std::endl;
    //for(auto const& arg : args) 
    //    std::cout << arg.first << ":" << arg.second << std::endl;
    // {
    //     if (arg.first == "<majoralgo>")
    //         majoralgo = arg.second.asString();

    //     if (arg.first == "<matrixsize>")
    //         n = arg.second.asLong();

    //     if (arg.first == "--minoralgo" && arg.second)
    //         minoralgo = arg.second.asString();
    //     
    //     if (arg.first == "--bs" && arg.second)
    //         blocksize = arg.second.asLong();

    //     if (arg.first == "-p")
    //         repitation = arg.second.asLong();

    // }

    if (majoralgo == "ltlt_unblockLL")
        std::tie(error_vec, time_vec) = performance::test(n, ltlt_unblockLL, repitation);

    else if (majoralgo == "ltlt_unblockRL")
        std::tie(error_vec, time_vec) = performance::test(n, ltlt_unblockRL, repitation);

    else if (majoralgo == "ltlt_unblockTSRL")
        std::tie(error_vec, time_vec) = performance::test(n, ltlt_unblockTSRL, repitation);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
        std::tie(error_vec, time_vec) = performance::test(n, blocksize, ltlt_blockRL, ltlt_unblockRL, repitation);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        std::tie(error_vec, time_vec) = performance::test(n, blocksize, ltlt_blockRL, ltlt_unblockLL, repitation);

    else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
        std::tie(error_vec, time_vec) = performance::test(n, blocksize, ltlt_blockLL, ltlt_unblockRL, repitation);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        std::tie(error_vec, time_vec) = performance::test(n, blocksize, ltlt_blockRL, ltlt_unblockLL, repitation);

    else
    {
        std::cerr << "The Algorithms is not suppotted" << std::endl;
        exit(1);
    }
    
    for (auto i : range(repitation))
        //std::cout << error_vec[i] << ", " <<  time_vec[i] << std::endl;
        output_to_csv(n, majoralgo, minoralgo, blocksize, error_vec[i], time_vec[i]);
    // std::cout << n << blocksize << majoralgo << minoralgo << error <<  time << std::endl;
    // std::cout << n << blocksize << majoralgo << minoralgo <<  time << std::endl;
    // output_to_csv(n, majoralgo, minoralgo, blocksize, error, time);

    
    
    // auto n = (int) args[1].first.asLong();
    // auto blocksize = (int)  args[1].second.asLong();


    //
    // test(n, ltlt_unblockLL);
    // test(n, ltlt_unblockRL);
    // test(n, ltlt_unblockTSRL);
    // test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);
    // test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);
    // test(n, blocksize, ltlt_blockLL, ltlt_unblockLL);
    // test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);
    //
    //
    //
    //
    //
    //
    // test(n, ltlt_blockRL, false);
    // test(n, ltlt_blockRL, true);
    // test(n, ltlt_blockLL, false);
    // test(n, ltlt_blockLL, true);

    return 0;

    //
    // Poviting 
    //
    //

}
