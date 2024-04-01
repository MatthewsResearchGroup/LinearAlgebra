#include "ltlt.hpp"
#include "test.hpp"
#include "docopt.h"
#include "io.hpp"
// #include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <map>
#include <type_traits>

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
        ltlt <majoralgo> <matrixsize> [--minoralgo=<minoralgo>]  [--bs=<blocksize>] 
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

        [--minoralgo]   If the majoralgo is Block algorithms, then a minoralgo is needed. 
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

    for(auto const& arg : args) 
        std::cout << arg.first << ":" << arg.second << ":"<< std::endl;

    // int n {0}, blocksize {0};
    // std::string majoralgo {}, minoralgo {};
    double error, time;


    auto majoralgo = args["<majoralgo>"].asString();
    auto n = args["<matrixsize>"].asLong();
    auto minoralgo = args["--minoralgo"] ? args["--minoralgo"].asString() : std::string("");
    auto blocksize = args["--bs"] ? args["--bs"].asLong() : 0; 


    if (majoralgo == "ltlt_unblockLL")
        std::tie(error, time) = test(n, ltlt_unblockLL);

    else if (majoralgo == "ltlt_unblockRL")
        std::tie(error, time) = test(n, ltlt_unblockRL);

    else if (majoralgo == "ltlt_unblockTSRL")
        std::tie(error, time) = test(n, ltlt_unblockTSRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
        std::tie(error, time) = test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        std::tie(error, time) = test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);

    else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
        std::tie(error, time) = test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        std::tie(error, time) = test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);

    else
    {
        std::cerr << "The Algorithms is not suppotted" << std::endl;
        exit(1);
    }

    // std::cout << n << blocksize << majoralgo << minoralgo << error <<  time << std::endl;
    // std::cout << n << blocksize << majoralgo << minoralgo <<  time << std::endl;
    output_to_csv(n, majoralgo, minoralgo, blocksize, error, time);

    
    
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
