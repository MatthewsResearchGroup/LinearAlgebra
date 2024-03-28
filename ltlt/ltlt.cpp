#include "ltlt.hpp"
#include "test.hpp"
#include "docopt.h"
// #include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

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

        [----minoralgo]   If the majoralgo is Block algorithms, then a minoralgo is needed. 
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

    int n {0}, blocksize {0};
    std::string majoralgo, minoralgo;

    std::cout << n << blocksize << majoralgo << minoralgo << std::endl;
    for(auto const& arg : args) 
    {
        if (arg.first == "<majoralgo>")
            majoralgo = arg.second.asString();

        if (arg.first == "<matrixsize>")
            n = arg.second.asLong();

        if (arg.first == "[--minoralgo=<minoralgo>]")
            minoralgo = arg.second.asString();
        
        if (arg.first == "[--bs=<blocksize>]")
            blocksize = arg.second.asLong();
    }

    std::cout << n << blocksize << majoralgo << minoralgo << std::endl;
    if (majoralgo == "ltlt_unblockLL")
        test(n, ltlt_unblockLL);

    else if (majoralgo == "ltlt_unblockRL")
        test(n, ltlt_unblockRL);

    else if (majoralgo == "ltlt_unblockTSRL")
        test(n, ltlt_unblockTSRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockRL")
        test(n, blocksize, ltlt_blockRL, ltlt_unblockRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);

    else if (majoralgo == "ltlt_blockLL" && minoralgo == "ltlt_unblockRL")
        test(n, blocksize, ltlt_blockLL, ltlt_unblockRL);

    else if (majoralgo == "ltlt_blockRL" && minoralgo == "ltlt_unblockLL")
        test(n, blocksize, ltlt_blockRL, ltlt_unblockLL);

    else
    {
        std::cerr << "The Algorithms is not suppotted" << std::endl;
        exit(1);
    }

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
