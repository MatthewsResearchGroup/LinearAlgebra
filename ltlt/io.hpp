#ifndef _LTLT_IO_HPP_
#define _LTLT_IO_HPP_

#include <string>
#include <stdio.h>
#include <fstream>
#include <filesystem>


void output_to_csv(const int& nt,
                   const int& MatrixSize, 
                   const std::string& MajorAlgo,
                   const std::string& MinorAlgo,
                   const int& BlockSize,
                   const double& time,
                   const double& GFLOPS)
{
    std::string filename;
    if (MinorAlgo.empty() )
        filename = "./time.csv";
    else
        filename = "./time_" + std::to_string(BlockSize) + ".csv";   
    // std::string filename = "./time_" + std::to_string(MatrixSize) + ".csv";

    auto out_csv = fopen(filename.c_str(),"a+");

    if (!out_csv)
    {
        printf("\nERROR: could not open %s for output\n",filename.c_str());
        exit(1);
    }
    //Generate formatting for output
    std::string header = "NUM_THREADS, MatrixSize, MajorAlgo, MinorAlgo, BlockSize, Time, GFLOPS";
    std::string values_f = "%6d, %6d, %s, %s, %6d, %E, %E\n";


    if (std::filesystem::is_empty(filename))
        fprintf(out_csv,"%s\n",header.c_str());

    fprintf(out_csv,           values_f.c_str(),  nt,
            MatrixSize,         MajorAlgo.c_str(),
            MinorAlgo.c_str(),  BlockSize,
            time,               GFLOPS);

    // close the file if it's the last element in the points set vector.
    fclose(out_csv);
}



#endif
