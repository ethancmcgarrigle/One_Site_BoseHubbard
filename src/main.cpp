#define MAIN

//#include <iostream>
#include "version.h"
#include "parameters.h"
#include "global.h"
#include "driver.h"

//int main(int argc, char const *argv[])
int main(int argc, char **argv)
{

    std::cout << "Starting the program" << std::endl;

    if(argc != 2)
    {
      std::cout << "USAGE: " << argv[0] << std::endl;
      exit(2);
    }

    // Setup the parser 
    YAMLParser parser(argv[1]);
    double U(1.0), mu(1.0), beta(1.0);
    int Nterms(10);
    bool success;
    success = parser.getInt("N_terms", {"numerics"}, Nterms);
    assert(success);
    success &= parser.getDouble("U", {"system"}, U);
    success &= parser.getDouble("mu", {"system"}, mu);
    success &= parser.getDouble("beta", {"system"}, beta);

    // Sweep different numbers of terms in the sum to get a sense of convergence 
    //std::vector<int> _N_list(10);
    //_N_list = {5, 10, 25, 50, 75, 100, 300, 500, 1000};
    Driver *sum_driver(nullptr);

    std::cout << "Computing sum with " << Nterms << " terms" << std::endl;
    // Build the driver 
    sum_driver = new Driver(U, mu, beta, Nterms);

    // Compute the sum 
    sum_driver->compute_sum();

    // Destroy the driver 
    delete sum_driver;
//
//
//    for(size_t i = 0; i < _N_list.size(); i++)
//    {
//        std::cout << 'Computing sum with ' << _N_list[i] << ' terms' << std::endl;
//        // Build the driver 
//        sum_driver = new driver(U, mu, beta, _N_list[i]);
//
//        // Compute the sum 
//        sum_driver->compute_sum();
//
//        // Destroy the driver 
//        delete sum_driver;
//        
//    }
    return 0;

}
