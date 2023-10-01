#ifndef _DRIVER_H_
#define _DRIVER_H_

//#include <iostream>
//#include <vector>
#include "global.h"

class Driver {
  public:
    // ctor 
    Driver(double U, double mu, double beta, int N_terms);

    // destructor
    ~Driver();

    void compute_Z();
    void compute_U();
    void compute_N();
    void compute_E_n();

  private:
    double _Z, _energy_internal, _avg_N, _avg_N2; // the desired sum 
    double _U;
    double _mu;
    double _beta; 
    int _N;
    std::vector<double> E_n, _tmp, _tmp2;
    std::vector<double> _summands;
    std::vector<int> _n; // vector of _n particle number basis states 
    std::ofstream _operators_output; 

};

#endif
