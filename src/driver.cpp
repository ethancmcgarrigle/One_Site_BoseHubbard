#include "driver.h"


Driver::Driver(double U, double mu, double beta, int N_terms)
  : _U(U)
  , _mu(mu)
  , _beta(beta)
  , _N(N_terms)
  , _summands(N_terms, 0.)
  , _n(N_terms, 0.)
  , E_n(N_terms, 0.)
  , _tmp(N_terms, 0.)
  , _tmp2(N_terms, 0.)
  , _operators_output("operators.dat")
{
  std::cout << "Constructing the driver" << std::endl;
  std::cout << " - Repulsion strength U = " << _U << std::endl;
  std::cout << " - Chemical potential = " << _mu << std::endl;
  std::cout << " - Temperature = " << 1./_beta << std::endl;
  std::cout << " Number of terms to compute in the sum: " << _N << std::endl;
  
  // initialize the results  
  _Z = 0.; 
  _energy_internal = 0.;
  _avg_N = 0.;
  _avg_N2 = 0.;

  // Setup the operators file 
  _operators_output << "# Z _U _N _N2" << std::endl;
  _operators_output << std::scientific << std::setprecision(IOFLOATDIGITS) << std::showpos;
 

  // Initialize a vector of length N with zeros
  //_summands.push_back()  
  assert(_summands.size() == _N);
  std::fill(E_n.begin(), E_n.end(), 0.);
  std::fill(_summands.begin(), _summands.end(), 0.);
  std::fill(_tmp.begin(), _tmp.end(), 0.);
  std::fill(_tmp2.begin(), _tmp2.end(), 0.);

  // Pre-compute the particle number basis states 
  //_n = std::iota(_n.begin(), _n.end(), 1);
  std::iota(_n.begin(), _n.end(), 1);


}

Driver::~Driver()
{
  // destructor 
  _operators_output.close();
}


void Driver::compute_E_n()
{
  // compute summand e{-beta * E(n)}
  std::fill(E_n.begin(), E_n.end(), 0.);
  std::fill(_tmp.begin(), _tmp.end(), 0.);

//  // *= operation
//  std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
//    a *= b;
//    return a;
//  }); 
//
//  // += operation
//  std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
//    a += b;
//    return a;
//  }); 


  //E_n += double(_n);
  // += operation
  std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
    a += double(b) * double(b);
    return a;
  }); 

  
  //E_n *= double(_n);
 //  std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
 //    a *= double(b);
 //    return a;
 //  }); 

  //E_n *= _U/2.;
  double factor(0.); 
  factor = _U * 0.5;
  std::transform(E_n.begin(), E_n.end(), E_n.begin(), [factor](double a ) {
    //a *= factor;
    return a * factor;
  }); 
  
  // Fill container with linear term
  //_tmp += _n;
  // += operation
  std::transform(_tmp.begin(), _tmp.end(), _n.begin(), _tmp.begin(), [](double& a, int b) {
    a += double(b);
    return a;
  }); 

  factor = 0.;
  factor = -(0.5*_U + _mu);
  std::transform(_tmp.begin(), _tmp.end(), _tmp.begin(), [factor](double a) {
    return a * factor;
  }); 

  std::transform(E_n.begin(), E_n.end(), _tmp.begin(), E_n.begin(), [](double& a, double b) {
    a += b;
    return a;
  }); 

}


void Driver::compute_Z()
{
    // compute summand e{-beta * E(n)}
    std::fill(_tmp.begin(), _tmp.end(), 0.);

    // Multiply by minus beta 
    //E_n *= -_beta;
    double factor = 0.;
    factor = -_beta;
    std::transform(E_n.begin(), E_n.end(), _tmp.begin(), [factor](double a) {
      return a * factor;
    }); 

    // Store the Boltzmann weight in _tmp
    std::transform(_tmp.begin(), _tmp.end(), _tmp.begin(), [](double x) {return std::exp(x);} ); 

    _Z += std::accumulate(_tmp.begin(), _tmp.end(), 0.); // accumulate the sum! 

    std::cout << "The partition function: " << _Z << std::endl;

    _operators_output << _Z;
}



void Driver::compute_U()
{
  // Compute <U>
  std::fill(_tmp.begin(), _tmp.end(), 0.);

  // Put the Boltzmann weight into _tmp container  
  double factor = 0.;
  factor = -_beta;
  std::transform(E_n.begin(), E_n.end(), _tmp.begin(), [factor](double a) {
    return a * factor;
  }); 

  // Store the Boltzmann weight in _tmp
  std::transform(_tmp.begin(), _tmp.end(), _tmp.begin(), [](double x) {return std::exp(x);} ); // e^{-\beta E(n)) 
  
  // Compute the sum_{n} E(n) * boltzmann-weight, store in _tmp
  // first, multiply E(n) by the boltzmann weight 
  std::transform(_tmp.begin(), _tmp.end(), E_n.begin(), _tmp.begin(), [](double& a, double b) {
    a *= b;
    return a; // E(n) * e^{-\beta E(n))
  }); 

  _energy_internal += std::accumulate(_tmp.begin(), _tmp.end(), 0.); // numerator of the statistical measure 
  // Normalize by Z 
  _energy_internal /= _Z;

  std::cout << "The average internal energy  " << _energy_internal << std::endl;

  _operators_output << " " << _energy_internal;
}



void Driver::compute_N()
{
  // Compute <U>
  std::fill(_tmp.begin(), _tmp.end(), 0.);
  std::fill(_tmp2.begin(), _tmp2.end(), 0.);

  // Put the Boltzmann weight into _tmp container  
  double factor = 0.;
  factor = -_beta;
  std::transform(E_n.begin(), E_n.end(), _tmp.begin(), [factor](double a) {
    return a * factor;
  }); 

  // Store the Boltzmann weight in _tmp
  std::transform(_tmp.begin(), _tmp.end(), _tmp.begin(), [](double x) {return std::exp(x);} ); // e^{-\beta E(n)) 

  // Fill container2 with boltzmann weight  
  //_tmp2 += _tmp;
  // += operation
  std::transform(_tmp2.begin(), _tmp2.end(), _tmp.begin(), _tmp2.begin(), [](double& a, double b) {
    a += b;
    return a;
  }); //_tmp2 += _tmp;
  
  // Compute the sum_{n} E(n) * boltzmann-weight, store in _tmp
  // first, multiply E(n) by the boltzmann weight 
  std::transform(_tmp.begin(), _tmp.end(), _n.begin(), _tmp.begin(), [](double& a, int b) {
    a *= double(b);
    return a; // n * e^{-\beta E(n))
  }); 

  // Compute the sum_{n^2} E(n) * boltzmann-weight, store in _tmp2
  // first, multiply E(n) by the boltzmann weight 
  std::transform(_tmp2.begin(), _tmp2.end(), _n.begin(), _tmp2.begin(), [](double& a, int b) {
    a *= (double(b) * double(b));
    return a; // n * e^{-\beta E(n))
  }); 

  _avg_N += std::accumulate(_tmp.begin(), _tmp.end(), 0.); // numerator of the statistical measure 
  // Normalize by Z 
  _avg_N /= _Z; 

  _avg_N2 += std::accumulate(_tmp2.begin(), _tmp2.end(), 0.); // numerator of N^2 avg 
  _avg_N2 /= _Z;

  std::cout << "The average particle number " << _avg_N << std::endl;
  std::cout << "The average squared-particle number " << _avg_N2 << std::endl;
  _operators_output << " " << _avg_N; 
  _operators_output << " " << _avg_N2 << std::endl;
}
