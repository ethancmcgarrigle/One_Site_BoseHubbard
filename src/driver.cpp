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
{
  std::cout << "Constructing the driver" << std::endl;
  std::cout << " - Repulsion strength U = " << _U << std::endl;
  std::cout << " - Chemical potential = " << _mu << std::endl;
  std::cout << " - Temperature = " << 1./_beta << std::endl;
  std::cout << " Number of terms to compute in the sum: " << _N << std::endl;
  
  // initialize the sum 
  S = 0.0; 

  // Initialize a vector of length N with zeros
  //_summands.push_back()  
  assert(_summands.size() == _N);
  std::fill(E_n.begin(), E_n.end(), 0.);
  std::fill(_summands.begin(), _summands.end(), 0.);
  std::fill(_tmp.begin(), _tmp.end(), 0.);

  // Pre-compute the particle number basis states 
  //_n = std::iota(_n.begin(), _n.end(), 1);
  std::iota(_n.begin(), _n.end(), 1);

}

Driver::~Driver()
{
    // destructor 

}

//
void Driver::compute_sum()
{
    // compute summand e{-beta * E(n)}
    std::fill(E_n.begin(), E_n.end(), 0.);
    std::fill(_tmp.begin(), _tmp.end(), 0.);

    // *= operation
    std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
      a *= b;
      return a;
    }); 

    // += operation
    std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
      a += b;
      return a;
    }); 


    //E_n += double(_n);
    // += operation
    std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
      a += double(b);
      return a;
    }); 

    
    //E_n *= double(_n);
    std::transform(E_n.begin(), E_n.end(), _n.begin(), E_n.begin(), [](double& a, int b) {
      a *= double(b);
      return a;
    }); 

    //E_n *= _U/2.;
    double factor = _U * 0.5;
    std::transform(E_n.begin(), E_n.end(), E_n.begin(), [factor](double a ) {
      //a *= factor;
      return a * factor;
    }); 
    
    // Fill container with linear term
    //_tmp += _n;
    // += operation
    std::transform(_tmp.begin(), _tmp.end(), _n.begin(), _tmp.begin(), [](double& a, double b) {
      a += double(b);
      return a;
    }); 
    factor = 0.;
    factor = -(0.5*_U + _mu);
    std::transform(_tmp.begin(), _tmp.end(), _tmp.begin(), [factor](double a) {
      //a *= double(b);
      return a * factor;
    }); 
    //_tmp *= (_U/2. + _mu);
    //_tmp *= -1.;

    //E_n += _tmp;
    std::transform(E_n.begin(), E_n.end(), _tmp.begin(), E_n.begin(), [](double& a, double b) {
      a += double(b);
      return a;
    }); 

    // Multiply by minus beta 
    //E_n *= -_beta;
    factor = 0.;
    factor = -_beta;
    std::transform(E_n.begin(), E_n.end(), E_n.begin(), [factor](double a) {
      //a *= double(b);
      return a * factor;
    }); 

    // reset _tmp? 
    std::transform(E_n.begin(), E_n.end(), _tmp.begin(), [](double x) {return std::exp(x);} ); 

    S += std::accumulate(_tmp.begin(), _tmp.end(), 0.); // accumulate the sum! 

    std::cout << "The desired sum is: " << S << std::endl;
}
