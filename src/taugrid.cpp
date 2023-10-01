#include "taugrid.h"

TauGrid::TauGrid(double taumax, size_t ntau, bool suppressOutput/*=false*/)
  : _taumax(taumax), _ntau(ntau)
{
  // Skip optional output if requested
  if(!suppressOutput)
  {
    std::cout << "* Initializing imaginary time grid: " << std::endl;
    std::cout << "  - taumax = " << _taumax << std::endl;
    std::cout << "  - ntau   = " << _ntau << std::endl;
    std::cout << std::endl;
  }
}


double TauGrid::tauOfIndx(size_t idx) const
{
  assert(idx < _ntau);

  return double(idx)/double(_ntau)*_taumax;
}


double TauGrid::wOfIndx(size_t idx) const
{
  assert(idx < _ntau);

  if(idx > _ntau/2)
    return long(idx-_ntau)*TPI/_taumax;
  else
    return idx*TPI/_taumax;
}
