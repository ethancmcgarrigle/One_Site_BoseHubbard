#include "spacegrid.h"

SpaceGrid::SpaceGrid(std::vector<double> L, std::vector<size_t> Nx, size_t ndim, bool suppressOutput/*=false*/)
  : _ndim(ndim), _L(L), _Vol(0.), _Nx(Nx), _ntot_spatial(1)
{
  // Validate input
  if(_ndim < 1 || _ndim > 3)
  {
    std::cout << "ndim = " << _ndim << std::endl;
    usererror_exit("Number of spatial dimensions must be between 1 and 3",__FILE__,__LINE__);
  }

  // Initialize computed members
  _ntot_spatial = _Nx[0];
  _Vol = _L[0];
  for(size_t i=1; i<_ndim; i++)
  {
    _ntot_spatial *= _Nx[i];
    _Vol *= _L[i];
  }

  // Skip optional output if requested
  if(!suppressOutput)
  {
    std::cout << "* Inititalizing spatial grid: " << std::endl;
    std::cout << "  - Dim = " << _ndim << std::endl;
    std::cout << "  - L   = " << std::endl;
    for(size_t i=0; i<_ndim; i++)
      std::cout << "      dir-" << i << "    = " << _L[i] << std::endl;

    std::cout << "  - Nx  = " << std::endl;
    for(size_t i=0; i<_ndim; i++)
      std::cout << "      dir-" << i << "    = " << Nx[i] << std::endl;

    std::cout << "  - Vol = " << _Vol << std::endl;
    std::cout << "  - Ntot= " << _ntot_spatial << std::endl;
    std::cout << std::endl;
  }
}


// NOTE: THIS IMPLEMENTATION ASSUMES ORTHORHOMBIC MESH.
std::vector<int> SpaceGrid::MapFromFFTindx(size_t idx, bool zerocentered/*=false*/) const
{
  std::vector<int> coord(_ndim);

  int lindx = idx;
  for(size_t idir=0; idir<_ndim; idir++)
  {
    size_t div=1; // Store the product of grid dimensions below current
    for(size_t jdir=idir+1; jdir<_ndim; jdir++)
      div *= _Nx[jdir]; // updated for orthorhombic mesh  
    coord[idir] = lindx / div;
    lindx -= div*coord[idir];
  }

  // Translate the grid if it is requested to have the origin at the center.
  // i.e., shift from [0,Nx) to [0,Nx/2)..[-Nx/2,0)
  if(zerocentered)
  {
    for(size_t idir=0; idir<_ndim; idir++)
      if(coord[idir] > int(_Nx[idir])/2)
        coord[idir] = coord[idir] - _Nx[idir];
  }

  return coord;
}


// NOTE: THIS IMPLEMENTATION ASSUMES ORTHORHMOBIC MESH.
size_t SpaceGrid::MapToFFTindx(std::vector<int> const &coord) const
{
  assert(coord.size() == _ndim);

  std::vector<int> local(coord);

  // First regularize the coordinate by mapping each
  // component into the range [0, Nx) using
  // periodic boundary conditions.
  for(size_t idir=0; idir<_ndim; idir++)
  {
    if(local[idir] < 0)
    {
      // Be careful for negative coordinates. We want coord-(coord/max-1)*max,
      // ASSUMING negative integer division rounds towards zero. This is language
      // (and maybe even implementation) dependent. Therefore, instead use
      // local = -abs(lmn) + ((-lmn)/max)*max, which uses only positive integer
      // division and is well defined
      local[idir] = ((-coord[idir]-1)/_Nx[idir]+1)*_Nx[idir];
      local[idir] += coord[idir]; // Since coord < 0, this is == -abs(coord)
    }
    else
    { // Positive coordinate => use a simple modulo
      local[idir] = coord[idir] % _Nx[idir];
    }
  }

  // Now compute the FFT index
  size_t idx = 0;
  for(size_t idir=0; idir<_ndim; idir++)
  {
    int shift=1;
    for(size_t jdir=idir+1; jdir<_ndim; jdir++)
      shift *= _Nx[jdir];
    idx += shift*local[idir];
  }

  return idx;
}


// NOTE: assumes orthorhombic cell and mesh
std::vector<double> SpaceGrid::rvecOfIndx(size_t idx, bool zerocentered/*=false*/) const
{
  std::vector<int> offset = MapFromFFTindx(idx, zerocentered);
  std::vector<double> r(_ndim);
  for(size_t idir=0; idir<_ndim; idir++)
    r[idir] = (_L[idir] * offset[idir]) / double(_Nx[idir]);

  return r;
}


// NOTE: assumes orthorhombic cell and mesh
std::vector<double> SpaceGrid::kvecOfIndx(size_t idx) const
{
  assert(idx < _ntot_spatial);

  std::vector<int> offset = MapFromFFTindx(idx, true);
  std::vector<double> kvec(_ndim);
  for(size_t idir=0; idir<_ndim; idir++)
    kvec[idir] = TPI / _L[idir] * offset[idir];

  return kvec;
}

// NOTE: assumes orthorhombic cell and mesh
std::vector<double> SpaceGrid::rfracOfIndx(size_t idx, bool zerocentered/*=false*/) const
{
  std::vector<int> offset = MapFromFFTindx(idx, zerocentered);
 
  std::vector<double> r(_ndim);
  for(size_t idir=0; idir<_ndim; idir++)
    r[idir] = offset[idir] / double(_Nx[idir]);

  return r;
}




double SpaceGrid::k2OfIndx(size_t idx) const
{
  assert(idx < _ntot_spatial);

  std::vector<double> kvec = kvecOfIndx(idx);
  double k2(0.);
  for(size_t idir=0; idir<_ndim; idir++)
    k2 += kvec[idir]*kvec[idir];

  return k2;
}


size_t SpaceGrid::indxOfNegK(size_t idx) const
{
  assert(idx < _ntot_spatial);

  std::vector<int> coord = MapFromFFTindx(idx);

  for(size_t idir=0; idir<_ndim; idir++)
    coord[idir] = -coord[idir];

  return MapToFFTindx(coord);
}
