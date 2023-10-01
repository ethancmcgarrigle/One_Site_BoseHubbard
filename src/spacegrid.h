#ifndef _SPACEGRID_H_
#define _SPACEGRID_H_

#include "global.h"

class SpaceGrid
{
  public:
    // ctor
    SpaceGrid(std::vector<double> L, std::vector<size_t> Nx, size_t ndim, bool suppressOutput=false);
    // dtor
    ~SpaceGrid() {};

    // Functions for mapping to/from grid indices.
    std::vector<int> MapFromFFTindx(size_t idx, bool zerocentered=false) const; /// Map between grid index and offset for the _ndim dimensional spatial grid
    size_t MapToFFTindx(std::vector<int> const &coord) const; /// Map from a grid offset to the mesh index for the _ndim dimensional spatial grid

    // Mesh points mappings on the spatial grid
    std::vector<double> rvecOfIndx(size_t idx, bool zerocentered=false) const;
    std::vector<double> rfracOfIndx(size_t idx, bool zerocentered=false) const;
    std::vector<double> kvecOfIndx(size_t idx) const;
    double k2OfIndx(size_t idx) const;
    size_t indxOfNegK(size_t idx) const;

    // Grid / domain extents
    size_t getSpatialDim() const {return _ndim;};
    size_t getGridSize(size_t idir) const {assert(idir<_ndim); return _Nx[idir];};
    size_t getNSpatial() const {return _ntot_spatial;};
    double getCellLength(size_t idir) const {assert(idir<_ndim); return _L[idir];}; // Cell is cubic
    double getVolume() const {return _Vol;};

  private:
    // Spatial grid
    size_t _ndim; /// Dimension of spatial cell
    std::vector<double> _L;    /// Linear size of simulation cell
    double _Vol;  /// Volume of simulation cell
    std::vector<size_t> _Nx;   /// Mesh points per dimension
    size_t _ntot_spatial; /// Total spatial mesh points
};

#endif // _SPACEGRID_H_
