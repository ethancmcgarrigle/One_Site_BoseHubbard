#ifndef _TAUGRID_H_
#define _TAUGRID_H_

#include "global.h"

class TauGrid
{
  public:
    // ctor
    TauGrid(double taumax, size_t ntau, bool suppressOutput=false);
    // dtor
    ~TauGrid() {};

    // Mesh points mappings on the imtime grid
    double tauOfIndx(size_t idx) const;
    double wOfIndx(size_t idx) const;

    // TauGrid / domain extents
    size_t getNTau() const {return _ntau;};
    double getTauMax() const {return _taumax;};

  private:
    // Temporal / contour grid
    double _taumax; /// Maximum extent of tau grid
    size_t _ntau;   /// Number of tau points
};

#endif // _TAUGRID_H_
