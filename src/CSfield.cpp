#include "CSfield.h"
#include "field.h"
#include <cstring>
#include "random.h"

#ifdef __GPU__
#include <cuda_runtime_api.h>
#include "GPUmemhandler.h"
#include "GPUerrorchk.h"
#include "GPUVectorOps.h"
#else
#include <mm_malloc.h>
#endif

#define ALIGNMENT 64 // Use 64-byte alignment on all architectures.
//#if defined __SSE3__¬
//#define ALIGNMENT 16¬
//#elif defined __AVX__¬
//#define ALIGNMENT 32¬
//#endif

#ifdef __GPU__
// Forward declare the instance of the memory manager
namespace gpu{
extern cuda_memory_manager_class cuda_memory_manager;
}
#endif


//====================================================================================
// Constructors, copy and move constructors, copy and move assignment, destructor
template<typename T>
CSfield<T>::CSfield(SpaceGrid const & spacegrid, TauGrid const & taugrid, bool FFTable_space/*=true*/, bool InRealSpace/*=true*/, bool FFTable_tau/*=true*/, bool InTauSpace/*=true*/, std::string label/*="unnamed"*/)
  : _nelem(spacegrid.getNSpatial() * taugrid.getNTau())
  , _data(nullptr)
  , _label(label)
  , _spacegrid(spacegrid)
  , _taugrid(taugrid)
  , _inrealspace(InRealSpace)
  , _isFFTable_space(FFTable_space)
  , _intauspace(InTauSpace)
  , _isFFTable_tau(FFTable_tau)
#ifdef __GPU__
  , _cufftplan_rk(0)
  , _cufftplan_tw(0)
#else
  , _fftw_plan_rtok_many(0)
  , _fftw_plan_ktor_many(0)
  , _fftw_plan_tautow_many(0)
  , _fftw_plan_wtotau_many(0)
#endif
{
  iobase::cdbg << "CSfield " << _label << ": constructor" << std::endl;

  assert(_nelem>0);

#ifdef __GPU__
  _data = reinterpret_cast<T*>(gpu::cuda_memory_manager.allocate(_nelem*sizeof(T),_label,__FILE__,__LINE__));
#else
  _data = reinterpret_cast<T*>(_mm_malloc(_nelem*sizeof(T), ALIGNMENT));
  if(reinterpret_cast<size_t>(_data)%ALIGNMENT)
    codeerror_abort("Array us not correctly aligned. This will cause problems for SIMD intrinsics.",__FILE__,__LINE__);
#endif

  if(_data == nullptr)
    codeerror_abort("Memory allocation failed",__FILE__,__LINE__);

  CreateFFTPlans();
}

// Destructor
template<typename T>
CSfield<T>::~CSfield()
{
  iobase::cdbg << "CSfield destructor: " << _label << std::endl;

  DestroyFFTPlans();

#ifdef __GPU__
  gpu::cuda_memory_manager.deallocate(_data,__FILE__,__LINE__);
#else
  _mm_free(_data);
#endif
}

// Copy constructor
template<typename T>
CSfield<T>::CSfield(CSfield<T> const &source)
  : CSfield<T>(source._spacegrid, source._taugrid,source._isFFTable_space, source._inrealspace, source._isFFTable_tau, source._intauspace, source._label) // malloc
{
  iobase::cdbg << "CSfield " << _label << ": copy constructor from " << source._label << std::endl;

#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(_data, source.getDataPtr(), _nelem*sizeof(T), cudaMemcpyDeviceToDevice),__FILE__,__LINE__);
#else
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = source.getDataPtr()[m];
#endif // __GPU__
}

// Copy assignment
template<typename T>
CSfield<T> & CSfield<T>::operator=(CSfield<T> const &source)
{
  iobase::cdbg << "CSfield: copy assignment from " << source._label << " to " << _label << std::endl;

  // Skip self assignment
  if(&source == this)
    return *this;

  // Note that the grids must be identical to copy CSfields - we can't unseat and replace the grid reference in this.
  assert(&_spacegrid == &(source._spacegrid));
  assert(&_taugrid == &(source._taugrid));

  // The following one-liner uses the copy constructor to create a temporary and then the move assignment operator to transfer it to this.
  //return *this = CSfield<T>(source); // Use copy ctor code to make the temporary object

  // A more explicit version that avoids reallocation when unnecessary
  bool realloc = (_nelem != source._nelem); // Do we need to reallocate the storage, or just replace the contents?
  if(realloc)
  {
#ifdef __GPU__
    // 1. Allocate a new array and copy elements.
    //    Do this first to avoid undefined object on exception.
    T* newdata = reinterpret_cast<T*>(gpu::cuda_memory_manager.allocate(source._nelem*sizeof(T),_label,__FILE__,__LINE__));
    gpu::cudaErrChk(cudaMemcpy(newdata, source.getDataPtr(), source._nelem*sizeof(T), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
    // 2. Deallocate the old array and nullify
    //    NOTE: since the FFT plans are tied to the old pointer, free them here if needed
    DestroyFFTPlans();
    gpu::cuda_memory_manager.deallocate(_data,__FILE__,__LINE__);
#else
    // A more explicit version:
    // 1. Allocate a new array and copy elements.
    //    Do this first to avoid undefined object on exception.
    T* newdata = reinterpret_cast<T*>(_mm_malloc(source._nelem*sizeof(T), ALIGNMENT));
    if(reinterpret_cast<size_t>(newdata)%ALIGNMENT)
      codeerror_abort("Array us not correctly aligned. This will cause problems for SIMD intrinsics.",__FILE__,__LINE__);

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<source._nelem; m++)
      newdata[m] = source.getDataPtr()[m];
    // 2. Deallocate the old array and nullify
    //    NOTE: since the FFT plans are tied to the old pointer, free them here if needed
    DestroyFFTPlans();
    _mm_free(_data);
#endif // __GPU__

    // 3. Transfer the pointer
    _data = newdata;
    _nelem = source._nelem;

    // This container will not change its FFTability based on source's.
    // We can imagine copying data into a destination container precisely because it allows FFTs.
    // If this container is FFTable, we now need to re-create the FFTW plans since the storage pointer has moved.
    CreateFFTPlans();
  } else {
    // Simply copy the data from source.getDataPtr() to this->_data
#ifdef __GPU__
    gpu::cudaErrChk(cudaMemcpy(_data, source.getDataPtr(), _nelem*sizeof(T), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
#else
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<source._nelem; m++)
      _data[m] = source.getDataPtr()[m];
#endif // __GPU__
  }

  // Other members not related to data storage
//  _label = source._label; // We typically don't want to change the label of the object when copying. Can override with explicit setLabel() where needed.
  _inrealspace = source._inrealspace;
  _intauspace = source._intauspace;

  return *this;
}

// Move constructor
template<typename T>
CSfield<T>::CSfield(CSfield<T>&& tmp) noexcept
  : _nelem(tmp._nelem) // Do not delegate ctor - avoid malloc
  , _data(tmp._data)
  , _label(tmp._label)
  , _spacegrid(tmp._spacegrid)
  , _taugrid(tmp._taugrid)
  , _inrealspace(tmp._inrealspace)
  , _isFFTable_space(tmp._isFFTable_space)
  , _intauspace(tmp._intauspace)
  , _isFFTable_tau(tmp._isFFTable_tau)
#ifdef __GPU__
  , _cufftplan_rk(tmp._cufftplan_rk)
  , _cufftplan_tw(tmp._cufftplan_tw)
#else
  , _fftw_plan_rtok_many(tmp._fftw_plan_rtok_many) // Since the storage location is from the tmp (and by definition has the same pointer, size and alignment), it is fine to transfer FFTW plan ownership.
  , _fftw_plan_ktor_many(tmp._fftw_plan_ktor_many)
  , _fftw_plan_tautow_many(tmp._fftw_plan_tautow_many)
  , _fftw_plan_wtotau_many(tmp._fftw_plan_wtotau_many)
#endif
{
  iobase::cdbg << "CSfield " << _label << ": move constructor" << std::endl;

  // Leave tmp in ill-defined state
  tmp._data  = nullptr;
  tmp._nelem = 0;
  // The next lines will prevent tmp's destructor from releasing the FFTW plans that have been claimed by ``this''
  tmp._isFFTable_space = false;
  tmp._isFFTable_tau   = false;
#ifdef __GPU__
  tmp._cufftplan_rk = 0;
  tmp._cufftplan_tw = 0;
#else
  tmp._fftw_plan_rtok_many   = 0;
  tmp._fftw_plan_ktor_many   = 0;
  tmp._fftw_plan_tautow_many = 0;
  tmp._fftw_plan_wtotau_many = 0;
#endif
}

// Move assignment
template<typename T>
CSfield<T> & CSfield<T>::operator=(CSfield<T>&& tmp) noexcept
{
  iobase::cdbg << "CSfield: move assignment from " << tmp._label << " to " << _label << std::endl;

  // Skip self assignment
  if(&tmp == this)
    return *this;

  // Note that the grids must be identical to copy CSfields - we can't unseat and replace the grid reference in this.
  assert(&_spacegrid == &(tmp._spacegrid));
  assert(&_taugrid == &(tmp._taugrid));

  // Simply swap all elements to claim ownership of tmp's contents.
  // We need to clean up what was previously inside ``this'', but tmp's destructor will do that when
  // it goes out of scope (immediately after assignment is complete).
  std::swap(_nelem, tmp._nelem);
  std::swap(_data, tmp._data);
//  std::swap(_label, tmp._label); // We typically don't want to change the label of the object when copying. Can override with explicit setLabel() where needed.
  std::swap(_inrealspace, tmp._inrealspace);
  std::swap(_isFFTable_space, tmp._isFFTable_space);
  std::swap(_intauspace, tmp._intauspace);
  std::swap(_isFFTable_tau, tmp._isFFTable_tau);
#ifdef __GPU__
  std::swap(_cufftplan_rk, tmp._cufftplan_rk);
  std::swap(_cufftplan_tw, tmp._cufftplan_tw);
#else
  std::swap(_fftw_plan_rtok_many, tmp._fftw_plan_rtok_many);
  std::swap(_fftw_plan_ktor_many, tmp._fftw_plan_ktor_many);
  std::swap(_fftw_plan_tautow_many, tmp._fftw_plan_tautow_many);
  std::swap(_fftw_plan_wtotau_many, tmp._fftw_plan_wtotau_many);
#endif

  return *this;
}

//====================================================================================
// FFTs
template<>
void CSfield<std::complex<double>>::CreateFFTPlans()
{
  // create FFT plans for (r)<->(k)
  if(_isFFTable_space)
  {
    int Nx[3]; // 3D is the maximum
    for(int i=0; i<3; i++)
      Nx[i] = _spacegrid.getGridSize(i);
    int howmany = _taugrid.getNTau();
    int dist    = _spacegrid.getNSpatial();
    int stride  = 1;
    int ndim(_spacegrid.getSpatialDim());
#ifdef __GPU__
    cufftResult_t ierr;
    ierr = cufftPlanMany(&_cufftplan_rk, ndim, Nx,
                          Nx, stride, dist,
                          Nx, stride, dist,
                          CUFFT_Z2Z, howmany);
    if(ierr != CUFFT_SUCCESS)
    {
      std::stringstream msg;
      msg << "Error creating cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
      codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
    }
#else
    fftw_complex* data_fftw = reinterpret_cast<fftw_complex*>(_data);
    _fftw_plan_rtok_many = fftw_plan_many_dft(ndim, Nx, howmany,
                                              data_fftw, Nx, stride, dist,
                                              data_fftw, Nx, stride, dist,
                                              FFTW_FORWARD, FFTW_PATIENT);
    _fftw_plan_ktor_many = fftw_plan_many_dft(ndim, Nx, howmany,
                                              data_fftw, Nx, stride, dist,
                                              data_fftw, Nx, stride, dist,
                                              FFTW_BACKWARD, FFTW_PATIENT);
#endif
  }

  // create FFT plans for (tau)<->(w)
  if(_isFFTable_tau)
  {
    int howmany = _spacegrid.getNSpatial();
    int dist    = 1;
    int ntau    = _taugrid.getNTau();
    int stride  = _spacegrid.getNSpatial();
#ifdef __GPU__
    cufftResult_t ierr;
    ierr = cufftPlanMany(&_cufftplan_tw, 1, &ntau,
                          &ntau, stride, dist,
                          &ntau, stride, dist,
                          CUFFT_Z2Z, howmany);
    if(ierr != CUFFT_SUCCESS)
    {
      std::stringstream msg;
      msg << "Error creating cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
      codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
    }
#else
    fftw_complex* data_fftw = reinterpret_cast<fftw_complex*>(_data);
    _fftw_plan_tautow_many = fftw_plan_many_dft(1, &ntau, howmany,
                                              data_fftw, &ntau, stride, dist,
                                              data_fftw, &ntau, stride, dist,
                                              FFTW_FORWARD, FFTW_PATIENT);
    _fftw_plan_wtotau_many = fftw_plan_many_dft(1, &ntau, howmany,
                                              data_fftw, &ntau, stride, dist,
                                              data_fftw, &ntau, stride, dist,
                                              FFTW_BACKWARD, FFTW_PATIENT);
#endif
  }
}

template<>
void CSfield<double>::CreateFFTPlans()
{
  if(!_isFFTable_space && !_isFFTable_tau)
    return;

  codeerror_abort("Real-valued CSfields are not currently FFTable",__FILE__,__LINE__);
}

template<typename T>
void CSfield<T>::DestroyFFTPlans()
{
  if(_isFFTable_space)
  {
#ifdef __GPU__
    if(_cufftplan_rk)
      cufftDestroy(_cufftplan_rk);
#else
    if(_fftw_plan_rtok_many != 0)
      fftw_destroy_plan(_fftw_plan_rtok_many);
    if(_fftw_plan_ktor_many != 0)
      fftw_destroy_plan(_fftw_plan_ktor_many);
#endif
  }

  if(_isFFTable_tau)
  {
#ifdef __GPU__
    if(_cufftplan_tw)
      cufftDestroy(_cufftplan_tw);
#else
    if(_fftw_plan_tautow_many != 0)
      fftw_destroy_plan(_fftw_plan_tautow_many);
    if(_fftw_plan_wtotau_many != 0)
      fftw_destroy_plan(_fftw_plan_wtotau_many);
#endif
  }
}

template<typename T>
void CSfield<T>::fft_rtok(bool applyscale/*=true*/)
{
  if(!_inrealspace)
    codeerror_abort((std::string("FFT r->k called on CSfield already in k space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_space)
    codeerror_abort((std::string("FFT r->k called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);
  // Check plan was created
#ifdef __GPU__
  assert(_cufftplan_rk != 0);
  cufftResult_t ierr = cufftExecZ2Z(_cufftplan_rk, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_FORWARD);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing r->k cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }

  if(applyscale)
  {
    double norm(1./double(_spacegrid.getNSpatial()));
    gpu::scaleGPUvector(_data, norm, _nelem);
  }
#else
  assert(_fftw_plan_rtok_many != 0);
  fftw_execute(_fftw_plan_rtok_many);
  if(applyscale)
  {
    double norm(1./double(_spacegrid.getNSpatial()));
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= norm;
  }
#endif // __GPU__

  _inrealspace = false;
}

template<typename T>
void CSfield<T>::fft_rtonegk(bool applyscale/*=true*/)
{
  if(!_inrealspace)
    codeerror_abort((std::string("FFT r->-k called on CSfield already in k space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_space)
    codeerror_abort(std::string(("FFT r->-k called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);

  // Pretend we're already in k space to execute the correct direction of transform to effectively get -k
  setInRealSpace(false);
  fft_ktor(); // FFT k->r => r->-k
  setInRealSpace(false);

  // Check plan was created
  if(applyscale)
  {
    double norm(1./double(_spacegrid.getNSpatial()));
#ifdef __GPU__
    gpu::scaleGPUvector(_data, norm, _nelem);
#else
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= norm;
#endif // __GPU__
  }
}

template<typename T>
void CSfield<T>::fft_ktor()
{
  if(_inrealspace)
    codeerror_abort((std::string("FFT k->r called on CSfield already in r space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_space)
    codeerror_abort(std::string(("FFT k->r called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);

#ifdef __GPU__
  // Check plan was created
  assert(_cufftplan_rk != 0);
  // Execute
  cufftResult_t ierr = cufftExecZ2Z(_cufftplan_rk, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_INVERSE);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing k->r cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }
#else
  // Check plan was created
  assert(_fftw_plan_ktor_many != 0);
  // Execute
  fftw_execute(_fftw_plan_ktor_many);
#endif

  _inrealspace = true;
}

template<typename T>
void CSfield<T>::fft_tautow(bool applyscale/*=true*/)
{
  if(!_intauspace)
    codeerror_abort((std::string("FFT tau->w called on CSfield already in w space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_tau)
    codeerror_abort((std::string("FFT tau->w called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);

#ifdef __GPU__
  assert(_cufftplan_tw != 0);
  cufftResult_t ierr = cufftExecZ2Z(_cufftplan_tw, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_FORWARD);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing tau->w cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }

  if(applyscale)
  {
    double norm(1./double(_taugrid.getNTau()));
    gpu::scaleGPUvector(_data, norm, _nelem);
  }
#else
  // Check plan was created
  assert(_fftw_plan_tautow_many != 0);
  // Execute
  fftw_execute(_fftw_plan_tautow_many);
  if(applyscale)
  {
    double norm(1./double(_taugrid.getNTau()));
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= norm;
  }
#endif // __GPU__

  _intauspace = false;
}

template<typename T>
void CSfield<T>::fft_tautonegw(bool applyscale/*=true*/)
{
  if(!_intauspace)
    codeerror_abort((std::string("FFT tau->-w called on CSfield already in w space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_tau)
    codeerror_abort((std::string("FFT tau->-w called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);

  // Pretend we're already in w space to execute the correct direction of transform to effectively get f(-w)
  setInTauRepresentation(false);
  fft_wtotau(); // FFT w->tau => tau->-w
  setInTauRepresentation(false);

  if(applyscale)
  {
    double norm(1./double(_taugrid.getNTau()));
#ifdef __GPU__
    gpu::scaleGPUvector(_data, norm, _nelem);
#else
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= norm;
#endif // __GPU__
  }
}

template<typename T>
void CSfield<T>::fft_wtotau()
{
  if(_intauspace)
    codeerror_abort((std::string("FFT w->tau called on CSfield already in w space: ")+_label).c_str(),__FILE__,__LINE__);
  if(!_isFFTable_tau)
    codeerror_abort((std::string("FFT w->tau called on non-FFTable CSfield: ")+_label).c_str(),__FILE__,__LINE__);

#ifdef __GPU__
  assert(_cufftplan_tw != 0);
  cufftResult_t ierr = cufftExecZ2Z(_cufftplan_tw, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_INVERSE);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing w->tau cuFFT plan in CSfield " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }

#else
  // Check plan was created
  assert(_fftw_plan_wtotau_many != 0);
  fftw_execute(_fftw_plan_wtotau_many);
#endif

  _intauspace = true;
}

//====================================================================================
// Miscellaneous assignment operators (non-CSfield types, both directions)
template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator=(CSfield<T2> const &source)
{
#ifdef DEBUG
  if(source._nelem != _nelem)
    codeerror_abort("Invalid buffer size on CSfield operator=",__FILE__,__LINE__);
#endif

  _inrealspace = source.isInRealSpace();
  _intauspace  = source.isInTauRepresentation();

#ifdef __GPU__
  gpu::copyGPUvector_difftype(_data, source.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] = T(source.getDataPtr()[m]);

#endif // __GPU__

    return *this;
}

template<typename T>
CSfield<T> & CSfield<T>::operator=(std::vector<T> const &source)
{
#ifdef DEBUG
  if(source.size() != _nelem)
    codeerror_abort("Invalid buffer size on CSfield operator=",__FILE__,__LINE__);
#endif

#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(_data, source.data(), _nelem*sizeof(T), cudaMemcpyHostToDevice),__FILE__,__LINE__);
  //-----------------------------------------
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] = source[m];

#endif // __GPU__

    return *this;
}

template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator=(T2 source)
{
#ifdef __GPU__
  gpu::setvalueGPUvector(_data, source, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] = T(source);
#endif // __GPU__

    return *this;
}

template<typename T>
void CSfield<T>::copyDataToBuffer(std::vector<T> &buffer) const
{
  if(buffer.size() != _nelem)
    buffer.resize(_nelem);

#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(buffer.data(), _data, _nelem*sizeof(T), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      buffer[m] = _data[m];
#endif // __GPU__
}


template<typename T>
void CSfield<T>::copyDataFromBuffer(const T* buffer, size_t bufferelem)
{
  if(bufferelem != _nelem)
    codeerror_abort("Buffer is not the same size as field",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(_data, buffer, bufferelem*sizeof(T), cudaMemcpyHostToDevice),__FILE__,__LINE__);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<bufferelem; m++)
      _data[m] = buffer[m];
#endif // __GPU__

}



template<typename T>
T CSfield<T>::getElement(size_t spaceindx, size_t tauindx) const
{
  size_t indx(tauindx*_spacegrid.getNSpatial() + spaceindx);
  T result;
#ifdef __GPU__
  iobase::cdbg << "WARNING:- setting / getting individual CSfield elements on CUDA device is inefficient" << std::endl;
  gpu::cudaErrChk(cudaMemcpy(&result, _data+indx, sizeof(T), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
#else
  result = _data[indx];
#endif // __GPU__
  return result;
}


template<typename T>
void CSfield<T>::zero()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  int zero(0);
  gpu::cudaErrChk(cudaMemset(_data, zero, _nelem*sizeof(T)),__FILE__,__LINE__);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = T();
  //memset(_data, 0, sizeof(T)*_nelem);
#endif // __GPU__
}

template<>
void CSfield<double>::zerorealpart()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  int zero(0);
  gpu::cudaErrChk(cudaMemset(_data, zero, _nelem*sizeof(double)),__FILE__,__LINE__);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = 0.;
#endif // __GPU__
}

template<>
void CSfield<std::complex<double>>::zerorealpart()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  gpu::zerorealpt_GPUvector(_data, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m].real(0.);
#endif // __GPU__
}

template<>
void CSfield<double>::zeroimagpart()
{
  // No imaginary part; do nothing.
}

template<>
void CSfield<std::complex<double>>::zeroimagpart()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  gpu::zeroimagpt_GPUvector(_data, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m].imag(0.);
#endif // __GPU__
}


template<typename T>
void CSfield<T>::zeropos()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  gpu::zeroposRe_GPUvector(_data, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    if(std::real(_data[m]) > 0.)
      _data[m] = T();
#endif // __GPU__
}

template<typename T>
void CSfield<T>::zeroneg()
{
  if(_nelem == 0 || _data == nullptr)
    return;

#ifdef __GPU__
  gpu::zeronegRe_GPUvector(_data, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    if(std::real(_data[m]) < 0.)
      _data[m] = T();
#endif // __GPU__
}

template<typename T>
void CSfield<T>::setElement(T const &value, size_t spaceindx, size_t tauindx)
{
  size_t indx(tauindx*_spacegrid.getNSpatial() + spaceindx);
#ifdef __GPU__
  iobase::cdbg << "WARNING:- setting / getting individual CSfield elements on CUDA device is inefficient" << std::endl;
  gpu::cudaErrChk(cudaMemcpy(_data+indx, &value, sizeof(T), cudaMemcpyHostToDevice),__FILE__,__LINE__);
#else
  _data[indx] = value;
#endif // __GPU__
}

template<>
void CSfield<double>::fillRandomUniform(bool fillimaginary/*=false*/)
{
  random::instance().generateUniform(_data, _nelem);
}

template<>
void CSfield<std::complex<double>>::fillRandomUniform(bool fillimaginary/*=false*/)
{
  if(fillimaginary)
  {
    random::instance().generateUniform(reinterpret_cast<double*>(_data), 2*_nelem);
  } else {
#ifdef __GPU__
    // Require a tmp non-complex buffer
    // TODO: This is inefficient. We should instead fill the first half (contiguous) of the complex buffer
    // and then implement a thread-safe algorithm to scatter the data (in-place) into the real
    // parts of the complex-interleaved data.
    size_t nalloc = nextmult2(_nelem); // Bug in cuRAND requires even lengths.
    double *tmp = reinterpret_cast<double*>(gpu::cuda_memory_manager.allocate(nalloc*sizeof(double),"tmpnoise",__FILE__,__LINE__));
    random::instance().generateUniform(tmp,nalloc);
    gpu::copyGPUvector_difftype(_data, tmp, _nelem);
    gpu::cuda_memory_manager.deallocate(tmp,__FILE__,__LINE__);
#else
    std::vector<double> tmp(_nelem);
    random::instance().generateUniform(tmp.data(), _nelem);
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t i=0; i<_nelem; i++)
      _data[i] = tmp[i];
#endif // __GPU__
  }
}

template<>
void CSfield<double>::fillRandomNormal(bool fillimaginary/*=false*/)
{
  random::instance().generateNormal(_data, _nelem);
}

template<>
void CSfield<std::complex<double>>::fillRandomNormal(bool fillimaginary/*=false*/)
{
  if(fillimaginary)
  {
    random::instance().generateNormal(reinterpret_cast<double*>(_data), 2*_nelem);
  } else {
#ifdef __GPU__
    // Require a tmp non-complex buffer
    // TODO: This is inefficient. We should instead fill the first half (contiguous) of the complex buffer
    // and then implement a thread-safe algorithm to scatter the data (in-place) into the real
    // parts of the complex-interleaved data.
    size_t nalloc = nextmult2(_nelem); // Bug in cuRAND requires even lengths.
    double *tmp = reinterpret_cast<double*>(gpu::cuda_memory_manager.allocate(nalloc*sizeof(double),"tmpnoise",__FILE__,__LINE__));
    random::instance().generateNormal(tmp,nalloc);
    gpu::copyGPUvector_difftype(_data, tmp, _nelem);
    gpu::cuda_memory_manager.deallocate(tmp,__FILE__,__LINE__);
#else
    std::vector<double> tmp(_nelem);
    random::instance().generateNormal(tmp.data(), _nelem);
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t i=0; i<_nelem; i++)
      _data[i] = tmp[i];
#endif // __GPU__
  }
}

template<typename T>
void CSfield<T>::replaceTauSlice(field<T> const &in, size_t tauidx)
{
  // Since we will be replacing only one data slice, we must retain the same spatial Fourier representation.
  if(_inrealspace != in.isInRealSpace())
    codeerror_abort("Incompatible Fourier representations.",__FILE__,__LINE__);

  size_t M = _spacegrid.getNSpatial();
  size_t offset = M * tauidx;

  // Check that the grids are compatible
  if(M != in.getGrid().getNSpatial() || M != in._nelem)
    codeerror_abort("Incompatible spatial meshes.",__FILE__,__LINE__);

  if(tauidx >= _taugrid.getNTau())
    codeerror_abort("Error: invalid tau value",__FILE__,__LINE__);

  // Replace the data in the specified tau slice
#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(_data+offset, in.getDataPtr(), M*sizeof(T), cudaMemcpyDeviceToDevice),__FILE__,__LINE__);
#else

  T* dataoffset = _data+offset;
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<M; m++)
    dataoffset[m] = in.getDataPtr()[m];
#endif // __GPU__
}

template<typename T>
void CSfield<T>::replaceTauSlice(T in, size_t tauidx)
{
  size_t M = _spacegrid.getNSpatial();
  size_t offset = M * tauidx;
  T* dataoffset = _data+offset;

  if(tauidx >= _taugrid.getNTau())
    codeerror_abort("Error: invalid tau value",__FILE__,__LINE__);

  // Replace the data in the specified tau slice
#ifdef __GPU__
  gpu::setvalueGPUvector(dataoffset, in, M);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<M; m++)
    dataoffset[m] = in;
#endif // __GPU__
}

template<typename T>
void CSfield<T>::fetchTauSlice(field<T> &out, size_t tauidx) const
{
  out.setInRealSpace(_inrealspace);

  size_t M = _spacegrid.getNSpatial();
  size_t offset = M * tauidx;

  // Check that the grids are compatible
  if(M != out.getGrid().getNSpatial() || M != out._nelem)
    codeerror_abort("Incompatible spatial meshes.",__FILE__,__LINE__);

  if(tauidx >= _taugrid.getNTau())
    codeerror_abort("Error: invalid tau value",__FILE__,__LINE__);

  // Replace the data in the specified tau slice
#ifdef __GPU__
  gpu::cudaErrChk(cudaMemcpy(out.getDataPtr(), _data+offset, M*sizeof(T), cudaMemcpyDeviceToDevice),__FILE__,__LINE__);
#else

  T* dataoffset = _data+offset;
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<M; m++)
    out.getDataPtr()[m] = dataoffset[m];
#endif // __GPU__
}


//====================================================================================
// CSfield compound-assignment arithmetic operators
//
template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator*=(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::directprodGPUvector(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= rhs.getDataPtr()[m];
#endif // __GPU__

  return *this;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator*=(T2 rhs)
{
#ifdef __GPU__
  gpu::scaleGPUvector(_data, rhs, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= rhs;
#endif // __GPU__

  return *this;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator+=(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::sumGPUvector(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] += rhs.getDataPtr()[m];
#endif // __GPU__

  return *this;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator+=(T2 rhs)
{
#ifdef __GPU__
  gpu::addconstantGPUvector(_data, rhs, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] += rhs;
#endif // __GPU__

  return *this;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator-=(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::subtractGPUvector(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] -= rhs.getDataPtr()[m];
#endif // __GPU__

  return *this;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::operator/=(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::divideGPUvector(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] /= rhs.getDataPtr()[m];
#endif // __GPU__

  return *this;
}

//====================================================================================
// Mathematical Operations
template<typename T>
void CSfield<T>::sqrt(CSfield<T> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::sqrtGPUvector(_data, in.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = std::sqrt(in.getDataPtr()[m]);
#endif // __GPU__
}

template<typename T>
void CSfield<T>::exponentiate(CSfield<T> const &in, T scale)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::exponentiateGPUvectorscaled(_data, in.getDataPtr(), scale, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = exp(scale * in.getDataPtr()[m]);
#endif // __GPU__
}

template<typename T>
void CSfield<T>::exponentiate(CSfield<T> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::exponentiateGPUvector(_data, in.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = exp(in.getDataPtr()[m]);
#endif // __GPU__
}


template<>
void CSfield<std::complex<double>>::complexconjugate(CSfield<std::complex<double>> const &in)
{
  _inrealspace = in.isInRealSpace();
  _intauspace  = in.isInTauRepresentation();

  if(_nelem != in._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::conjg_GPUvector(_data, in.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = std::conj(in.getDataPtr()[m]);
#endif // __GPU__
}

template<>
void CSfield<double>::complexconjugate(CSfield<double> const &in)
{
  _inrealspace = in.isInRealSpace();
  _intauspace  = in.isInTauRepresentation();

  if(_nelem != in._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  // Since the data is non-complex, this reduces to a copy operation
  gpu::cudaErrChk(cudaMemcpy(_data, in.getDataPtr(), sizeof(double)*_nelem, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = in.getDataPtr()[m];
#endif // __GPU__
}

template<>
void CSfield<std::complex<double>>::complexconjugate_inplace()
{
#ifdef __GPU__
  gpu::conjg_GPUvector(_data, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
      for(size_t m = 0 ; m<_nelem ; m++)
        _data[m] = std::conj(_data[m]);
#endif // __GPU__
}

template<>
void CSfield<double>::complexconjugate_inplace()
{
  // Identity op.
  return;
}


template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::multiplyconjugated(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::directprodGPUvector_conjg(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] *= std::conj(rhs.getDataPtr()[m]);
#endif // __GPU__

  return *this;
}

template<typename T>
template<typename T2>
CSfield<T> & CSfield<T>::addconjugated(CSfield<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("CSField operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::sumGPUvector_conjg(_data, rhs.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<_nelem; m++)
      _data[m] += std::conj(rhs.getDataPtr()[m]);
#endif // __GPU__

  return *this;
}

template<typename T>
void CSfield<T>::accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T> const &in2, T coef)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), coef, _nelem);
#else

  T tmp;
#ifdef __OMP__
#pragma omp parallel for default(shared) private(tmp)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    tmp = in1.getDataPtr()[m] * in2.getDataPtr()[m];
    _data[m] += coef*tmp;
  }

#endif // __GPU__
}

template<typename T>
void CSfield<T>::accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T> const &in2)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] += in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}

template<typename T>
template<typename T2>
void CSfield<T>::accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T2> const &in2, T2 coef)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), coef, _nelem);
#else

  T tmp;
#ifdef __OMP__
#pragma omp parallel for default(shared) private(tmp)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    tmp = in1.getDataPtr()[m] * in2.getDataPtr()[m];
    _data[m] += coef*tmp;
  }

#endif // __GPU__
}

template<typename T>
template<typename T2>
void CSfield<T>::accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T2> const &in2)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in CSfield accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] += in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}


template<typename T>
template<typename T2>
void CSfield<T>::axpby_inplace(CSfield<T> const &y, T2 a, T2 b)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in CSfield axpby operator",__FILE__,__LINE__);
  if(_intauspace != y._intauspace)
    codeerror_abort("Inconsistent representations in CSfield axpby operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::axpbyGPUvector(_data, y.getDataPtr(), a, b, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    _data[m] *= a;
    _data[m] += y.getDataPtr()[m] * b;
  }

#endif // __GPU__
}

template<typename T>
template<typename T2>
void CSfield<T>::xpby_inplace(CSfield<T> const &y, T2 b)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in CSfield xpby operator",__FILE__,__LINE__);
  if(_intauspace != y._intauspace)
    codeerror_abort("Inconsistent representations in CSfield xpby operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::xpbyGPUvector(_data, y.getDataPtr(), b, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    _data[m] += y.getDataPtr()[m] * b;
  }

#endif // __GPU__
}

template<typename T>
template<typename T2>
void CSfield<T>::axpy_inplace(CSfield<T> const &y, T2 a)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in CSfield axpy operator",__FILE__,__LINE__);
  if(_intauspace != y._intauspace)
    codeerror_abort("Inconsistent representations in CSfield axpy operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("CSField operands have different sizes",__FILE__,__LINE__);

#ifdef __GPU__
  gpu::axpyGPUvector(_data, y.getDataPtr(), a, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    _data[m] *= a;
    _data[m] += y.getDataPtr()[m];
  }

#endif // __GPU__
}


template<typename T>
T CSfield<T>::integrate_r_intensive_oneslice(size_t sliceidx) const
{
  T result = T();
  size_t M = _spacegrid.getNSpatial();
  size_t offset = M*sliceidx;

#ifdef __GPU__
  if(_inrealspace)
  {
    result = gpu::sumelements(_data+offset, M);
    result *= 1./T(M);
  }
  else
  {
    cudaMemcpy(&result, _data, sizeof(T), cudaMemcpyDeviceToHost);
  }
  //-----------------------------------------

#else


  if(_inrealspace)
  {
    size_t end   = (sliceidx+1)*M;

#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
    {
      T tmp = T();

      // Use nowait to avoid implicit barrier at end of loop:
      // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
      for(size_t m = offset ; m < end; m++)
        tmp += _data[m];

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
      result += tmp;
    }

    result *= 1./T(M);
  }
  else
  {
    result = _data[sliceidx*_spacegrid.getNSpatial()]; // k=0 mode; Fourier modes are intensive already
  }
#endif // __GPU__

  return result;
}

template<typename T>
std::vector<T> CSfield<T>::integrate_r_intensive_allslices() const
{
  std::vector<T> result(_taugrid.getNTau());

  for(size_t itau = 0; itau<_taugrid.getNTau(); itau++)
    result[itau] = integrate_r_intensive_oneslice(itau);

  return result; // Stdlib move semantics or compiler RVO employed
}


template<typename T>
void CSfield<T>::integrate_tau_intensive(field<T> &result) const
{
  result.zero();
  result.setInRealSpace(_inrealspace);
  size_t M = _spacegrid.getNSpatial();
  size_t ntau = _taugrid.getNTau();
  double invntau = 1./double(ntau);

  // Sum over tau slices with 1/ntau factor included

#ifdef __GPU__
  for(size_t itau=0; itau<ntau; itau++)
    gpu::xpbyGPUvector(result.getDataPtr(), _data + M*itau, invntau, M);
#else

  for(size_t itau=0; itau<ntau; itau++)
  {
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
    for(size_t m=0; m<M; m++)
    {
      result.getDataPtr()[m] += _data[m + M*itau] * invntau;
    }
  }
#endif // __GPU__
}

template<typename T>
T CSfield<T>::integrate_tau_intensive_onepoint(size_t spaceidx) const
{
  T result{0.};
  size_t M = _spacegrid.getNSpatial();
  assert(spaceidx < M);
  size_t ntau = _taugrid.getNTau();
  double invntau = 1./double(ntau);

  // Sum over tau slices with 1/ntau factor included

#ifdef __GPU__
  T tmp;
  for(size_t itau=0; itau<ntau; itau++)
  {
    cudaMemcpy(&tmp, _data + M*itau + spaceidx, sizeof(T), cudaMemcpyDeviceToHost);
    result += tmp*invntau;
  }
#else
#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t itau=0; itau<ntau; itau++)
  {
    result += _data[spaceidx + M*itau] * invntau;
  }
#endif // __GPU__

  return result;
}



template<typename T>
T CSfield<T>::l1norm() const
{
  T result = T();

#ifdef __GPU__
  gpu::l1norm(_data, result, _nelem);
  //-----------------------------------------

#else


#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
  {
    T tmp = T();

    // Use nowait to avoid implicit barrier at end of loop:
    // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
    for(size_t m = 0 ; m<_nelem ; m++)
      tmp += std::abs(_data[m]);

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
    result += tmp;
  }
#endif // __GPU__

  return result;
}


template<typename T>
T CSfield<T>::l2norm() const
{
  T result = T();

#ifdef __GPU__
  gpu::l2norm(_data, result, _nelem);
  //-----------------------------------------

#else


#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
  {
    T tmp = T();

    // Use nowait to avoid implicit barrier at end of loop:
    // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
    for(size_t m = 0 ; m<_nelem ; m++)
      tmp += std::abs(_data[m]) * std::abs(_data[m]);

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
    result += tmp;
  }
#endif // __GPU__

  return result;
}



template<typename T>
T CSfield<T>::maxabs() const
{
  T result = T();

#ifdef __GPU__
  gpu::max(_data, result, _nelem);
  //-----------------------------------------

#else


#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
  {
    T tmp = T();

    // Use nowait to avoid implicit barrier at end of loop:
    // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
    for(size_t m = 0 ; m<_nelem ; m++)
      if(std::abs(tmp) < std::abs(_data[m]))
        tmp = _data[m];

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
    if(std::abs(result) < std::abs(tmp))
      result = tmp;
  }
#endif // __GPU__

  return result;
}


template<typename T>
T CSfield<T>::maxreal() const
{
  T result = T(std::numeric_limits<double>::min());

#ifdef __GPU__
  gpu::maxsigned(_data, result, _nelem);
  //-----------------------------------------

#else


#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
  {
    T tmp = T(std::numeric_limits<double>::min());

    // Use nowait to avoid implicit barrier at end of loop:
    // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
    for(size_t m = 0 ; m<_nelem ; m++)
      if(std::real(tmp) < std::real(_data[m]))
        tmp = _data[m];

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
    if(std::real(result) < std::real(tmp))
      result = tmp;
  }
#endif // __GPU__

  return result;
}


template<typename T>
T CSfield<T>::minreal() const
{
  T result = T(std::numeric_limits<double>::max());

#ifdef __GPU__
  gpu::minsigned(_data, result, _nelem);
  //-----------------------------------------

#else


#ifdef __OMP__
#pragma omp parallel default(shared)
#endif // __OMP__
  {
    T tmp = T(std::numeric_limits<double>::max());

    // Use nowait to avoid implicit barrier at end of loop:
    // Don't block threads before a synchronous update
#ifdef __OMP__
#pragma omp for nowait
#endif // __OMP__
    for(size_t m = 0 ; m<_nelem ; m++)
      if(std::real(tmp) > std::real(_data[m]))
        tmp = _data[m];

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
    if(std::real(result) > std::real(tmp))
      result = tmp;
  }
#endif // __GPU__

  return result;
}



// Comparison operator
template<typename T>
bool CSfield<T>::operator==(CSfield<T> const &rhs) const
{
  if(_nelem != rhs._nelem)
    return false;

  if(_inrealspace != rhs._inrealspace)
    return false;

  if(_intauspace != rhs._intauspace)
    return false;

#ifdef __GPU__
  //--- TODO: REPLACE WITH KERNEL LAUNCH ---
//  std::cout << "GPU INEFFICIENCY TO BE REMOVED " << __FILE__ << " " << __LINE__ << std::endl;
  std::vector<T> host_L(_nelem);
  std::vector<T> host_R(_nelem);
  cudaMemcpy(host_L.data(), _data, _nelem*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_R.data(), rhs.getDataPtr(), _nelem*sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t m=0; m<_nelem; m++)
    if(!CompareEqual(host_L[m],host_R[m]))
      return false;
  //-----------------------------------------
#else
  for(size_t m=0; m<_nelem; m++)
    if(!CompareEqual(_data[m],rhs.getDataPtr()[m]))
      return false;
#endif

  return true;
}


// Template specialization stubs
template class CSfield<std::complex<double>>;
template class CSfield<double>;
// Template member function specializations
// * operator=
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator=(CSfield<double> const &);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator=(std::complex<double>);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator=(double);
//template CSfield<double> & CSfield<double>::operator=(std::complex<double>); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator=(double);
//
// * operator*= CSfield
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator*=(CSfield<std::complex<double>> const &);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator*=(CSfield<double> const &);
//template CSfield<double> & CSfield<double>::operator*=(CSfield<std::complex<double>> const &); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator*=(CSfield<double> const &);
//
// * operator*= const
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator*=(std::complex<double>);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator*=(double);
//template CSfield<double> & CSfield<double>::operator*=(std::complex<double>); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator*=(double);
//
// * operator+= CSfield
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator+=(CSfield<std::complex<double>> const &);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator+=(CSfield<double> const &);
//template CSfield<double> & CSfield<double>::operator+=(CSfield<std::complex<double>> const &); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator+=(CSfield<double> const &);
//
// * operator+= const
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator+=(std::complex<double>);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator+=(double);
//template CSfield<double> & CSfield<double>::operator+=(std::complex<double>); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator+=(double);
//
// * operator-= CSfield
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator-=(CSfield<std::complex<double>> const &);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator-=(CSfield<double> const &);
//template CSfield<double> & CSfield<double>::operator-=(CSfield<std::complex<double>> const &); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator-=(CSfield<double> const &);
//
// * operator/= CSfield
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator/=(CSfield<std::complex<double>> const &);
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::operator/=(CSfield<double> const &);
//template CSfield<double> & CSfield<double>::operator/=(CSfield<std::complex<double>> const &); // Need a specialization for this
template CSfield<double> & CSfield<double>::operator/=(CSfield<double> const &);
//
// * multiplyconjugated
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::multiplyconjugated(CSfield<std::complex<double>> const &);
//
// * addconjugated
template CSfield<std::complex<double>> & CSfield<std::complex<double>>::addconjugated(CSfield<std::complex<double>> const &);
//
// * axpby-type functions - there is currently no GPU offload implementation for complex coefficients - add before uncommenting disabled entries below
//template void CSfield<std::complex<double>>::axpby_inplace(CSfield<std::complex<double>> const &, std::complex<double>, std::complex<double>);
template void CSfield<std::complex<double>>::axpby_inplace(CSfield<std::complex<double>> const &, double, double);
//template void CSfield<std::complex<double>>::xpby_inplace(CSfield<std::complex<double>> const &, std::complex<double>);
template void CSfield<std::complex<double>>::xpby_inplace(CSfield<std::complex<double>> const &, double);
//template void CSfield<std::complex<double>>::axpy_inplace(CSfield<std::complex<double>> const &, std::complex<double>);
template void CSfield<std::complex<double>>::axpy_inplace(CSfield<std::complex<double>> const &, double);

