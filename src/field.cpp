#include "field.h"
#include "CSfield.h"
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
field<T>::field(SpaceGrid const &grid, bool FFTable/*=true*/, bool isInRealSpace/*=true*/, std::string label/*="unnamed"*/)
  : _nelem(grid.getNSpatial())
  , _data(nullptr)
  , _label(label)
  , _grid(grid)
  , _inrealspace(isInRealSpace)
  , _isFFTable(FFTable)
#ifdef __GPU__
  , _cufftplan(0)
#else
  , _fftw_plan_fwd(0)
  , _fftw_plan_bck(0)
#endif
{
  iobase::cdbg << "field " << _label << ": constructor" << std::endl;

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

  if(_isFFTable)
    CreateFFTPlans();
}

// Destructor
template<typename T>
field<T>::~field()
{
  iobase::cdbg << "field destructor: " << _label << std::endl;

  if(_isFFTable)
    DestroyFFTPlans();

#ifdef __GPU__
  gpu::cuda_memory_manager.deallocate(_data,__FILE__,__LINE__);
#else
  _mm_free(_data);
#endif
}

// Copy constructor
template<typename T>
field<T>::field(field<T> const &source)
  : field<T>(source._grid, source._isFFTable, source._inrealspace, source._label+"_copy") // malloc
{
  iobase::cdbg << "field " << _label << ": copy constructor from " << source._label << std::endl;

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
field<T> & field<T>::operator=(field<T> const &source)
{
  iobase::cdbg << "field: copy assignment from " << source._label << " to " << _label << std::endl;

  // Skip self assignment
  if(&source == this)
    return *this;

  // Note that the grids must be identical to copy fields - we can't unseat and replace the grid reference in "this".
  assert(&_grid == &(source._grid));

  // The following one-liner uses the copy constructor to create a temporary and then the move assignment operator to transfer it to this.
  //return *this = field<T>(source); // Use copy ctor code to make the temporary object

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
    if(_isFFTable)
      DestroyFFTPlans();
    gpu::cuda_memory_manager.deallocate(_data,__FILE__,__LINE__);
#else
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
    if(_isFFTable)
      DestroyFFTPlans();
    _mm_free(_data);
#endif // __GPU__

    // 3. Transfer the pointer
    _data = newdata;
    _nelem = source._nelem;

    // This container will not change its FFTability based on source's.
    // We can imagine copying data into a destination container precisely because it allows FFTs.
    // If this container is FFTable, we now need to re-create the FFTW plans since the storage pointer has moved.
    if(_isFFTable)
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

  return *this;
}

// Move constructor
template<typename T>
field<T>::field(field<T>&& tmp) noexcept
  : _nelem(tmp._nelem) // Do not delegate ctor - avoid malloc and transfer ownership instead
  , _data(tmp._data)
  , _label(tmp._label)
  , _grid(tmp._grid)
  , _inrealspace(tmp._inrealspace)
  , _isFFTable(tmp._isFFTable)
#ifdef __GPU__
  , _cufftplan(tmp._cufftplan)
#else
  , _fftw_plan_fwd(tmp._fftw_plan_fwd) // Since the storage location is from the tmp (and by definition has the same pointer, size and alignment), it is fine to transfer FFTW plan ownership.
  , _fftw_plan_bck(tmp._fftw_plan_bck)
#endif
{
  iobase::cdbg << "field " << _label << ": move constructor" << std::endl;

  // Leave tmp in ill-defined state
  tmp._data  = nullptr;
  tmp._nelem = 0;
  // The next three lines will prevent tmp's destructor from releasing the FFTW plans that have been claimed by ``this''
  tmp._isFFTable = false;
#ifdef __GPU__
  tmp._cufftplan = 0;
#else
  tmp._fftw_plan_fwd = 0;
  tmp._fftw_plan_bck = 0;
#endif
}

// Move assignment
template<typename T>
field<T> & field<T>::operator=(field<T>&& tmp) noexcept
{
  iobase::cdbg << "field: move assignment from " << tmp._label << " to " << _label << std::endl;

  // Skip self assignment
  if(&tmp == this)
    return *this;

  // Note that the grids must be identical to copy fields - we can't unseat and replace the grid reference in ``this''.
  assert(&_grid == &(tmp._grid));

  // Simply swap all elements to claim ownership of tmp's contents.
  // We need to clean up what was previously inside ``this'', but tmp's destructor will do that when
  // it goes out of scope (immediately after assignment is complete).
  std::swap(_nelem, tmp._nelem);
  std::swap(_data, tmp._data);
//  std::swap(_label, tmp._label); // We typically don't want to change the label of the object when copying. Can override with explicit setLabel() where needed.
  std::swap(_inrealspace, tmp._inrealspace);
  std::swap(_isFFTable, tmp._isFFTable);
#ifdef __GPU__
  std::swap(_cufftplan, tmp._cufftplan);
#else
  std::swap(_fftw_plan_fwd, tmp._fftw_plan_fwd);
  std::swap(_fftw_plan_bck, tmp._fftw_plan_bck);
#endif

  return *this;
}

//====================================================================================
// FFTs
template<>
void field<std::complex<double>>::CreateFFTPlans()
{
  iobase::cdbg << "Create FFT plans for field " << _label << std::endl;

  // Mesh dimension for plan creation
  int Nx[_grid.getSpatialDim()]; // 3D is the maximum
  for(int i=0; i<_grid.getSpatialDim(); i++)
    Nx[i] = _grid.getGridSize(i);

#ifdef __GPU__
  // Create cuFFT plan
  cufftResult_t ierr;
  switch(_grid.getSpatialDim())
  {
    case 1:
      ierr = cufftPlan1d(&_cufftplan, Nx[0], CUFFT_Z2Z, 1);
      break;
    case 2:
      ierr = cufftPlan2d(&_cufftplan, Nx[0], Nx[1], CUFFT_Z2Z);
      break;
    case 3:
      ierr = cufftPlan3d(&_cufftplan, Nx[0], Nx[1], Nx[2], CUFFT_Z2Z);
      break;
    default:
      codeerror_abort("Error creating CUDA FFT plan: dimension must be 1, 2, or 3",__FILE__,__LINE__);
  }
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error creating cuFFT plan in field " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }
#else
  // create FFTW plans
  fftw_complex* data_fftw = reinterpret_cast<fftw_complex*>(_data);
  _fftw_plan_fwd = fftw_plan_dft(int(_grid.getSpatialDim()), Nx, data_fftw, data_fftw, FFTW_FORWARD, FFTW_PATIENT);
  _fftw_plan_bck = fftw_plan_dft(int(_grid.getSpatialDim()), Nx, data_fftw, data_fftw, FFTW_BACKWARD, FFTW_PATIENT);
#endif
}

template<>
void field<double>::CreateFFTPlans()
{
  codeerror_abort("Real-valued fields are not currently FFTable",__FILE__,__LINE__);
}

template<typename T>
void field<T>::DestroyFFTPlans()
{
#ifdef __GPU__
  if(_cufftplan)
    cufftDestroy(_cufftplan);
#else
  if(_fftw_plan_fwd != 0)
    fftw_destroy_plan(_fftw_plan_fwd);
  if(_fftw_plan_bck != 0)
    fftw_destroy_plan(_fftw_plan_bck);
#endif
}

template<typename T>
void field<T>::fft_rtok(bool applyscale/*=true*/)
{
  if(!_inrealspace)
    codeerror_abort("FFT r->k called on field already in k space",__FILE__,__LINE__);
  if(!_isFFTable)
    codeerror_abort("FFT r->k called on non-FFTable field",__FILE__,__LINE__);

#ifdef __GPU__
  assert(_cufftplan != 0);

  cufftResult_t ierr = cufftExecZ2Z(_cufftplan, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_FORWARD);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing r->k cuFFT plan in field " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }

  if(applyscale)
  {
    double norm(1./double(_grid.getNSpatial()));
    gpu::scaleGPUvector(_data, norm, _nelem);
  }
#else
  // Check plan was created
  assert(_fftw_plan_fwd != 0);

  fftw_execute(_fftw_plan_fwd);
  if(applyscale)
  {
    double norm(1./double(_grid.getNSpatial()));
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
void field<T>::fft_rtonegk(bool applyscale/*=true*/)
{
  if(!_inrealspace)
    codeerror_abort("FFT r->-k called on field already in k space",__FILE__,__LINE__);
  if(!_isFFTable)
    codeerror_abort("FFT r->-k called on non-FFTable field",__FILE__,__LINE__);

  // Pretend we're already in k space to execute the correct direction of transform to effectively get -k
  setInRealSpace(false);
  fft_ktor(); // FFT k->r => r->-k
  setInRealSpace(false);

  if(applyscale)
  {
    double norm(1./double(_grid.getNSpatial()));
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
void field<T>::fft_ktor()
{
  if(_inrealspace)
    codeerror_abort("FFT k->r called on field already in r space",__FILE__,__LINE__);
  if(!_isFFTable)
    codeerror_abort("FFT k->r called on non-FFTable field",__FILE__,__LINE__);

#ifdef __GPU__
  // Check plan was created
  assert(_cufftplan != 0);
  // Execute
  cufftResult_t ierr = cufftExecZ2Z(_cufftplan, reinterpret_cast<cuDoubleComplex*>(_data),
                                    reinterpret_cast<cuDoubleComplex*>(_data), CUFFT_INVERSE);
  if(ierr != CUFFT_SUCCESS)
  {
    std::stringstream msg;
    msg << "Error executing k->r cuFFT plan in field " << _label << " ERROR = " << ierr;
    codeerror_abort(msg.str().c_str(), __FILE__, __LINE__);
  }
#else
  // Check plan was created
  assert(_fftw_plan_bck != 0);
  // Execute
  fftw_execute(_fftw_plan_bck);
#endif // __GPU__

  _inrealspace = true;
}

//====================================================================================
// Miscellaneous assignment operators (non-field types or fields with different data kinds)
template<typename T>
template<typename T2>
field<T> & field<T>::operator=(field<T2> const &source)
{
#ifdef DEBUG
  if(source._nelem != _nelem)
    codeerror_abort("Invalid buffer size on field operator=",__FILE__,__LINE__);
#endif

  _inrealspace = source.isInRealSpace();

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
field<T> & field<T>::operator=(std::vector<T> const &source)
{
#ifdef DEBUG
  if(source.size() != _nelem)
    codeerror_abort("Invalid buffer size on field operator=",__FILE__,__LINE__);
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
field<T> & field<T>::operator=(T2 source)
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
void field<T>::copyDataToBuffer(std::vector<T> &buffer) const
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
void field<T>::copyDataFromBuffer(const T* buffer, size_t bufferelem)
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
T field<T>::getElement(size_t indx) const
{
  T result;
#ifdef __GPU__
  iobase::cdbg << "WARNING:- setting / getting individual field elements on CUDA device is inefficient" << std::endl;
  gpu::cudaErrChk(cudaMemcpy(&result, _data+indx, sizeof(T), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
#else
  result = _data[indx];
#endif
  return result;
}


template<typename T>
void field<T>::zero()
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
void field<double>::zerorealpart()
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
void field<std::complex<double>>::zerorealpart()
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
void field<double>::zeroimagpart()
{
  // No imaginary part; do nothing.
}

template<>
void field<std::complex<double>>::zeroimagpart()
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
void field<T>::zeropos()
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
void field<T>::zeroneg()
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
void field<T>::setElement(T const &value, size_t indx)
{
#ifdef __GPU__
  iobase::cdbg << "WARNING:- setting / getting individual field elements on CUDA device is inefficient" << std::endl;
  gpu::cudaErrChk(cudaMemcpy(_data+indx, &value, sizeof(T), cudaMemcpyHostToDevice),__FILE__,__LINE__);
#else
  _data[indx] = value;
#endif // __GPU__
}

template<>
void field<double>::fillRandomUniform(bool fillimaginary/*=false*/)
{
  random::instance().generateUniform(_data, _nelem);
}

template<>
void field<std::complex<double>>::fillRandomUniform(bool fillimaginary/*=false*/)
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
void field<double>::fillRandomNormal(bool fillimaginary/*=false*/)
{
  random::instance().generateNormal(_data, _nelem);
}

template<>
void field<std::complex<double>>::fillRandomNormal(bool fillimaginary/*=false*/)
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

//====================================================================================
// field compound-assignment arithmetic operators
//
template<typename T>
template<typename T2>
field<T> & field<T>::operator*=(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
field<T> & field<T>::operator*=(T2 rhs)
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
field<T> & field<T>::operator+=(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
field<T> & field<T>::operator+=(T2 rhs)
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
field<T> & field<T>::operator-=(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
field<T> & field<T>::operator/=(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::sqrt(field<T> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::exponentiate(field<T> const &in, T scale)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::exponentiate(field<T> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<std::complex<double>>::complexconjugate(field<std::complex<double>> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<double>::complexconjugate(field<double> const &in)
{
  _inrealspace = in.isInRealSpace();

  if(_nelem != in._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<std::complex<double>>::complexconjugate_inplace()
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
void field<double>::complexconjugate_inplace()
{
  // Identity op.
  return;
}


template<typename T>
template<typename T2>
field<T> & field<T>::multiplyconjugated(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
field<T> & field<T>::addconjugated(field<T2> const &rhs)
{
  if(_inrealspace != rhs._inrealspace)
    codeerror_abort("Field operands in different Fourier representations",__FILE__,__LINE__);

  if(_nelem != rhs._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::accumulateproduct_inplace(field<T> const &in1, field<T> const &in2, T coef)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::accumulateproduct_inplace(field<T> const &in1, field<T> const &in2)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::accumulateproduct_inplace(field<T> const &in1, field<T2> const &in2, T2 coef)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::accumulateproduct_inplace(field<T> const &in1, field<T2> const &in2)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
void field<T>::accumulateproduct_inplace(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);

  size_t M = _grid.getNSpatial();
  // size_t ntau = _grid.getNTau();

  assert(_nelem == M);
  // assert(in1._nelem == M*ntau);
  // assert(in2._nelem == M*ntau);

  size_t offset1 = M*tauidx1;
  size_t offset2 = M*tauidx2;

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr()+offset1, in2.getDataPtr()+offset2, _nelem);
#else

  T tmp;
#ifdef __OMP__
#pragma omp parallel for default(shared) private(tmp)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    tmp = in1.getDataPtr()[m+offset1] * in2.getDataPtr()[m+offset2];
    _data[m] += tmp;
  }

#endif // __GPU__
}

template<typename T>
void field<T>::accumulateproduct_inplace(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2, T coef)
{
  if(_inrealspace != in1._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);
  if(_inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field accumulateproduct operator",__FILE__,__LINE__);

  size_t M = _grid.getNSpatial();
  // size_t ntau = _grid.getNTau();

  assert(_nelem == M);
  // assert(in1._nelem == M*ntau);
  // assert(in2._nelem == M*ntau);

  size_t offset1 = M*tauidx1;
  size_t offset2 = M*tauidx2;

#ifdef __GPU__
  gpu::accumulateproduct_GPUvector(_data, in1.getDataPtr()+offset1, in2.getDataPtr()+offset2, coef, _nelem);
#else

  T tmp;
#ifdef __OMP__
#pragma omp parallel for default(shared) private(tmp)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
  {
    tmp = in1.getDataPtr()[m+offset1] * in2.getDataPtr()[m+offset2];
    _data[m] += coef * tmp;
  }

#endif // __GPU__
}

template<typename T>
void field<T>::settoproduct(field<T> const &in1, field<T> const &in2, T coef)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

  _inrealspace = in1._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), coef, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = coef * in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}

template<typename T>
void field<T>::settoproduct(field<T> const &in1, field<T> const &in2)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

  _inrealspace = in1._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}

template<typename T>
template<typename T2>
void field<T>::settoproduct(field<T> const &in1, field<T2> const &in2, T2 coef)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

  _inrealspace = in1._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), coef, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = coef*in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}

template<typename T>
template<typename T2>
void field<T>::settoproduct(field<T> const &in1, field<T2> const &in2)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);
  if(_nelem != in1._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);
  if(_nelem != in2._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

  _inrealspace = in1._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr(), in2.getDataPtr(), _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = in1.getDataPtr()[m] * in2.getDataPtr()[m];

#endif // __GPU__
}

template<typename T>
void field<T>::settoproduct(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);

  size_t M = _grid.getNSpatial();
  assert(_nelem == M);

  size_t offset1 = M*tauidx1;
  size_t offset2 = M*tauidx2;

  _inrealspace = in1._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr()+offset1, in2.getDataPtr()+offset2, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = in1.getDataPtr()[m+offset1] * in2.getDataPtr()[m+offset2];

#endif // __GPU__
}

template<typename T>
void field<T>::settoproduct(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2, T coef)
{
  if(in1._inrealspace != in2._inrealspace)
    codeerror_abort("Inconsistent Fourier representations in field settoproduct operator",__FILE__,__LINE__);

  size_t M = _grid.getNSpatial();
  assert(_nelem == M);

  size_t offset1 = M*tauidx1;
  size_t offset2 = M*tauidx2;

  _inrealspace = in2._inrealspace;

#ifdef __GPU__
  gpu::settoproduct_GPUvector(_data, in1.getDataPtr()+offset1, in2.getDataPtr()+offset2, coef, _nelem);
#else

#ifdef __OMP__
#pragma omp parallel for default(shared)
#endif // __OMP__
  for(size_t m=0; m<_nelem; m++)
    _data[m] = coef * in1.getDataPtr()[m+offset1] * in2.getDataPtr()[m+offset2];

#endif // __GPU__
}

template<typename T>
template<typename T2, typename T3>
void field<T>::axpby_inplace(field<T2> const &y, T3 a, T3 b)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in field axpby operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
template<typename T2, typename T3>
void field<T>::xpby_inplace(field<T2> const &y, T3 b)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in field xpby operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
template<typename T2, typename T3>
void field<T>::axpy_inplace(field<T2> const &y, T3 a)
{
  if(_inrealspace != y._inrealspace)
    codeerror_abort("Inconsistent representations in field axpy operator",__FILE__,__LINE__);
  if(_nelem != y._nelem)
    codeerror_abort("Field operands have different sizes",__FILE__,__LINE__);

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
T field<T>::integrate_r_intensive() const
{
  T result = T();

#ifdef __GPU__
  if(_inrealspace)
  {
    result = gpu::sumelements(_data, _nelem);
    result *= 1./T(_nelem);
  }
  else
  {
    cudaMemcpy(&result, _data, sizeof(T), cudaMemcpyDeviceToHost);
  }
  //-----------------------------------------

#else


  if(_inrealspace)
  {
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
        tmp += _data[m];

      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
#ifdef __OMP__
#pragma omp critical
#endif // __OMP__
      result += tmp;
    }

    result *= 1./T(_nelem);
  }
  else
  {
    result = _data[0]; // k=0 mode; Fourier modes are intensive already
  }
#endif // __GPU__

  return result;
}

template<typename T>
T field<T>::sum_elem() const
{
  T result = T();

#ifdef __GPU__
    result = gpu::sumelements(_data, _nelem);
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
      tmp += _data[m];

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
T field<T>::maxabs() const
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
T field<T>::maxreal() const
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
T field<T>::minreal() const
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


// copied from Kkeithley - unit test needed ---- ECM 5/2022 -- wouldn't compile 
 //template<typename T>
 //template<typename T2>
 //void field<T>::avg_arg(T2 &out) const 
 //{
 //#ifdef __GPU__
 //  // no GPU implementation - do nothing
 //  codeerror_abort("No GPU implementation for arg averaging", __FILE__, __LINE__);
 //#else
 //
 //  //TODO: parallel implementation
 //  // assert(_isinrealspace);
 //  double avg_arg = 0;
 //  double arg_tmp;
 //  for(size_t m = 0; m < _nelem; m++){
 //    arg_tmp = std::arg(_data[m]);
 //    if (arg_tmp < 0){
 //      arg_tmp += 2*PI;
 //     }
 //    avg_arg += arg_tmp;
 //   }
 //  avg_arg *= 1/double(_nelem);
 //  out = avg_arg;
 //
 //#endif //__GPU__
 //}


//template<typename T> // modified avg_arg() fxn, ECM 05/2022
//T field<T>::avg_arg() const
//{
//  T result = T();
//
//#ifdef __GPU__
//  // no GPU implementation - do nothing
//  codeerror_abort("No GPU implementation for arg averaging", __FILE__, __LINE__);
// //  if(_inrealspace)
// //  {
// //    result = gpu::sumelements(_data, _nelem);
// //    result *= 1./T(_nelem);
// //  }
// //  else
// //  {
// //    cudaMemcpy(&result, _data, sizeof(T), cudaMemcpyDeviceToHost);
// //  }
//  //-----------------------------------------
//
//#else
//
//
//  if(_inrealspace)
//  {
//#ifdef __OMP__
//#pragma omp parallel default(shared)
//#endif // __OMP__
////    {
//    T tmp = T();
//
//      // Use nowait to avoid implicit barrier at end of loop:
//      // Don't block threads before a synchronous update
//#ifdef __OMP__
//#pragma omp for nowait
//#endif // __OMP__
// //      for(size_t m = 0 ; m<_nelem ; m++)
// //        tmp += _data[m];
//    for(size_t m = 0; m < _nelem; m++)
//    {
//
//      if(std::abs(_data[m]) < 0.1)
//        tmp += 0.; // choose arbitrary phase of 0. for values close to zero  
//      else
//      {
//        if(std::arg(_data[m]) < 0)
//          tmp += (std::arg(_data[m]) + TPI);
//        else
//          tmp += std::arg(_data[m]);
//      }
//    }
//
//      // Update atomically using critical region (explicit atomics are faster but restrict types of op)
//#ifdef __OMP__
//#pragma omp critical
//#endif // __OMP__
//    result += tmp;
//   // }
//
//    result *= 1./T(_nelem);
//  }
//  else
//    codeerror_abort("Abort from argument average (field.cpp), field is not in real space!", __FILE__, __LINE__);
//
//#endif // __GPU__
//
//  return result;
//}









// Comparison operator
template<typename T>
bool field<T>::operator==(field<T> const &rhs) const
{
  if(_nelem != rhs._nelem)
    return false;

  if(_inrealspace != rhs._inrealspace)
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
template class field<std::complex<double>>;
template class field<double>;
// Template member function specializations
// * operator=
template field<std::complex<double>> & field<std::complex<double>>::operator=(field<double> const &);
template field<std::complex<double>> & field<std::complex<double>>::operator=(std::complex<double>);
template field<std::complex<double>> & field<std::complex<double>>::operator=(double);
//template field<double> & field<double>::operator=(std::complex<double>); // Need a specialization for this
template field<double> & field<double>::operator=(double);
//
// * operator*= field
template field<std::complex<double>> & field<std::complex<double>>::operator*=(field<std::complex<double>> const &);
template field<std::complex<double>> & field<std::complex<double>>::operator*=(field<double> const &);
//template field<double> & field<double>::operator*=(field<std::complex<double>> const &); // Need a specialization for this
template field<double> & field<double>::operator*=(field<double> const &);
//
// * operator*= const
template field<std::complex<double>> & field<std::complex<double>>::operator*=(std::complex<double>);
template field<std::complex<double>> & field<std::complex<double>>::operator*=(double);
//template field<double> & field<double>::operator*=(std::complex<double>); // Need a specialization for this
template field<double> & field<double>::operator*=(double);
//
// * operator+= field
template field<std::complex<double>> & field<std::complex<double>>::operator+=(field<std::complex<double>> const &);
template field<std::complex<double>> & field<std::complex<double>>::operator+=(field<double> const &);
//template field<double> & field<double>::operator+=(field<std::complex<double>> const &); // Need a specialization for this
template field<double> & field<double>::operator+=(field<double> const &);
//
// * operator+= const
template field<std::complex<double>> & field<std::complex<double>>::operator+=(std::complex<double>);
template field<std::complex<double>> & field<std::complex<double>>::operator+=(double);
//template field<double> & field<double>::operator+=(std::complex<double>); // Need a specialization for this
template field<double> & field<double>::operator+=(double);
//
// * operator-= field
template field<std::complex<double>> & field<std::complex<double>>::operator-=(field<std::complex<double>> const &);
template field<std::complex<double>> & field<std::complex<double>>::operator-=(field<double> const &);
//template field<double> & field<double>::operator-=(field<std::complex<double>> const &); // Need a specialization for this
template field<double> & field<double>::operator-=(field<double> const &);
//
// * operator/= field
template field<std::complex<double>> & field<std::complex<double>>::operator/=(field<std::complex<double>> const &);
template field<std::complex<double>> & field<std::complex<double>>::operator/=(field<double> const &);
//template field<double> & field<double>::operator/=(field<std::complex<double>> const &); // Need a specialization for this
template field<double> & field<double>::operator/=(field<double> const &);
//
// * multiplyconjugated
template field<std::complex<double>> & field<std::complex<double>>::multiplyconjugated(field<std::complex<double>> const &);
//
// * addconjugated
template field<std::complex<double>> & field<std::complex<double>>::addconjugated(field<std::complex<double>> const &);
//
// * axpby-type functions - there is currently no GPU offload implementation for complex coefficients - add before uncommenting disabled entries below
template void field<std::complex<double>>::axpby_inplace(field<std::complex<double>> const &, std::complex<double>, std::complex<double>);
template void field<std::complex<double>>::axpby_inplace(field<std::complex<double>> const &, double, double);
template void field<std::complex<double>>::xpby_inplace(field<std::complex<double>> const &, std::complex<double>);
template void field<std::complex<double>>::xpby_inplace(field<std::complex<double>> const &, double);
template void field<std::complex<double>>::axpy_inplace(field<std::complex<double>> const &, std::complex<double>);
template void field<std::complex<double>>::axpy_inplace(field<std::complex<double>> const &, double);
template void field<std::complex<double>>::axpby_inplace(field<double> const &, std::complex<double>, std::complex<double>);
template void field<std::complex<double>>::axpby_inplace(field<double> const &, double, double);
template void field<std::complex<double>>::xpby_inplace(field<double> const &, std::complex<double>);
template void field<std::complex<double>>::xpby_inplace(field<double> const &, double);
template void field<std::complex<double>>::axpy_inplace(field<double> const &, std::complex<double>);
template void field<std::complex<double>>::axpy_inplace(field<double> const &, double);
