#ifndef _CSFIELD_H_
#define _CSFIELD_H_

#include "spacegrid.h"
#include "taugrid.h"

// FFT library
#ifdef __GPU__
#include <cufft.h>
#else
#include <fftw3.h>
#ifdef __OMP__
#include <omp.h>
#endif // __OMP__
#endif // __GPU__


// Forward declaration
template<typename T> class field;

// A class for CSfields in d+1 dimensions with d spatial dimensions.
// Elements are of datatype T.
// This class manages its own memory and implements
// "the rule of 5"
// - Destructor
// - Copy constructor
// - Copy assignment operator
// - Move constructor
// - Move assignment operator
template<typename T>
class CSfield
{
  public:
    // ctor
    CSfield(SpaceGrid const & spacegrid, TauGrid const & taugrid, bool FFTable_space=true, bool InRealSpace=true, bool FFTable_tau=true, bool InTauSpace=true, std::string label="unnamed");
    // dtor
    virtual ~CSfield();
    // copy ctor
    CSfield(CSfield<T> const &source);
    // move ctor
    CSfield(CSfield<T>&& source) noexcept;
    // copy assignment
    CSfield<T> & operator=(CSfield<T> const &source);
    // move assignment
    CSfield<T> & operator=(CSfield<T>&& source) noexcept;

    // Getters
    bool isInRealSpace() const {return _inrealspace;};
    bool isInTauRepresentation() const {return _intauspace;};
    SpaceGrid const & getSpaceGrid() const {return _spacegrid;};
    TauGrid const & getTauGrid() const {return _taugrid;};
    void copyDataToBuffer(std::vector<T> &buffer) const;
    void copyDataFromBuffer(const T* buffer, size_t bufferelem);
    T getElement(size_t spaceindx, size_t tauindx) const;
    std::string const & getLabel() const {return _label;};
    // Setters
    void setInRealSpace(bool inrealspace) {_inrealspace = inrealspace;};
    void setInTauRepresentation(bool intauspace) {_intauspace = intauspace;};
    void zero();
    void zerorealpart();
    void zeroimagpart();
    void zeropos(); /// Set elements that are positive (real part) to zero
    void zeroneg(); /// Set elements that are positive (real part) to zero
    void setElement(T const &value, size_t spaceindx, size_t tauindx);
    void setLabel(std::string const &label) {_label = label;};
    void fillRandomUniform(bool fillimpart = false);
    void fillRandomNormal(bool fillimpart = false);
    void replaceTauSlice(field<T> const &in, size_t tauidx);
    void replaceTauSlice(T in, size_t tauidx); // Replace tau slice with a position-independent constant
    void fetchTauSlice(field<T> &out, size_t tauidx) const;

    // Data manipulation
    void fft_rtok(bool applyscale=true);
    void fft_rtonegk(bool applyscale=true);
    void fft_ktor();
    void fft_tautow(bool applyscale=true);
    void fft_tautonegw(bool applyscale=true);
    void fft_wtotau();

    // Operators
    CSfield<T> & operator=(std::vector<T> const &source);
    template<typename T2> CSfield<T> & operator=(CSfield<T2> const &source);
    template<typename T2> CSfield<T> & operator=(T2 source);
    template<typename T2> CSfield<T> & operator*=(CSfield<T2> const &rhs);
    template<typename T2> CSfield<T> & operator*=(T2 rhs);
    template<typename T2> CSfield<T> & operator+=(CSfield<T2> const &rhs);
    template<typename T2> CSfield<T> & operator+=(T2 rhs);
    template<typename T2> CSfield<T> & operator-=(CSfield<T2> const &rhs); // For scalar, use "+= -const"
    template<typename T2> CSfield<T> & operator/=(CSfield<T2> const &rhs); // For scalar, use "*= 1./const"

    // NOTE: binary arithmetic operators follow the class declaration as free functions

    // Mathematical operations
    void sqrt(CSfield<T> const &in);
    void exponentiate(CSfield<T> const &in, T scale);
    void exponentiate(CSfield<T> const &in);
    void complexconjugate(CSfield<T> const &in);
    void complexconjugate_inplace();
    template<typename T2> CSfield<T> & multiplyconjugated(CSfield<T2> const &rhs);
    template<typename T2> CSfield<T> & addconjugated(CSfield<T2> const &rhs);
    void accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T> const &in2, T coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    void accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    template<typename T2> void accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T2> const &in2, T2 coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    template<typename T2> void accumulateproduct_inplace(CSfield<T> const &in1, CSfield<T2> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    template<typename T2> void axpby_inplace(CSfield<T> const &y, T2 a, T2 b);
    template<typename T2> void xpby_inplace(CSfield<T> const &y, T2 b);
    template<typename T2> void axpy_inplace(CSfield<T> const &y, T2 a);
    // Reduction operations
    T integrate_r_intensive_oneslice(size_t sliceidx) const;
    std::vector<T> integrate_r_intensive_allslices() const;
    void integrate_tau_intensive(field<T> &result) const;
    T integrate_tau_intensive_onepoint(size_t spaceidx) const;

    T l1norm() const;
    T l2norm() const;
    T maxabs() const; // Return the element with the max abs value
    T maxreal() const; // Return the element with the max real part (signed)
    T minreal() const; // Return the element with the min real part (signed)

    // Comparison
    bool operator==(CSfield<T> const &rhs) const;
    inline bool operator!=(CSfield<T> const &rhs) const {return !(*this == rhs);};

  private:
    void CreateFFTPlans();
    void DestroyFFTPlans();
    T * getDataPtr() {return _data;}; /// Return raw data pointer; private, so only friends can use
    T const * getDataPtr() const {return _data;}; /// Return const raw data pointer; private, so only friends can use

    // Data members
    size_t      _nelem;
    T*          _data;
    std::string _label;

    SpaceGrid const &_spacegrid;
    TauGrid const &_taugrid;

    bool   _inrealspace;
    bool   _isFFTable_space;
    bool   _intauspace;
    bool   _isFFTable_tau;

#ifdef __GPU__
    cufftHandle _cufftplan_rk, _cufftplan_tw;
#else
    fftw_plan _fftw_plan_rtok_many, _fftw_plan_ktor_many;
    fftw_plan _fftw_plan_tautow_many, _fftw_plan_wtotau_many;
#endif

    template<typename T2> friend class CSfield; /// Allow CSfields to access each others private members for implementing type-asymmetric operators
    template<typename T2> friend class field;
};

// Binary operators: implemented as free functions so that we can exploit rvalue references for either operand
//
// CSfield * CSfield
template<typename T>
CSfield<T> operator*(CSfield<T> const &L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", both operands L-value references" << std::endl;
  CSfield<T> result(L);
  result *= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
CSfield<T> operator*(CSfield<T> &&L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result *= R;
  return result;
}

template<typename T>
CSfield<T> operator*(CSfield<T> const &L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(R));
  result *= L;
  return result;
}

template<typename T>
CSfield<T> operator*(CSfield<T> &&L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result *= R;
  return result;
}

// CSfield + CSfield
template<typename T>
CSfield<T> operator+(CSfield<T> const &L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary add: " << L.getLabel() << "+" << R.getLabel() << ", both operands L-value references" << std::endl;
  CSfield<T> result(L);
  result += R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
CSfield<T> operator+(CSfield<T> &&L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary add: " << L.getLabel() << "+" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result += R;
  return result;
}

template<typename T>
CSfield<T> operator+(CSfield<T> const &L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary add: " << L.getLabel() << "+" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(R));
  result += L;
  return result;
}

template<typename T>
CSfield<T> operator+(CSfield<T> &&L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary add: " << L.getLabel() << "+" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result += R;
  return result;
}

// CSfield - CSfield
template<typename T>
CSfield<T> operator-(CSfield<T> const &L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", both operands L-value references" << std::endl;
  CSfield<T> result(L);
  result -= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
CSfield<T> operator-(CSfield<T> &&L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result -= R;
  return result;
}

template<typename T>
CSfield<T> operator-(CSfield<T> const &L, CSfield<T> &&R)
{
  // BE CAREFUL WITH NON-COMMUTATIVITY for this implementation
  iobase::cdbg << "CSfield binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  // We can't MOVE CONSTRUCT from R and use the compound arithmetic to get the right commutativity. Use a copy construction instead.
  // TODO Alternative: re-implement element-wise division in place.
  CSfield<T> result(L);
  result -= R;
  return std::move(result);
}

template<typename T>
CSfield<T> operator-(CSfield<T> &&L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result -= R;
  return result;
}


// CSfield / CSfield
template<typename T>
CSfield<T> operator/(CSfield<T> const &L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary divide: " << L.getLabel() << "/" << R.getLabel() << ", both operands L-value references" << std::endl;
  CSfield<T> result(L);
  result /= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
CSfield<T> operator/(CSfield<T> &&L, CSfield<T> const &R)
{
  iobase::cdbg << "CSfield binary divide: " << L.getLabel() << "/" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result /= R;
  return result;
}

template<typename T>
CSfield<T> operator/(CSfield<T> const &L, CSfield<T> &&R)
{
  // BE CAREFUL WITH NON-COMMUTATIVITY for this implementation
  iobase::cdbg << "CSfield binary divide: " << L.getLabel() << "/" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  // We can't MOVE CONSTRUCT from R and use the compound arithmetic to get the right commutativity. Use a copy construction instead.
  // TODO Alternative: re-implement element-wise division in place.
  CSfield<T> result(L);
  result /= R;
  return std::move(result);
}

template<typename T>
CSfield<T> operator/(CSfield<T> &&L, CSfield<T> &&R)
{
  iobase::cdbg << "CSfield binary divide: " << L.getLabel() << "/" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  CSfield<T> result(std::move(L));
  result /= R;
  return result;
}

#endif // _CSFIELD_H_
