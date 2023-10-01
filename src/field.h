#ifndef _FIELD_H_
#define _FIELD_H_

#include "spacegrid.h"

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
template<typename T> class CSfield;

// A class for fields in d spatial dimensions.
// Elements are of datatype T.
// This class manages its own memory and implements
// "the rule of 5"
// - Destructor
// - Copy constructor
// - Copy assignment operator
// - Move constructor
// - Move assignment operator
template<typename T>
class field
{
  public:
    // ctor
    field(SpaceGrid const & grid, bool FFTable=true, bool isInRealSpace=true, std::string label="unnamed");
    // dtor
    virtual ~field();
    // copy ctor
    explicit field(field<T> const &source);
    // move ctor
    field(field<T>&& tmp) noexcept;
    // copy assignment
    field<T> & operator=(field<T> const &source);
    // move assignment
    field<T> & operator=(field<T>&& tmp) noexcept;

    // Getters
    bool isInRealSpace() const {return _inrealspace;};
    SpaceGrid const & getGrid() const {return _grid;};
    void copyDataToBuffer(std::vector<T> &buffer) const;
    void copyDataFromBuffer(const T* buffer, size_t bufferelem);
    T getElement(size_t indx) const;
    std::string const & getLabel() const {return _label;};
    // Setters
    void setInRealSpace(bool inrealspace) {_inrealspace = inrealspace;};
    void zero();
    void zerorealpart();
    void zeroimagpart();
    void zeropos(); /// Set elements that are positive (real part) to zero
    void zeroneg(); /// Set elements that are negative (real part) to zero
    void setElement(T const &value, size_t indx);
    void setLabel(std::string const &label) {_label = label;};
    void fillRandomUniform(bool fillimpart = false);
    void fillRandomNormal(bool fillimpart = false);

    // Data manipulation
    void fft_rtok(bool applyscale=true);
    void fft_rtonegk(bool applyscale=true);
    void fft_ktor();

    // Operators
    field<T> & operator=(std::vector<T> const &source);
    template<typename T2> field<T> & operator=(field<T2> const &source); // Copy from field of different type
    template<typename T2> field<T> & operator=(T2 source);
    template<typename T2> field<T> & operator*=(field<T2> const &rhs);
    template<typename T2> field<T> & operator*=(T2 rhs);
    template<typename T2> field<T> & operator+=(field<T2> const &rhs);
    template<typename T2> field<T> & operator+=(T2 rhs);
    template<typename T2> field<T> & operator-=(field<T2> const &rhs);
    template<typename T2> field<T> & operator/=(field<T2> const &rhs);

    // NOTE: binary arithmetic operators follow the class declaration as free functions

    // Mathematical operations
    void sqrt(field<T> const &in);
    void exponentiate(field<T> const &in, T scale);
    void exponentiate(field<T> const &in);
    void complexconjugate(field<T> const &in);
    void complexconjugate_inplace();
    template<typename T2> field<T> & multiplyconjugated(field<T2> const &rhs);
    template<typename T2> field<T> & addconjugated(field<T2> const &rhs);
    void accumulateproduct_inplace(field<T> const &in1, field<T> const &in2, T coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    void accumulateproduct_inplace(field<T> const &in1, field<T> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    template<typename T2> void accumulateproduct_inplace(field<T> const &in1, field<T2> const &in2, T2 coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    template<typename T2> void accumulateproduct_inplace(field<T> const &in1, field<T2> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    void accumulateproduct_inplace(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2);
    void accumulateproduct_inplace(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2, T coef);
    void settoproduct(field<T> const &in1, field<T> const &in2, T coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    void settoproduct(field<T> const &in1, field<T> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    template<typename T2> void settoproduct(field<T> const &in1, field<T2> const &in2, T2 coef); /// this += coef * in1 * in2 without creating or copying extra temporaries
    template<typename T2> void settoproduct(field<T> const &in1, field<T2> const &in2);         /// this +=        in1 * in2 without creating or copying extra temporaries
    void settoproduct(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2);
    void settoproduct(CSfield<T> const &in1, size_t tauidx1, CSfield<T> const &in2, size_t tauidx2, T coef);
    template<typename T2, typename T3> void axpby_inplace(field<T2> const &y, T3 a, T3 b);
    template<typename T2, typename T3> void xpby_inplace(field<T2> const &y, T3 b);
    template<typename T2, typename T3> void axpy_inplace(field<T2> const &y, T3 a);

    // Reduction operations
    T integrate_r_intensive() const;
    T sum_elem() const;
    T l1norm() const;
    //T l2norm() const;
    T maxabs() const; // Return the element with the max abs value
    T maxreal() const; // Return the element with the max real part (signed)
    T minreal() const; // Return the element with the min real part (signed)


    // -------------------------- 
    //template<typename T2> void avg_arg(T2 &out) const; // Return avg arg of field entries. Only works for T == std::double
    //T avg_arg() const; // Average argument of field  
    // -------------------------- 

    // Comparison
    bool operator==(field<T> const &rhs) const;
    inline bool operator!=(field<T> const &rhs) const {return !(*this == rhs);};

  private:
    void CreateFFTPlans();
    void DestroyFFTPlans();
    T * getDataPtr() {return _data;}; /// Return raw data pointer; private, so only friends can use
    T const * getDataPtr() const {return _data;}; /// Return const raw data pointer; private, so only friends can use

    // Data members
    size_t      _nelem;
    T*          _data;
    std::string _label;

    SpaceGrid const &_grid;

    bool   _inrealspace;
    bool   _isFFTable;

#ifdef __GPU__
    cufftHandle _cufftplan; // Only one plan required for both directions
#else
    fftw_plan _fftw_plan_fwd, _fftw_plan_bck;
#endif

    template<typename T2> friend class field; /// Allow fields to access each others private members for implementing type-asymmetric operators
    template<typename T2> friend class CSfield;
};

// Binary operators: implemented as free functions so that we can exploit rvalue references for either operand
//
// field * field
template<typename T>
field<T> operator*(field<T> const &L, field<T> const &R)
{
  iobase::cdbg << "field binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", both operands L-value references" << std::endl;
  field<T> result(L);
  result *= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
field<T> operator*(field<T> &&L, field<T> const &R)
{
  iobase::cdbg << "field binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(L));
  result *= R;
  return result;
}

template<typename T>
field<T> operator*(field<T> const &L, field<T> &&R)
{
  iobase::cdbg << "field binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(R));
  result *= L;
  return result;
}

template<typename T>
field<T> operator*(field<T> &&L, field<T> &&R)
{
  iobase::cdbg << "field binary multiply: " << L.getLabel() << "*" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  field<T> result(std::move(L));
  result *= R;
  return result;
}

// field + field
template<typename T>
field<T> operator+(field<T> const &L, field<T> const &R)
{
  iobase::cdbg << "field binary add: " << L.getLabel() << "+" << R.getLabel() << ", both operands L-value references" << std::endl;
  field<T> result(L);
  result += R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
field<T> operator+(field<T> &&L, field<T> const &R)
{
  iobase::cdbg << "field binary add: " << L.getLabel() << "+" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(L));
  result += R;
  return result;
}

template<typename T>
field<T> operator+(field<T> const &L, field<T> &&R)
{
  iobase::cdbg << "field binary add: " << L.getLabel() << "+" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(R));
  result += L;
  return result;
}

template<typename T>
field<T> operator+(field<T> &&L, field<T> &&R)
{
  iobase::cdbg << "field binary add: " << L.getLabel() << "+" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  field<T> result(std::move(L));
  result += R;
  return result;
}

// field - field
template<typename T>
field<T> operator-(field<T> const &L, field<T> const &R)
{
  iobase::cdbg << "field binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", both operands L-value references" << std::endl;
  field<T> result(L);
  result -= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
field<T> operator-(field<T> &&L, field<T> const &R)
{
  iobase::cdbg << "field binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(L));
  result -= R;
  return result;
}

template<typename T>
field<T> operator-(field<T> const &L, field<T> &&R)
{
  // BE CAREFUL WITH NON-COMMUTATIVITY for this implementation
  iobase::cdbg << "field binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  // We can't MOVE CONSTRUCT from R and use the compound arithmetic to get the right commutativity. Use a copy construction instead.
  // TODO Alternative: re-implement element-wise division in place.
  field<T> result(L);
  result -= R;
  return std::move(result);
}

template<typename T>
field<T> operator-(field<T> &&L, field<T> &&R)
{
  iobase::cdbg << "field binary subtract: " << L.getLabel() << "-" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  field<T> result(std::move(L));
  result -= R;
  return result;
}


// field / field
template<typename T>
field<T> operator/(field<T> const &L, field<T> const &R)
{
  iobase::cdbg << "field binary divide: " << L.getLabel() << "/" << R.getLabel() << ", both operands L-value references" << std::endl;
  field<T> result(L);
  result /= R;
  return result; // Note that "return std::move(result)" would force move ctor, but disables compiler return-value optimization
}

template<typename T>
field<T> operator/(field<T> &&L, field<T> const &R)
{
  iobase::cdbg << "field binary divide: " << L.getLabel() << "/" << R.getLabel() << ", left operand " << L.getLabel() << " is R-value movable" << std::endl;
  field<T> result(std::move(L));
  result /= R;
  return result;
}

template<typename T>
field<T> operator/(field<T> const &L, field<T> &&R)
{
  // BE CAREFUL WITH NON-COMMUTATIVITY for this implementation
  iobase::cdbg << "field binary divide: " << L.getLabel() << "/" << R.getLabel() << ", right operand " << R.getLabel() << " is R-value movable" << std::endl;
  // We can't MOVE CONSTRUCT from R and use the compound arithmetic to get the right commutativity. Use a copy construction instead.
  // TODO Alternative: re-implement element-wise division in place.
  field<T> result(L);
  result /= R;
  return std::move(result);
}

template<typename T>
field<T> operator/(field<T> &&L, field<T> &&R)
{
  iobase::cdbg << "field binary divide: " << L.getLabel() << "/" << R.getLabel() << ", both operands are R-value movable" << std::endl;
  field<T> result(std::move(L));
  result /= R;
  return result;
}

#endif // _FIELD_H_
