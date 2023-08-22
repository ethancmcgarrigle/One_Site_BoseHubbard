// A global include file - will be included everywhere in the code
#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#define TPI 6.28318530717958647693
#define PI 3.14159265358979323846

#define Xindex 0
#define Yindex 1
#define Zindex 2


#define IOFLOATDIGITS 10
//#define IOFLOATDIGITS 15

// Some extra debug output and checks are enabled if DEBUG is defined
// If it is NOT defined, we defined NDEBUG (no debug) which additionally
// optimizes away all assert() checks in the code - for higher performance
#ifndef DEBUG
#define NDEBUG
#endif

// Elements of the std library needed throughout
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <numeric> // for std::iota 
#include <algorithm>
#include <cmath>

#include <assert.h>

// Also fetch std integer types
#include <inttypes.h>
typedef uint64_t UInt; // unsigned integer used throughout for iterating and sizing objects

// Timers & other system-specific
#include <ctime>
#ifdef _WIN32
#include <windows.h>
#include <stdint.h> // portable: uint64_t;   MSVC: __int64
#else
// Here we assume the system is POSIX compliant - may need to break out more non-compliant cases in future
#include "time.h"      // For ctime()
#include <unistd.h>    // For gethostname
#include <sys/param.h> // For gethostname
#include <sys/types.h> // For gethostname
#include <sys/time.h>
#include <sys/times.h>
#endif

// Forward declarations of global data and functions.
bool CompareEqual(double A, double B, double tol=-1.0); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(std::complex<double> A, double B, double tol=-1.0); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(double A, std::complex<double> B, double tol=-1.0); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(std::complex<double> A, std::complex<double> B, double tol=-1.0); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(float A, float B, float tol=-1.0f); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(std::complex<float> A, float B, float tol=-1.0f); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(float A, std::complex<float> B, float tol=-1.0f); // Compare two floating point numbers for equality within machine precision
bool CompareEqual(std::complex<float> A, std::complex<float> B, float tol=-1.0f); // Compare two floating point numbers for equality within machine precision


// Useful general functions defined in global.cpp
#ifdef __cplusplus
extern "C" {
#endif
  void memerr(const char* file, int line);      // Report location of a memory allocation error
  void codeerror_abort(const char* message, const char* file, int line);      // abort due to some code inconsistency
  void usererror_exit(const char* message, const char* file, int line);      // exit due to user input
  double gettime(); // Get the current time in seconds
  size_t nextpow2(size_t in); // Round up to next integer power of 2
  size_t prevpow2(size_t in); // Round down to next integer power of 2
  size_t nextmult2(size_t in); // Round up to next multiple of 2
  void indexx(UInt n, const double arr[], UInt indx[]);
  bool checkStopStatus();
  void clearStopStatus();
#ifdef __cplusplus
}
#endif

// Useful output streams
namespace iobase {
  // Create a nullstream class that discards all content sent to it
  class nullstreambuf : public std::streambuf
  {
    protected:
      virtual int overflow(int c) {return 0;};
  };
  // Create a debug stream that writes messages with "!DEBUG " at
  // the start of each line and sends the remaining part of the message to the
  // ostream object that was passed in the ctor arg.
  class dbgstreambuf : public std::streambuf
  {
    public:
      explicit dbgstreambuf(std::ostream &out) : _out(out), _pSink(0), _newline(true)
      {
        _pSink = _out.rdbuf(); // Take a copy of the output's streambuf
      };
    protected:
      virtual int_type overflow(int_type c = traits_type::eof())
      {
        if(traits_type::eq_int_type(c, traits_type::eof()))
          return _pSink->pubsync() == -1 ? c : traits_type::not_eof(c); // Convention: sync buffer if EOF
        if(_newline)
        {
          std::ostream str(_pSink);
          if(!(str<<"!DEBUG "))
            return traits_type::eof();
        }
        _newline = traits_type::to_char_type(c) == '\n';
        return _pSink->sputc(c); // Send char to output
      };
    private:
      std::ostream &_out;
      std::streambuf *_pSink;
      bool _newline;
  };
  // Specify that the cdbg object exists in this namespace scope
  // to any translation unit that includes global.h
  extern std::ostream cdbg;
}



#endif
