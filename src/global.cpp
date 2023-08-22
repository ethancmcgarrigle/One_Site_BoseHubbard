#include "global.h"

// IN THE GLOBAL SCOPE:
// Useful global functions
bool CompareEqual(double A, double B, double RelativeTol/*=-1.0*/)
{
  // Take the difference
  double diff = fabs(A-B);
  // Set up tolerance
  double T;
  if(RelativeTol < 0.)
    T = 100. * std::numeric_limits<double>::epsilon(); // Allow up to 100 epsilon difference by default
  else
    T = RelativeTol;

  // Compare for absolute difference close to zero
  if(diff<=T)
    return true;

  // Test based on RELATIVE error
  // Find the largest abs
  A = fabs(A);
  B = fabs(B);
  double largest = (B > A) ? B : A;
  if(diff <= largest * T)
    return true;

  return false;
}
bool CompareEqual(std::complex<double> A, double B, double tol/*=-1.0*/)
{
  return(CompareEqual(A.real(), B, tol) && CompareEqual(A.imag(),0.0, tol));
}
bool CompareEqual(std::complex<double> A, std::complex<double> B, double tol/*=-1.0*/)
{
  return(CompareEqual(A.real(), B.real(), tol) && CompareEqual(A.imag(), B.imag(), tol));
}
bool CompareEqual(double A, std::complex<double> B, double tol/*=-1.0*/)
{
  return(CompareEqual(A, B.real(), tol) && CompareEqual(0.0, B.imag(), tol));
}

bool CompareEqual(float A, float B, float RelativeTol/*=-1.0f*/)
{
  // Take the difference
  float diff = fabsf(A-B);
  // Set up tolerance
  float T;
  if(RelativeTol < 0.)
    T = 100.f * std::numeric_limits<float>::epsilon(); // Allow up to 100 epsilon difference by default
  else
    T = RelativeTol;

  // Compare for absolute difference close to zero
  if(diff<=T)
    return true;

  // Test based on RELATIVE error
  // Find the largest abs
  A = fabsf(A);
  B = fabsf(B);
  float largest = (B > A) ? B : A;
  if(diff <= largest * T)
    return true;

  return false;
}
bool CompareEqual(std::complex<float> A, float B, float tol/*=-1.0f*/)
{
  return(CompareEqual(A.real(), B, tol) && CompareEqual(A.imag(),0.0f, tol));
}
bool CompareEqual(std::complex<float> A, std::complex<float> B, float tol/*=-1.0f*/)
{
  return(CompareEqual(A.real(), B.real(), tol) && CompareEqual(A.imag(), B.imag(), tol));
}
bool CompareEqual(float A, std::complex<float> B, float tol/*=-1.0f*/)
{
  return(CompareEqual(A, B.real(), tol) && CompareEqual(0.0f, B.imag(), tol));
}


double gettime()
{
#ifdef _WIN32
  // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
  // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
  // until 00:00:00 January 1, 1970
  static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);
  SYSTEMTIME system_time;
  FILETIME file_time;
  uint64_t time;
  GetSystemTime( &system_time );
  SystemTimeToFileTime( &system_time, &file_time );
  time = ((uint64_t)file_time.dwLowDateTime );
  time = ((uint64_t)file_time.dwHighDateTime ) << 32;
  double currtime = (double) ((time - EPOCH) / 10000000L);
  currtime += (double) (system_time.wMilliseconds * 1000);
#else
  struct timeval tv;
  gettimeofday(&tv,0);
  double currtime = static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec)*1e-6;
#endif
  return currtime;
}


// Handle memory error: print location and exit code
void memerr(const char* file, int line)
{
  std::cerr << std::endl << std::endl << "  ***  Error allocating memory at line " << line << " in " << file << std::endl;
  exit(1);
}

#if defined __GCC__ || defined __GLIBC__
// Technically we should test for the ability to link to glibc. On some platforms
// including features.h will define __GNU_LIBRARY__ and/or __GLIBC__. However, features.h
// does not always exist even when gcc is installed.
// Proceed by assuming that gcc will always link to glibc
#include <execinfo.h>
#endif

#define SWAP(a,b) itemp=(a);(a)=(b);(b)=itemp;
#define M 7
#define NSTACK 50
// Numerical Recipes in C, Sec. 8.4 - Quicksort Routine
void indexx(UInt n, const double arr[], UInt indx[])
// Indexes an array arr[1..n], i.e., outputs the array indx[1..n] such that arr[indx[j]] is
// in ascending order for j =1, 2,. ..,N. The input quantities n and arr are not changed.
{
  UInt i, indxt, ir=n, itemp, j, k, l=1;
  int jstack=0;
  double a;
  // Set up stack
  std::vector<int> istack(NSTACK);
  // Initialize indx to input order
  for(j=1;j<=n;j++)
    indx[j]=j-1;

  for(;;)
  {
    if(ir-l < M)
    {
      for(j=l+1;j<=ir;j++)
      {
        indxt=indx[j];
        a=arr[indxt];
        for(i=j-1;i>=l;i--)
        {
          if(arr[indx[i]] <= a)
            break;
          indx[i+1]=indx[i];
        }
        indx[i+1]=indxt;
      }
      if(jstack == 0)
        break;
      ir=istack[jstack--];
      l=istack[jstack--];
    }
    else
    {
      k=(l+ir) >> 1;
      SWAP(indx[k],indx[l+1]);
      if(arr[indx[l]] > arr[indx[ir]])
      {
        SWAP(indx[l],indx[ir])
      }
      if(arr[indx[l+1]] > arr[indx[ir]])
      {
        SWAP(indx[l+1],indx[ir])
      }
      if(arr[indx[l]] > arr[indx[l+1]])
      {
        SWAP(indx[l],indx[l+1])
      }
      i=l+1;
      j=ir;
      indxt=indx[l+1];
      a=arr[indxt];
      for (;;)
      {
        do i++; while(arr[indx[i]] < a);
        do j--; while(arr[indx[j]] > a);
        if (j < i)
          break;
        SWAP(indx[i],indx[j])
      }
      indx[l+1]=indx[j];
      indx[j]=indxt;
      jstack += 2;
      if(jstack > NSTACK)
        codeerror_abort("NSTACK too small in indexx.",__FILE__,__LINE__);
      if(ir-i+1 >= j-l)
      {
        istack[jstack]=ir;
        istack[jstack-1]=i;
        ir=j-1;
      }
      else
      {
        istack[jstack]=j-1;
        istack[jstack-1]=l;
        l=i;
      }
    }
  }
}



// Abort due to code error or inconsistency
void codeerror_abort(const char *message, const char* file, int line)
{
  std::cerr << std::endl;
  std::cerr << " ERROR: code error or inconsistency" << std::endl;
  std::cerr << "  - Message = " << message << std::endl;
  std::cerr << "  - Received from line " << line << " in source file " << file << std::endl;
#if defined __GCC__ || defined __GLIBC__
  // If we're using glibc we can recover a full stack trace readily
  std::cerr << " STACK TRACE:" << std::endl;
  void* callstack[128];
  int frames = backtrace(callstack, 128);
  backtrace_symbols_fd(callstack, frames, STDERR_FILENO);
//  char** strs = backtrace_symbols(callstack, frames);
//  for (int i = 0; i < frames; ++i)
//    std::cout << "\t" << strs[i] << std::endl;
//  free(strs);
#endif
  abort();
}

// Exit due to unreasonable user behavior
void usererror_exit(const char *message, const char* file, int line)
{
  std::cerr << std::endl;
  std::cerr << " USAGE ERROR:" << std::endl;
  std::cerr << "  - Message = " << message << std::endl;
  std::cerr << "  - Received from line " << line << " in source file " << file << std::endl;
  exit(2); // Throw a non-zero return code
}

// Utility function for fast determination of power of 2 numbers
// used, for example, for fixing CUDA block sizes
size_t nextpow2(size_t v)
{
  // This algorithm works by propagating right-most set bit to all lower
  // bits, then adding one which carries the all bits to zero except
  // rightmost+1

  // ASSUMPTION: 'size_t' is 64 bits wide
  assert(sizeof(size_t)==8);

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v++;
  return(v);
}

// Utility function for fast determination of power of 2 numbers
size_t prevpow2(size_t v)
{
  // If already a power of 2, return original
  if((v != 0) && !(v & (v - 1)))
    return(v);
  else
    // Use nextpow2 to round to the next higher power of 2, then logic shift by 1
    return(nextpow2(v)>>1);
}

// Utility function for fast determination of multiples of 2
size_t nextmult2(size_t v)
{
  // Using integer rounding, we want v = (v+1)/2 * 2
  // However, i/2*2 can be replaced with (i & ~0x1)
  // Similarly, i/4*4 could be replaced with (i & ~0x3)
  size_t mask(0x1);
  mask = ~mask;
  return((v + 1) & mask);
}

bool checkStopStatus()
{
  FILE *status=fopen("STOP","r");
  if(status==NULL)
    return false;
  else
  {
    fclose(status);
    return true;
  }
}

void clearStopStatus()
{
  remove("STOP");
}

namespace iobase{
  // Create an instance of the nullstream class that will reside in this namespace for the process duration
#ifdef DEBUG
  dbgstreambuf dbgstream(std::cout);
  std::ostream cdbg(&dbgstream); // If debugging is on, send cdbg to cout with a stamp at the start of each line.
#else
  nullstreambuf nullstream;
  std::ostream cdbg(&nullstream); // If debugging is not on, send cdbg to oblivion
#endif
}
