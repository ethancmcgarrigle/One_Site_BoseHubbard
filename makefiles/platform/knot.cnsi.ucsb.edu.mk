# platform.mk architecture/machine dependent paths and make rules

FFTWPATH_DP=/home/kdelaney/lib/fftw-3.3.4_dualPll_DoublePrec
FFTWPATH_SP=/home/kdelaney/lib/fftw-3.3.4_dualPll_SinglePrec
MKLPATH=/opt/intel/composer_xe_2013_sp1.0.080/mkl

ifeq ($(PLL),serial)
  CXX               = icpc

  CFLAGS            = -g -m64 $(DEFINE)

  OPTFLAGS.release  = -xSSE3 -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug    = -O0 -Wall -DDEBUG
  OPTFLAGS.profile  = -xSSE3 -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH_DP}/include

  DEFINE           := -DENABLESP

  ASMFLAGS = -masm=intel

  LDFLAGS.release = -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3 -lfftw3f
  LDFLAGS.debug   = -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3 -lfftw3f
  LDFLAGS.profile = -g -pg -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3 -lfftw3f
else ifeq ($(PLL),pll)
  CXX = mpic++

  CFLAGS            = -g -openmp -m64 $(DEFINE)

  OPTFLAGS.release  = -xSSE3 -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug    = -O0 -Wall -DDEBUG
  OPTFLAGS.profile  = -xSSE3 -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH_DP}/include

  DEFINE           := -DENABLESP -D__MPI__ -D__OMP__ 

  ASMFLAGS = -masm=intel

  LDFLAGS.release = -openmp -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3_mpi -lfftw3f_mpi -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f
  LDFLAGS.debug   = -openmp -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3_mpi -lfftw3f_mpi -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f
  LDFLAGS.profile = -g -pg -openmp -lm -L${FFTWPATH_DP}/lib -L${FFTWPATH_SP}/lib -lfftw3_mpi -lfftw3f_mpi -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f
else ifeq ($(PLL),gpu)
  CXX = nvcc

  CFLAGS            = -lineinfo -arch sm_20 -use_fast_math $(DEFINE)

  OPTFLAGS.release  = -O3
  OPTFLAGS.debug    = -O0 -DDEBUG
  OPTFLAGS.profile  = -Xcompiler "-g -pg" -O3

  INCLUDE          := -I/usr/local/cuda/include

  DEFINE           := -DENABLESP -D__GPU__

  ASMFLAGS  =

  LDFLAGS.release = -lcufft -lcurand
  LDFLAGS.debug   = -lcufft -lcurand
  LDFLAGS.profile = -lcufft -lcurand
else ifeq ($(PLL),mic)
  CXX = icpc

  CFLAGS            = -g -mmic -openmp -m64 -mkl=parallel $(DEFINE)

  OPTFLAGS.release  = -qopt-report -qopt-report-file=$@.optrpt -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug    = -qopt-report -qopt-report-file=$@.optrpt -O0 -Wall -DDEBUG
  OPTFLAGS.profile  = -qopt-report -qopt-report-file=$@.optrpt -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${MKLPATH}/include/fftw

  DEFINE           := -DENABLESP -D__OMP__

  ASMFLAGS  =

  LDFLAGS.release = -mmic -openmp -L${MKLPATH}/lib/mic -lpthread -lm
  LDFLAGS.debug   = -mmic -openmp -L${MKLPATH}/lib/mic -lpthread -lm
  LDFLAGS.profile = -mmic -g -pg -openmp -L${MKLPATH}/lib -lpthread -lm
endif

# Select specific set of OPTFLAGS and LDFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))
LDFLAGS  := $(LDFLAGS.$(BUILD))
