# platform.mk architecture/machine dependent paths and make rules

FFTWPATH=${HOME}/lib/$(shell uname -s)/fftw-3.3.5_AVX_DP

ifeq ($(PLL),serial)
  CXX              = g++

  CFLAGS           = -g -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release = -O3 -w -DNDEBUG -march=core2 -fexpensive-optimizations -mtune=core2
  OPTFLAGS.debug   = -O0 -Wall -pedantic -DDEBUG
  OPTFLAGS.profile = -pg -O3 -w -DNDEBUG -march=core2 -fexpensive-optimizations -mtune=core2
  OPTFLAGS.coverage= -O0 -DDEBUG --coverage

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include

  DEFINE           := -D__DISABLESIMD__

  ASMFLAGS = -masm=intel

  LDFLAGS         = -g -rdynamic -lm -L${FFTWPATH}/lib -lfftw3 -L./include/yaml-cpp -lyaml-cpp_macos
else ifeq ($(PLL),pll)
  CXX               = g++

  # Note: no MPI on Spock
  CFLAGS            = -g -m64 -fopenmp $(DEFINE)

  OPTFLAGS.release  = -O3 -w -DNDEBUG -march=core2 -fexpensive-optimizations -mtune=core2
  OPTFLAGS.debug    = -O0 -Wall -pedantic -DDEBUG
  OPTFLAGS.profile  = -pg -O3 -w -DNDEBUG -march=core2 -fexpensive-optimizations -mtune=core2

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include

  DEFINE = -D__OMP__

  ASMFLAGS  = -masm=intel

  LDFLAGS         = -g -fopenmp -lm -L${FFTWPATH}/lib -lfftw3_omp -lfftw3 -L./include/yaml-cpp -lyaml-cpp_macos
else ifeq ($(PLL),gpu)
  CXX = nvcc

  CFLAGS            = -std=c++11 -use_fast_math $(DEFINE)

  OPTFLAGS.release  = -O3
  OPTFLAGS.debug    = -g -O0 -DDEBUG
  OPTFLAGS.profile  = -lineinfo -g -Xcompiler "-g -pg" -O3

  INCLUDE          := -I/usr/local/cuda/include -I./include/yaml-cpp/include

  DEFINE   = -D__GPU__ -D__DISABLESIMD__

  ASMFLAGS  =

  LDFLAGS         = -lcufft -lcurand -L./include/yaml-cpp -lyaml-cpp_macos
else
  $(error Parallel option not enabled for this machine / platform : $(PLL))
endif

# Select specific set of OPTFLAGS and LDFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))
