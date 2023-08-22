# platform.mk architecture/machine dependent paths and make rules

FFTWPATH:=/home/kdelaney/lib/fftw-3.3.8_dualPLL_AVX512_ICPC

ifeq ($(PLL),serial)
  CXX               = icpc

  # CFLAGS is recursively expanded so we can keep adding to DEFINE
  CFLAGS            = -g -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include

  DEFINE           :=

  ASMFLAGS         :=

  LDFLAGS          := -lm -L${FFTWPATH}/lib -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64_CentOS7
else ifeq ($(PLL),pll)
  CXX               = icpc

  CFLAGS            = -g -qopenmp -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include

  DEFINE           := -D__OMP__

  ASMFLAGS         :=

  LDFLAGS          := -qopenmp -lm -L${FFTWPATH}/lib -lfftw3 -lfftw3_omp -L./include/yaml-cpp -lyaml-cpp_linux64_CentOS7
else ifeq ($(PLL),gpu)
  # Note: use module load intel/18 first
  CXX               = nvcc

  CFLAGS            = -lineinfo -std=c++11 -g -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"

  OPTFLAGS.release  = -O3
  OPTFLAGS.debug    = -O0 -DDEBUG
  OPTFLAGS.profile  = -Xcompiler "-g -pg" -O3

  INCLUDE          := -I/usr/local/cuda/include -I./include/yaml-cpp/include

  DEFINE           := -D__GPU__

  LDFLAGS          := -lcufft -lcurand -L./include/yaml-cpp -lyaml-cpp_linux64_CentOS7
endif

# Select specific set of OPTFLAGS and LDFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))
