# platform.mk architecture/machine dependent paths and make rules

FFTWPATH:=/home/kdelaney/lib/fftw-3.3.8_dualPLL_AVX512_ICPC
CUDAPATH:=/usr/local/cuda-9.0
#CUDAPATH:=/usr/local/cuda-10.0

ifeq ($(PLL),serial)
  CXX               = icpc

  CFLAGS            = -g -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include

  DEFINE           :=

  ASMFLAGS         :=

  LDFLAGS          := -lm -L${FFTWPATH}/lib -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64
else ifeq ($(PLL),pll)
  CXX              = icpc

  CFLAGS           = -g -qopenmp -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include

  DEFINE           := -D__OMP__

  ASMFLAGS         :=

  LDFLAGS          := -qopenmp -lm -L${FFTWPATH}/lib -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64
else ifeq ($(PLL),gpu)
  CXX               = nvcc

  CFLAGS            = -std=c++11 -lineinfo -g -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"

  OPTFLAGS.release  = -O3
  OPTFLAGS.debug    = -O0 -DDEBUG
  OPTFLAGS.profile  = -Xcompiler "-g -pg" -O3

  INCLUDE          := -I${CUDAPATH}/include -I./include/yaml-cpp/include

  DEFINE           := -D__GPU__

  ASMFLAGS         :=

#  LDFLAGS          := -lcufft -lcurand -L./include/yaml-cpp -lyaml-cpp_linux64 --linker-options="-R ${CUDAPATH}/lib64/"
# The following is for static linking
  LDFLAGS          := -lcufft_static -lcurand_static -lculibos -L./include/yaml-cpp -lyaml-cpp_linux64 -Xcompiler "-static-libgcc -static-libstdc++"
endif

# Select specific set of OPTFLAGS and LDFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))
