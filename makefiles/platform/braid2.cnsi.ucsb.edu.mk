# platform.mk architecture/machine dependent paths and make rules

FFTWPATH:=/home/kdelaney/lib/fftw-3.3.8_dualPLL_AVX512_ICPC
BOOSTPATH:=/home/emcgarrigle/boost_1_81_0
CUDAPATH:=/sw/cuda/cuda-11.7

ifeq ($(PLL),serial)
  CXX               = icpc

  CFLAGS            = -g -m64 -std=c++11 $(DEFINE)

# Note: using -xavx may be more efficient, but doing so
# includes multiple dispatch code with run-time checks for
# Intel CPUs.
# Any AMD nodes will crash, so I'm using -mavx instead.
  OPTFLAGS.release := -mavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -mavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include -I./include/yaml-cpp/include -I${BOOSTPATH}

  DEFINE           :=

  ASMFLAGS         :=

  LDFLAGS          := -lm -L${FFTWPATH}/lib -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64
else ifeq ($(PLL),pll)
  CXX              = icpc

  CFLAGS           = -g -qopenmp -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -w -DNDEBUG
  OPTFLAGS.debug   := -O0 -Wall -DDEBUG
  OPTFLAGS.profile := -xavx -O3 -no-prec-div -no-prec-sqrt -complex-limited-range -funroll-loops -pg -Wall -DNDEBUG

  INCLUDE          := -I${FFTWPATH}/include -I${BOOSTPATH} -I./include/yaml-cpp/include

  DEFINE           := -D__OMP__

  ASMFLAGS         :=

  LDFLAGS          := -qopenmp -lm -L${FFTWPATH}/lib -lfftw3 -lfftw3_omp -L./include/yaml-cpp -lyaml-cpp_linux64
else ifeq ($(PLL),gpu)
  CXX               = nvcc

  # CUDA -> PTX -> SASS
  #  -arch selects PTX generation. PTX -> SASS at runtime via JIT compilation. Requires a 'virtual' architecture argument (i.e., compute_XX)
  #  -code includes precompiled assembly (SASS) for the listed SMs directly in the binary. Executable grows in size with # targets.
  # * Tesla K80  : sm_37
  # * Tesla P100 : sm_60
  # * Tesla V100 : sm_70
  # * Tesla A100 : sm_80
  # NOTE: do not include SASS assembling for any target that is older than the oldest PTX virtual architecture.
  CFLAGS            = -std=c++11 -lineinfo -g --Wno-deprecated-gpu-targets -arch compute_37 -code sm_37 -code sm_60 -code sm_70 -code sm_80 -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"

  OPTFLAGS.release  = -O3
  OPTFLAGS.debug    = -O0 -DDEBUG
  OPTFLAGS.profile  = -Xcompiler "-g -pg" -O3

  INCLUDE          := -I${CUDAPATH}/include -I./include/yaml-cpp/include -I${BOOSTPATH}

  DEFINE           := -D__GPU__

  ASMFLAGS         :=

  LDFLAGS          := -lcufft -lcurand -L./include/yaml-cpp -lyaml-cpp_linux64 --linker-options="-R ${CUDAPATH}/lib64/"
# The following is for static linking
#  LDFLAGS          := -lcufft_static -lcurand_static -lculibos -L./include/yaml-cpp -lyaml-cpp_linux64 -Xcompiler "-static-libgcc -static-libstdc++"
endif

# Select specific set of OPTFLAGS and LDFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))
