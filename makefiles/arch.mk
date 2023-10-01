# platform.mk architecture/machine dependent paths and make rules

FFTWPATH= ~/libraries/fftw-3.3.10
BOOSTPATH= ~/libraries/boost_1_81_0

ifeq ($(PLL),serial)
  CXX              := g++

  # CFLAGS is recursively expanded so we can keep adding to DEFINE
  CFLAGS            = -g -rdynamic -m64 -std=c++11 $(DEFINE)

  OPTFLAGS.release := -O3 -w -DNDEBUG -march=native -fexpensive-optimizations -mtune=native -mavx2
  OPTFLAGS.debug   := -O0 -Wall -pedantic -DDEBUG
  OPTFLAGS.profile := -pg -O3 -w -DNDEBUG -march=native -fexpensive-optimizations -mtune=native -mavx2
  OPTFLAGS.coverage:= -O0 -DDEBUG -fprofile-arcs -ftest-coverage

  INCLUDE          := -I ${FFTWPATH}/api -I./include/yaml-cpp/include -I ${BOOSTPATH}

  ASMFLAGS         := #-masm=intel

  LDFLAGS          := -g -lm -L${FFTWPATH} -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64

else ifeq ($(PLL),pll)
  CXX              := g++

  # CFLAGS is recursively expanded so we can keep adding to DEFINE
  CFLAGS            = -g -m64 -fopenmp $(DEFINE)

  OPTFLAGS.release := -O3 -w -DNDEBUG -march=native -fexpensive-optimizations -mtune=native -mavx2
  OPTFLAGS.debug   := -O0 -Wall -pedantic -DDEBUG
  OPTFLAGS.profile := -pg -O3 -w -DNDEBUG -march=native -fexpensive-optimizations -mtune=native -mavx2
  OPTFLAGS.coverage:= -O0 -w -DDEBUG -fprofile-arcs -ftest-coverage

  INCLUDE          := -I${FFTWPATH}/api -I./include/yaml-cpp/include -I ${BOOSTPATH}

  DEFINE           += -D__OMP__

  ASMFLAGS         := -masm=intel

  # Note that this is combined with CFLAGS when linking, so -fopenmp will be present
  LDFLAGS          := -lm -L${FFTWPATH} -lfftw3_omp -lfftw3 -L./include/yaml-cpp -lyaml-cpp_linux64
else ifeq ($(PLL),gpu)
  CXX              := nvcc  

  # CFLAGS is recursively expanded so we can keep adding to DEFINE
  #CFLAGS            = -lineinfo -std=c++11 -g -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"
  CFLAGS            = -std=c++11 -lineinfo -g --Wno-deprecated-gpu-targets -arch sm_86 -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"
  #CFLAGS            = -std=c++11 -lineinfo -g --Wno-deprecated-gpu-targets -arch compute_37 -code sm_37 -code sm_60 -code sm_70 -code sm_80 -use_fast_math $(DEFINE) -Xcompiler "-std=c++11"


  OPTFLAGS.release := -O3
  OPTFLAGS.debug   := -O0 -DDEBUG
  OPTFLAGS.profile := -Xcompiler "-g -pg" -O3

  INCLUDE          := -I/usr/local/cuda/include -I./include/yaml-cpp/include -I ${BOOSTPATH}

  DEFINE           += -D__GPU__

  ASMFLAGS         :=

  # For static linking, link to static versions of all CUDA dependences AND statically link to libgcc and libstdc++
  #LDFLAGS         = -lcufft_static -lcurand_static -lculibos -Xcompiler="-static-libgcc -static-libstdc++"
  # If -static-libstdc++ doesn't work, g++ / gcc version might be too old. Then try this:
  # ln -s `g++ $(CXXFLAGS) -print-file-name=libstdc++.a` libstdc++.a to get a link to the system .a file in the current location, and add -L. to linker arguments.
  #
  # Dynamic linking with added RPATH for this machine.
  LDFLAGS         := -lcufft -lcurand -L./include/yaml-cpp -lyaml-cpp_linux64
else
  $(error Parallel option not enabled for this machine / platform : $(PLL))
endif

# Select specific set of OPTFLAGS according to BUILD option
OPTFLAGS := $(OPTFLAGS.$(BUILD))

