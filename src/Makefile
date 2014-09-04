MATLAB_ROOT = /Applications/MATLAB_R2013a.app

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

OPENBLAS_LINK = /usr/local/opt/openblas/lib

CXX=g++

CXXBLAS = $(MATLAB_INCLUDE)
LDBLAS = -lmwlapack -lmwblas -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 -L$(MATLAB_ROOT)/bin/maci64 

ifeq ($(USE_ACCELERATE), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -framework Accelerate -framework CoreFoundation
endif	
ifeq ($(USE_OPENBLAS), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -llapack -lopenblas -L$(OPENBLAS_LINK)
endif


CXXOPTFLAGS = -Wall -fpic  -fno-omit-frame-pointer -std=c++0x -O3 -DNDEBUG $(CXXBLAS)
LDMEXFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/maci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 $(LDBLAS) -L$(MATLAB_ROOT)/bin/maci64

SOURCES=sics_lhac.cpp sics_lhac-mex.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=LHAC.mexmaci64


all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(LDMEXFLAGS) $^ -o $@	

.cpp.o:
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

clean :
	rm -f *.o *.mexmaci64 *.mexglx *.cmd






