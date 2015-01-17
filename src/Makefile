MATLAB_ROOT = /Applications/MATLAB_R2013a.app

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

MATLAB_LINK = $(MATLAB_ROOT)/bin/maci64

OPENBLAS_LINK = /usr/local/opt/openblas/lib

CXX=g++

CXXBLAS = $(MATLAB_INCLUDE)
LDBLAS = -lmwlapack -lmwblas -Wl,-rpath,$(MATLAB_LINK) -L$(MATLAB_LINK)

ifeq ($(USE_ACCELERATE), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -framework Accelerate -framework CoreFoundation
endif	
ifeq ($(USE_OPENBLAS), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -llapack -lopenblas -L$(OPENBLAS_LINK)
endif


CXXOPTFLAGS = -Wall -fpic  -fno-omit-frame-pointer -std=c++0x -O3 -DNDEBUG $(CXXBLAS)
LDMEXFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-rpath,$(MATLAB_LINK) $(LDBLAS) -L$(MATLAB_LINK)

SOURCES=sics_lhac.cpp sics_lhac-mex.cpp drive.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=lhac.mexmaci64 lhac.cmd


all: $(SOURCES) $(EXECUTABLE)
	
# $(EXECUTABLE): $(OBJECTS) 
# 	$(CXX) $(LDMEXFLAGS) $^ -o $@	
%.mexmaci64: sics_lhac.o sics_lhac-mex.o
	$(CXX) $(LDMEXFLAGS) $^ -o $@	

%.cmd: sics_lhac.o drive.o
	$(CXX) $(LDBLAS) $^ -o $@

.cpp.o:
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@

%-mex.o: %-mex.cpp
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@	

clean :
	rm -f *.o *.mexmaci64 *.mexglx *.cmd






