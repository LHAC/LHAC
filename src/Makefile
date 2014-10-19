## pass it as argument to make, e.g., make USE_ACCELERATE=true
# USE_ACCELERATE=no

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
CXXMEXFLAGS = -O3 -DNDEBUG -std=c++0x $(MATLAB_INCLUDE)
LDMEXFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/maci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 $(LDBLAS) -L$(MATLAB_ROOT)/bin/maci64 


all: LHACl1log.mexmaci64 LHACl1log.cmd

%.mexmaci64: LogReg.o lhac-mex-gen.o Lasso.o
	$(CXX) $(LDMEXFLAGS) $^ -o $@	

%.cmd: LogReg.o drive.o Lasso.o
	$(CXX) $(LDBLAS) $^ -o $@	

lhac-mex-gen.o: lhac-mex-gen.cpp
	$(CXX) $(CXXMEXFLAGS) -m64 -c $< -o $@

Lasso.o: Lasso.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@

LogReg.o: LogReg.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@	

drive.o: drive.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@

clean :
	rm -f *.o *.mexmaci64 *.mexglx *.cmd





