## pass it as argument to make, e.g., make USE_ACCELERATE=true
## USE_ACCELERATE=no

## Contact your administrator if you do not know 
## where your matlab is installed.
MATLAB_ROOT = /Applications/MATLAB_R2013a.app
## on linux this should be $(MATLAB_ROOT)/bin/glnxa64 
MATLAB_LINK = $(MATLAB_ROOT)/bin/maci64

OPENBLAS_LINK = /usr/local/opt/openblas/lib

CXX=g++-4.9

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

CXXBLAS = $(MATLAB_INCLUDE) 
LDBLAS = -lmwlapack -lmwblas -Wl,-rpath,$(MATLAB_LINK) -L$(MATLAB_LINK) -fopenmp

ifeq ($(USE_ACCELERATE), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -framework Accelerate -framework CoreFoundation -fopenmp
endif	
ifeq ($(USE_OPENBLAS), true)
	CXXBLAS =  -DUSE_CBLAS
	LDBLAS = -llapack -lopenblas -L$(OPENBLAS_LINK) -fopenmp
endif

CXXOPTFLAGS = -Wall -fpic  -fno-omit-frame-pointer -std=c++0x -O3 -DNDEBUG $(CXXBLAS) -fopenmp
CXXMEXFLAGS = -O3 -fpic -DNDEBUG -std=c++0x $(MATLAB_INCLUDE) -fopenmp
LDMEXFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-rpath,$(MATLAB_LINK) $(LDBLAS) -L$(MATLAB_LINK) 


all: lhac.mexmaci64 lhac.cmd

%.mexmaci64: LogReg.o lhac-mex-gen.o Lasso.o
	$(CXX) $(LDMEXFLAGS) $^ -o $@	

%.mexa64: LogReg.o lhac-mex-gen.o Lasso.o
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
	rm -f *.o *.mexmaci64 *.mexglx *.cmd *.mexa64





