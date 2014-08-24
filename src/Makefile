## pass it as argument to make, e.g., make USE_ACCELERATE=true
# USE_ACCELERATE=no

MATLAB_ROOT = /Applications/MATLAB_R2013a.app

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

CXX=g++

ifeq ($(USE_ACCELERATE), true)
	# CXXFLAGS = -Wall -fpic  -fno-omit-frame-pointer -D_FILE_OFFSET_BITS=64 -std=c++0x -DUSE_ACCELERATE
	CXXFLAGS = -Wall -fpic  -fno-omit-frame-pointer -std=c++0x -DUSE_ACCELERATE
	LDFLAGS = -framework Accelerate -framework CoreFoundation
else
	CXXFLAGS = -Wall -fpic  -fno-omit-frame-pointer -std=c++0x $(MATLAB_INCLUDE)
	LDFLAGS = -lmwlapack -lmwblas -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 -L$(MATLAB_ROOT)/bin/maci64 
endif


CXXOPTFLAGS = -O3 -DNDEBUG $(CXXFLAGS)
CXXMEXFLAGS = -O3 -DNDEBUG -std=c++0x $(MATLAB_INCLUDE)
LDMEXFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/maci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 $(LDFLAGS) -L$(MATLAB_ROOT)/bin/maci64 


all: LHACl1log.mexmaci64 LHACl1log.cmd

%.mexmaci64: Lbfgs.o LogReg.o lhac-mex-gen.o
	$(CXX) $(LDMEXFLAGS) $^ -o $@	

%.cmd: Lbfgs.o LogReg.o drive.o
	$(CXX) $(LDFLAGS) $^ -o $@	

lhac-mex-gen.o: lhac-mex-gen.cpp
	$(CXX) $(CXXMEXFLAGS) -m64 -c $< -o $@

Lbfgs.o: Lbfgs.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@

LogReg.o: LogReg.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@	

drive.o: drive.cpp
	$(CXX) $(CXXOPTFLAGS) -m64 -c $< -o $@

clean :
	rm -f *.o *.mexmaci64 *.mexglx *.cmd





