MATLAB_ROOT = /Applications/MATLAB_R2013a.app

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

CXX=g++

CXXFLAGS = -Wall -fpic  -fno-omit-frame-pointer -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -std=c++0x
CXXOPTFLAGS = -O3 -DNDEBUG $(CXXFLAGS)

LDFLAGS = -framework Accelerate -framework CoreFoundation
LDMACFLAGS = -Wall -shared -m64 -lmex -lmx  -lmat -lm -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/maci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 -L$(MATLAB_ROOT)/bin/maci64 $(LDFLAGS)


all: LHACl1log.mexmaci64 LHACl1log

%.mexmaci64: Lbfgs.o LogReg.o lhac-mex-gen.o
	$(CXX) $(LDMACFLAGS) $^ -o $@	

LHACl1log: Lbfgs.o LogReg.o drive.o
	$(CXX) $(LDFLAGS) $^ -o $@	

lhac-mex-gen.o: lhac-mex-gen.cpp
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

Lbfgs.o: Lbfgs.cpp
	$(CXX) $(CXXFLAGS) -m64 -c $< -o $@

LogReg.o: LogReg.cpp
	$(CXX) $(CXXFLAGS) -m64 -c $< -o $@	

drive.o: drive.cpp
	$(CXX) $(CXXFLAGS) -m64 -c $< -o $@

clean :
	rm -f *.o *.mexa64 *.mexglx *.oct *.mex





