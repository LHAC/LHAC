MATLAB_ROOT = /Applications/MATLAB_R2013a.app

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include

CXX=g++

CXXFLAGS = -Wall -fpic  -fno-omit-frame-pointer -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -std=c++0x
CXXOPTFLAGS = -O3 -DNDEBUG $(CXXFLAGS)

LDFLAGS = -Wall -shared -lmx -lmex -lmat -lm -framework Accelerate -framework CoreFoundation
LDMACFLAGS = -m64 -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/maci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 -L$(MATLAB_ROOT)/bin/maci64 $(LDFLAGS)

SOURCES=sics_lhac.cpp sics_lhac-mex.cpp Lbfgs.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=LHAC.mexmaci64


all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(LDMACFLAGS) $^ -o $@	

.cpp.o:
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

clean :
	rm -f *.o *.mexa64 *.mexglx *.oct *.mex






