CC=g++
CFLAGS=-c -Wall
LDFLAGS=-framework Accelerate -framework CoreFoundation
SOURCES=drive.cpp Lbfgs.cpp lhac.cpp myUtilities.cpp sics_lhac.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=SICS

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@