CC=g++
CFLAGS=-c -Wall
LDFLAGS=-framework Accelerate -framework CoreFoundation
SOURCES=drive.cpp Lbfgs.cpp lhac.cpp LogReg.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=lhac-log-reg

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@