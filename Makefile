# To make run
# make dep
# make

# C++ compiler
CPP = c++

SNOWGOOSE=../snowgoose

COMPI=../COMPI

2DLAT=../2DLATTICE

LOCSEARCH=../LOCSEARCH


# include folders
INCLUDE = -I$(SNOWGOOSE) -I$(COMPI) -I$(2DLAT) -I$(LOCSEARCH)

# C++ compiler options
CPPOPTS = --std=c++14 $(INCLUDE) -g -fopenmp

# Libraries to include 
LIBS = -fopenmp

# Linkers flags
LDFLAGS = -pthread 


all: testppproblem.exe testlocsearch.exe

-include deps.inc

clean: 
	rm -f *.exe *.o *.a *~ *.log deps.inc

dep:
	$(CPP) $(CPPOPTS) -MM -c *.cpp >> deps.inc;\
        true
tests:
	@for i in $(TESTS); do if ./$$i > /dev/null; then echo TEST PASSED; continue; else echo TEST FAILED; fi done


.o.exe:
	$(CPP) $(OPTS) -o $@ $< $(LIBS) $(EXTLIBS) $(LDFLAGS)

.cpp.o:
	$(CPP) $(CPPOPTS) -c $<

.c.o:
	$(CC) $(COPTS) -c $<

.SUFFIXES:
.SUFFIXES: .o .a .cpp .c .exe

