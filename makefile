CFLAGS = -funroll-all-loops -O3
CC = g++
LIBS = -L$(HOME)/lib -lbabak_lib_linux -L/usr/local/dmp/lib -ldcdf -llevmar -llapack -lblas -lf2c
CLIBS = -L/usr/local/dmp/nifti/lib -lniftiio -lznz -lm -lz -lc
INC= -I$(HOME)/include -I/usr/local/dmp/include -I/usr/local/dmp/nifti/include

all: kaiba

kaiba: kaiba.cxx
	$(CC) $(CFLAGS) -o kaiba kaiba.cxx $(LIBS) $(CLIBS) $(INC) 
