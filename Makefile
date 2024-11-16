all: altis relaxed_oift altis_roift

#Compiladores
CC=gcc
CXX=g++

FLAGS= -Wall -O3 -msse
#-march=native 

LINKS= -lpthread -lz -lm -fopenmp

#Bibliotecas
GFTLIB  = -L./lib/gft/lib -lgft
GFTFLAGS  = -I./lib/gft/include

#Rules
libgft:
	$(MAKE) -C ./lib/gft

altis: altis.cpp libgft
	$(CXX) $(FLAGS) $(GFTFLAGS) \
	altis.cpp $(GFTLIB) -o altis $(LINKS)

relaxed_oift: relaxed_oift.cpp libgft
	$(CXX) $(FLAGS) $(GFTFLAGS) \
	relaxed_oift.cpp $(GFTLIB) -o relaxed_oift $(LINKS)

altis_roift: altis_roift.cpp libgft
	$(CXX) $(FLAGS) $(GFTFLAGS) \
	altis_roift.cpp $(GFTLIB) -o altis_roift $(LINKS)

clean:
	$(RM) *~ *.o altis relaxed_oift altis_roift ./out/* ./debug/*
