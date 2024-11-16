all: oiftrelax

#Compiladores
CC=gcc
CXX=g++

FLAGS= -Wall -O3 -lpthread -msse
#-march=native 

LINKS= -lz -lm -fopenmp

#Bibliotecas
GFTLIB  = -L./lib/gft/lib -lgft
GFTFLAGS  = -I./lib/gft/include

#Rules
libgft:
	$(MAKE) -C ./lib/gft

oiftrelax: oiftrelax.cpp libgft
	$(CXX) $(FLAGS) $(GFTFLAGS) \
	oiftrelax.cpp $(GFTLIB) -o oiftrelax $(LINKS)

clean:
	$(RM) *~ *.o oiftrelax
