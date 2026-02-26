CC=gcc
CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra -Iinclude
LDFLAGS=-lm

OBJ=main.o src/mat_trans.o

all: transpose

transpose: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

main.o: main.c include/transpose.h
	$(CC) $(CFLAGS) -c main.c

src/mat_trans.o: src/mat_trans.c include/transpose.h
	$(CC) $(CFLAGS) -c src/mat_trans.c -o src/mat_trans.o

clean:
	rm -f *.o src/*.o transpose