CC      = mpicc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -Iinclude
LDFLAGS = -lm

SUMMA_TARGET  = summa_rankk_c
CANNON_TARGET = cannon_mm

SUMMA_SRC = src/main_summa.c src/summa.c src/matrix_utils.c
SUMMA_OBJ = $(SUMMA_SRC:.c=.o)

CANNON_SRC = src/main_cannon.c src/cannon.c src/matrix_utils.c
CANNON_OBJ = $(CANNON_SRC:.c=.o)

all: $(SUMMA_TARGET) $(CANNON_TARGET)

$(SUMMA_TARGET): $(SUMMA_OBJ)
	$(CC) $(CFLAGS) -o $@ $(SUMMA_OBJ) $(LDFLAGS)

$(CANNON_TARGET): $(CANNON_OBJ)
	$(CC) $(CFLAGS) -o $@ $(CANNON_OBJ) $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(SUMMA_TARGET) $(CANNON_TARGET)

run_summa:
	mpirun -np 4 ./$(SUMMA_TARGET) 8 2 verify

run_cannon_2x2:
	mpirun -np 4 ./$(CANNON_TARGET) 12 12345

run_cannon_3x3:
	mpirun -np 9 ./$(CANNON_TARGET) 12 12345

run_cannon_4x4:
	mpirun -np 16 ./$(CANNON_TARGET) 12 12345