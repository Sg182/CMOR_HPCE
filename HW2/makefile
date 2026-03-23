ifeq ($(shell command -v gcc-15 2>/dev/null),)
  ifeq ($(shell command -v gcc-14 2>/dev/null),)
    CC = gcc
  else
    CC = gcc-14
  endif
else
  CC = gcc-15
endif

CFLAGS = -O3 -fopenmp -std=c11 -Wall -Wextra -Iinclude
SRC = src/main.c src/integration_serial.c src/integration_reduction.c src/integration_atomic.c src/timing.c
OBJ = $(SRC:.c=.o)
TARGET = integration

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

%.o: %.c include/integration.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)