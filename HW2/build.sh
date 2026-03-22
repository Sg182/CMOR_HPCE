#!/usr/bin/env bash

set -e

if command -v gcc-15 >/dev/null 2>&1; then
    CC=gcc-15
elif command -v gcc-14 >/dev/null 2>&1; then
    CC=gcc-14
else
    echo "Error: Homebrew GCC not found (gcc-15 or gcc-14)."
    exit 1
fi

CFLAGS="-O3 -fopenmp -Iinclude -Wall -Wextra -std=c11"
SRC="main.c src/axpy.c"
OUT="axpy.out"

echo "Using compiler: $CC"
echo "Building $OUT ..."
$CC $CFLAGS $SRC -o $OUT
echo "Done."
echo "Run with:"
echo " $ export OMP_NUM_THREADS=4"
echo " $ ./$OUT 100000000"