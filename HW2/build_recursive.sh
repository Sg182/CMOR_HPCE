#!/usr/bin/env bash
set -euo pipefail

SRC="src/matmul_recursive_omp.c"
OUT="matmul_recursive_tasks"

if command -v gcc-15 >/dev/null 2>&1; then
    CC=gcc-15
elif command -v gcc-14 >/dev/null 2>&1; then
    CC=gcc-14
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc
else
    echo "Error: Homebrew GCC not found (gcc-15 or gcc-14)."
    exit 1
fi

CFLAGS="-O3 -fopenmp -Iinclude -Wall -Wextra -std=c11"
echo "Compiling ${SRC} -> ${OUT}"
 $CC $CFLAGS $SRC -lm -o $OUT
echo "Build complete: ./${OUT}"
echo "Example run: ./${OUT} 1024 5 32 256"