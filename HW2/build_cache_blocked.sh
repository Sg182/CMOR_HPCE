#!/usr/bin/env bash
set -euo pipefail

SRC="src/matmul_blocked_omp.c"
OUT="matmul_blocked_omp"

if command -v gcc-15 >/dev/null 2>&1; then
    CC=gcc-15
elif command -v gcc-14 >/dev/null 2>&1; then
    CC=gcc-14
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc
else
    echo "Error: no suitable GCC compiler found."
    exit 1
fi

CFLAGS="-O3 -fopenmp -Wall -Wextra -std=c11"

echo "Using compiler: $(which $CC)"
$CC --version | head -n 1

echo "Compiling ${SRC} -> ${OUT}"
$CC $CFLAGS $SRC -lm -o $OUT

echo "Build complete: ./${OUT}"