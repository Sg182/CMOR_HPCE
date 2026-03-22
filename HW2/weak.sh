#!/usr/bin/env bash

set -e

Base=6250000

for t in 1 2 4 8 16
do
    N=$((Base*t))
    echo "===== Threads: $t | n = $N ====="
    export OMP_NUM_THREADS=$t
    ./axpy.out $N
    echo
done 