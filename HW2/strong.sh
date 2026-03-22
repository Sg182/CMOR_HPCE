#!/usr/bin/env bash

set -e

N=100000000

for t in 1 2 4 8 16
do
    echo "===== Threads: $t | n = $N ====="
    export OMP_NUM_THREADS=$t
    ./axpy.out $N
    echo
done