#ifndef CANNON_H
#define CANNON_H

#include <mpi.h>

void local_matmul_add(const double *A, const double *B, double *C, int nb);

void initial_skew(double *A_local, double *B_local,int nb, int q, int pr, int pc,
                  MPI_Comm comm);

void cannon_multiply(double *A_local, double *B_local, double *C_local,int nb, int q, int pr, int pc,
     MPI_Comm comm);

#endif