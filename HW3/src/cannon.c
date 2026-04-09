#include "cannon.h"

#include <mpi.h>

void local_matmul_add(const double *A, const double *B, double *C, int nb) {
    for (int i = 0; i < nb; i++) {
    for (int k = 0; k < nb; k++) {
            double a = A[i * nb + k];
            for (int j = 0; j < nb; j++) {
                C[i * nb + j] += a * B[k * nb + j];
            }
        }
    }
}

void initial_skew(double *A_local, double *B_local,int nb, int q, int pr, int pc,
                  MPI_Comm comm) {
    int left  = pr * q + (pc - 1 + q) % q;
    int right = pr * q + (pc + 1) % q;
    int up    = ((pr - 1 + q) % q) * q + pc;
    int down  = ((pr + 1) % q) * q + pc;

    for (int s = 0; s < pr; s++) {
        MPI_Sendrecv_replace(A_local, nb * nb, MPI_DOUBLE,
                             left, 0,
                             right, 0,
                             comm, MPI_STATUS_IGNORE);
    }

    for (int s = 0; s < pc; s++) {
        MPI_Sendrecv_replace(B_local, nb * nb, MPI_DOUBLE,up, 1,down, 1,
                             comm, MPI_STATUS_IGNORE);
    }
}

void cannon_multiply(double *A_local, double *B_local, double *C_local,
                     int nb, int q, int pr, int pc,MPI_Comm comm) {
    int left  = pr * q + (pc - 1 + q) % q;
    int right = pr * q + (pc + 1) % q;
    int up    = ((pr - 1 + q) % q) * q + pc;
    int down  = ((pr + 1) % q) * q + pc;

    initial_skew(A_local, B_local, nb, q, pr, pc, comm);

    for (int step = 0; step < q; step++) {
        local_matmul_add(A_local, B_local, C_local, nb);

        if (step < q - 1) {
            MPI_Sendrecv_replace(A_local, nb * nb, MPI_DOUBLE,left, 2,right, 2,
                                 comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv_replace(B_local, nb * nb, MPI_DOUBLE,up, 3,
                                 down, 3,
                                 comm, MPI_STATUS_IGNORE);
        }
    }
}