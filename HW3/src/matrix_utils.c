#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix_utils.h"

double A_global_value(int i, int j) {
    return 1.0 + 0.01 * i + 0.001 * j;
}

double B_global_value(int i, int j) {
    return 0.5 + 0.002 * i - 0.001 * j;
}

void init_local_block_A(double *A, int N, int q, int pr, int pc) {
    int nb = N / q;
    int global_row0 = pr * nb;
    int global_col0 = pc * nb;

    for (int i = 0; i < nb; i++) {
        int gi = global_row0 + i;
        for (int j = 0; j < nb; j++) {
            int gj = global_col0 + j;
            A[i * nb + j] = A_global_value(gi, gj);
        }
    }
}

void init_local_block_B(double *B, int N, int q, int pr, int pc) {
    int nb = N / q;
    int global_row0 = pr * nb;
    int global_col0 = pc * nb;

    for (int i = 0; i < nb; i++) {
        int gi = global_row0 + i;
        for (int j = 0; j < nb; j++) {
            int gj = global_col0 + j;
            B[i * nb + j] = B_global_value(gi, gj);
        }
    }
}

void pack_A_panel(const double *A_local,double *A_panel,int nb, int offset, int k) {
    for (int i = 0; i < nb; i++) {
        for (int kk = 0; kk < k; kk++) {
            A_panel[i * k + kk] = A_local[i * nb + (offset + kk)];
        }
    }
}

void pack_B_panel(const double *B_local,double *B_panel,int nb, int offset, int k) {
    const double *src = B_local + offset * nb;
    memcpy(B_panel, src, (size_t)(k * nb) * sizeof(double));
}

double verify_result(const double *C_local,int N, int q, int world_rank, int world_size) {
    int nb = N / q;
    double *gathered = NULL;
    double max_err = 0.0;

    if (world_rank == 0) {
        gathered = (double *)malloc((size_t)world_size * nb * nb * sizeof(double));
    }

    MPI_Gather((void *)C_local, nb * nb, MPI_DOUBLE,
               gathered, nb * nb, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double *C_global = (double *)calloc((size_t)N * N, sizeof(double));
        double *C_ref    = (double *)calloc((size_t)N * N, sizeof(double));

        for (int r = 0; r < world_size; r++) {
            int pr = r / q;
            int pc = r % q;
            const double *block = gathered + (size_t)r * nb * nb;

            for (int i = 0; i < nb; i++) {
                int gi = pr * nb + i;
                for (int j = 0; j < nb; j++) {
                    int gj = pc * nb + j;
                    C_global[gi * N + gj] = block[i * nb + j];
                }
            }
        }

        for (int i = 0; i < N; i++) {
            for (int kk = 0; kk < N; kk++) {
                double a = A_global_value(i, kk);
                for (int j = 0; j < N; j++) {
                    C_ref[i * N + j] += a * B_global_value(kk, j);
                }
            }
        }

        for (int idx = 0; idx < N * N; idx++) {
            double err = fabs(C_global[idx] - C_ref[idx]);
            if (err > max_err) {
                max_err = err;
            }
        }

        free(gathered);
        free(C_global);
        free(C_ref);
    }

    MPI_Bcast(&max_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return max_err;}

    void fill_random_matrix(double *M, int N, unsigned int seed){
        srand(seed);
        for (int i =0; i <N*N;i++){
            M[i] = (double)rand()/(double)RAND_MAX;

        }
    }
    void serial_matmul(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void pack_blocks(const double *global, double *packed, int N, int q) {
    int nb = N / q;

    for (int r = 0; r < q * q; r++) {
        int pr = r / q;
        int pc = r % q;

        double *block = packed + r * nb * nb;

        for (int i = 0; i < nb; i++) {
            int gi = pr * nb + i;
            for (int j = 0; j < nb; j++) {
                int gj = pc * nb + j;
                block[i * nb + j] = global[gi * N + gj];
            }
        }
    }
}

void unpack_blocks(const double *packed, double *global, int N, int q) {
    int nb = N / q;

    for (int r = 0; r < q * q; r++) {
        int pr = r / q;
        int pc = r % q;

        const double *block = packed + r * nb * nb;

        for (int i = 0; i < nb; i++) {
            int gi = pr * nb + i;
            for (int j = 0; j < nb; j++) {
                int gj = pc * nb + j;
                global[gi * N + gj] = block[i * nb + j];
            }
        }
    }
}

    double local_max_error(const double *X, const double *Y, int n) {
        double err = 0.0;

        for (int i = 0; i < n; i++) {
            double diff = fabs(X[i] - Y[i]);
            if (diff > err) {
                err = diff;
        }
    }

    return err;
}
