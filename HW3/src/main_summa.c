#include "matrix_utils.h"
#include "summa.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <p> ./summa_rankk_c <N> <k> [seed]\n");
            fprintf(stderr, "  N    = global matrix size (NxN)\n");
            fprintf(stderr, "  k    = rank-k update width\n");
            fprintf(stderr, "  seed = optional random seed (default 12345)\n");
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int k = atoi(argv[2]);
    unsigned int seed = 12345U;
    if (argc >= 4) {
        seed = (unsigned int)atoi(argv[3]);
    }

    int q = (int)lround(sqrt((double)world_size));
    if (q * q != world_size) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: p must be a perfect square.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (N % q != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: N must be divisible by sqrt(p).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nb = N / q;
    if (nb % k != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: local block size N/sqrt(p) must be divisible by k.\n");
            fprintf(stderr, "Equivalent requirement: N divisible by k*sqrt(p).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int pr = world_rank / q;
    int pc = world_rank % q;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &col_comm);

    double *A_local = (double *)malloc((size_t)nb * nb * sizeof(double));
    double *B_local = (double *)malloc((size_t)nb * nb * sizeof(double));
    double *C_local = (double *)calloc((size_t)nb * nb, sizeof(double));
    double *C_ref_local = (double *)malloc((size_t)nb * nb * sizeof(double));

    double *A_global = NULL;
    double *B_global = NULL;
    double *C_ref_global = NULL;
    double *C_global = NULL;

    double *A_packed = NULL;
    double *B_packed = NULL;
    double *C_packed = NULL;
    double *C_ref_packed = NULL;

    if (world_rank == 0) {
        A_global = (double *)malloc((size_t)N * N * sizeof(double));
        B_global = (double *)malloc((size_t)N * N * sizeof(double));
        C_ref_global = (double *)malloc((size_t)N * N * sizeof(double));
        C_global = (double *)malloc((size_t)N * N * sizeof(double));

        A_packed = (double *)malloc((size_t)world_size * nb * nb * sizeof(double));
        B_packed = (double *)malloc((size_t)world_size * nb * nb * sizeof(double));
        C_packed = (double *)malloc((size_t)world_size * nb * nb * sizeof(double));
        C_ref_packed = (double *)malloc((size_t)world_size * nb * nb * sizeof(double));

        fill_random_matrix(A_global, N, seed);
        fill_random_matrix(B_global, N, seed + 1U);

        pack_blocks(A_global, A_packed, N, q);
        pack_blocks(B_global, B_packed, N, q);

        serial_matmul(A_global, B_global, C_ref_global, N);
        pack_blocks(C_ref_global, C_ref_packed, N, q);
    }

    MPI_Scatter(A_packed, nb * nb, MPI_DOUBLE,A_local, nb * nb, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B_packed, nb * nb, MPI_DOUBLE,B_local, nb * nb, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    summa_rankk(A_local, B_local, C_local, N, q, pr, pc, k, row_comm, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    double max_time = 0.0;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gather(C_local, nb * nb, MPI_DOUBLE,C_packed, nb * nb, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Scatter(C_ref_packed, nb * nb, MPI_DOUBLE,C_ref_local, nb * nb, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double local_err = local_max_error(C_local, C_ref_local, nb * nb);
    double global_err = 0.0;

    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        unpack_blocks(C_packed, C_global, N, q);
    }

    for (int r = 0; r < world_size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == r) {
            printf("Rank %2d (row=%d, col=%d): local max error = %.12e\n",
                 world_rank, pr, pc, local_err);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\nSUMMA rank-k completed\n");
        printf("p                = %d\n", world_size);
        printf("sqrt(p)          = %d\n", q);
        printf("N                = %d\n", N);
        printf("local block      = %d x %d\n", nb, nb);
        printf("k                = %d\n", k);
        printf("time (max rank)  = %.6f s\n", max_time);
        printf("global max error = %.12e\n", global_err);
    }

    free(A_local);
    free(B_local);
    free(C_local);
    free(C_ref_local);

    if (world_rank == 0) {
        free(A_global);
        free(B_global);
        free(C_global);
        free(C_ref_global);
        free(A_packed);
        free(B_packed);
        free(C_packed);
        free(C_ref_packed);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}