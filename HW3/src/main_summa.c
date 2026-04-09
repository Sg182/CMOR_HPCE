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
            fprintf(stderr, "Usage: mpirun -np <p> ./summa_rankk_c <N> <k> [verify]\n");
            fprintf(stderr, "  N      = global matrix size (NxN)\n");
            fprintf(stderr, "  k      = rank-k update width\n");
            fprintf(stderr, "  verify = optional word: verify\n");
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int k = atoi(argv[2]);
    int do_verify = (argc >= 4 && strcmp(argv[3], "verify") == 0);

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

    init_local_block_A(A_local, N, q, pr, pc);
    init_local_block_B(B_local, N, q, pr, pc);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    summa_rankk(A_local, B_local, C_local, N, q, pr, pc, k, row_comm, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    double max_time = 0.0;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("SUMMA rank-k completed\n");
        printf("p               = %d\n", world_size);
        printf("sqrt(p)         = %d\n", q);
        printf("N               = %d\n", N);
        printf("local block     = %d x %d\n", nb, nb);
        printf("k               = %d\n", k);
        printf("time (max rank) = %.6f s\n", max_time);
    }

    if (do_verify) {
        double max_err = verify_result(C_local, N, q, world_rank, world_size);
        if (world_rank == 0) {
            printf("verification max |error| = %.12e\n", max_err);
        }
    }

    free(A_local);
    free(B_local);
    free(C_local);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}