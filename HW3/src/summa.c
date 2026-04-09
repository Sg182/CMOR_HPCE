#include "summa.h"
#include "matrix_utils.h"

#include <stdlib.h>

void local_rankk_update(double *C_local,const double *A_panel,const double *B_panel,
                        int nb, int k) {
    for (int i = 0; i < nb; i++) {
        double *Crow = C_local + i * nb;
        const double *Arow = A_panel + i * k;

        for (int kk = 0; kk < k; kk++) {
            double a = Arow[kk];
            const double *Brow = B_panel + kk * nb;

            for (int j = 0; j < nb; j++) {
                Crow[j] += a * Brow[j];
            }
        }
    }
}

void summa_rankk(const double *A_local,const double *B_local,double *C_local,
                 int N, int q, int pr, int pc, int k,MPI_Comm row_comm, MPI_Comm col_comm) {
    int nb = N / q;
    int panels_per_block = nb / k;
    int num_steps = N / k;

    double *A_panel = (double *)malloc((size_t)nb * k * sizeof(double));
    double *B_panel = (double *)malloc((size_t)k * nb * sizeof(double));

    for (int step = 0; step < num_steps; step++) {
        int owner = step / panels_per_block;
        int local_panel_index = step % panels_per_block;
        int offset = local_panel_index * k;

        if (pc == owner) {
            pack_A_panel(A_local, A_panel, nb, offset, k);
        }

        if (pr == owner) {
            pack_B_panel(B_local, B_panel, nb, offset, k);
        }

        MPI_Bcast(A_panel, nb * k, MPI_DOUBLE, owner, row_comm);
        MPI_Bcast(B_panel, k * nb, MPI_DOUBLE, owner, col_comm);

        local_rankk_update(C_local, A_panel, B_panel, nb, k);
    }

    free(A_panel);
    free(B_panel);
}