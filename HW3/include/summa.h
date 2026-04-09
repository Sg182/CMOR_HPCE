#ifndef SUMMA_H
#define SUMMA_H

#include <mpi.h>

void local_rankk_update(double *C_local,const double *A_panel,const double *B_panel,
                        int nb, int k);

void summa_rankk(const double *A_local,const double *B_local,double *C_local,
                 int N, int q, int pr, int pc, int k,
                 MPI_Comm row_comm, MPI_Comm col_comm);

#endif