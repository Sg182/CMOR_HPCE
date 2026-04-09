#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

double A_global_value(int i, int j);
double B_global_value(int i, int j);

void init_local_block_A(double *A, int N, int q, int pr, int pc);
void init_local_block_B(double *B, int N, int q, int pr, int pc);

void pack_A_panel(const double *A_local,double *A_panel,int nb, int offset, int k);

void pack_B_panel(const double *B_local,double *B_panel,int nb, int offset, int k);

double verify_result(const double *C_local,int N, int q, int world_rank, int world_size);

 
void fill_random_matrix(double *M, int N, unsigned int seed);

void serial_matmul(const double *A, const double *B, double *C, int N);

void pack_blocks(const double *global, double *packed, int N, int q);

void unpack_blocks(const double *packed, double *global, int N, int q);

double local_max_error(const double *X, const double *Y, int n);

#endif