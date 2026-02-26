#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stddef.h>


// Initializing a function pointer
extern int Block_size;
extern int Thresh;
typedef void(*transpose_fn)(const int n, double *AT, double *A);

void transpose_naive(const int n, double *AT, double *A);
void transpose_blocked(const int n, double *AT, double *A);
void transpose_recursive(const int n, double *AT, double *A);


double time_transpose(transpose_fn f, const int n, double *AT, double *A, int trials);

void transpose_reference(const int n, double *AT, const double *A);

double max_error(const int n, const double *AT, const double *AT_ref); 