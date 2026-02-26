#include <stdio.h>
#include "transpose.h"
#include <errno.h>
#include <time.h>
#include <string.h>
#include <math.h>

//This contains different functions for transposing a NxN matrix

int Block_size=8;     //PUT THE OPTIMAL BLOCK SIZE HERE
int Thresh=24;         //PUT THE OPTIMAL THRESHOLD

void transpose_reference(const int n, double *AT, const double *A) {
    for (int i = 0; i < n; ++i) {
        const int irow = i * n;
        for (int j = 0; j < n; ++j) {
            AT[j * n + i] = A[irow + j];
        }
    }
}

static inline double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fprintf(stderr, "clock_gettime failed: %s\n", strerror(errno));
        return 0.0;
    }
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}



double max_error(const int n, const double *AT, const double *AT_ref){
    double max_err = 0.0;
    const int N = n*n;
    for (int i=0; i<N; ++i){
        const double abs_err = fabs(AT_ref[i] - AT[i]);
        const double denom = fmax(1.0, fabs(AT_ref[i]));
        const double rel = abs_err / denom;
        if (rel > max_err) max_err = rel;

    }
    
    return max_err;

}

//=========================== NAIVE TRANSPOSE ===================================//
void transpose_naive(const int n, double* AT, double *A){
   for (int i =0; i < n; ++i){
      for (int j =0; j<n;++j){
         AT[j*n + i] = A[i*n +j];
      }
   }
}

//====================CACHED-BLOCKED TRANSPOSE===============================//

void transpose_blocked(const int n, double *AT, double *A) {
    const int B = Block_size;  
    for (int ii = 0; ii < n; ii += B) {
        for (int jj = 0; jj < n; jj += B) {
            const int i_max = (ii + B < n) ? (ii + B) : n;
            const int j_max = (jj + B < n) ? (jj + B) : n;
            for (int i = ii; i < i_max; ++i) {
                const int irow = i * n;
                for (int j = jj; j < j_max; ++j) {
                    AT[j * n + i] = A[irow + j];
                }
            }
        }
    }
}

//================================RECURSIVE TRANSPOSE====================================//

static void transpose_rec_impl(
    const int n,
    double *AT, const int at_ld, const int at_r0, const int at_c0,
    double *A,  const int a_ld,  const int a_r0,  const int a_c0,
    const int r, const int c
) {
    
    const int THRESH = Thresh; //BE MINDFUL HERE, I WAS BEING LAZY TO COME UP WITH A DIFFERENT Var NAME

    if (r <= THRESH && c <= THRESH) {
        for (int i = 0; i < r; ++i) {
            const int arow = (a_r0 + i) * a_ld;
            for (int j = 0; j < c; ++j) {
                /* AT(at_c0 + j, at_r0 + i) = A(a_r0 + i, a_c0 + j) */
                AT[(at_c0 + j) * at_ld + (at_r0 + i)] =
                    A[arow + (a_c0 + j)];
            }
        }
        return;
    }

    
    if (r >= c) {
        const int r1 = r / 2;
        const int r2 = r - r1;

        /* Top half rows */
        transpose_rec_impl(n, AT, at_ld, at_r0,     at_c0, A, a_ld, a_r0,     a_c0, r1, c);
        /* Bottom half rows */
        transpose_rec_impl(n, AT, at_ld, at_r0 + r1, at_c0, A, a_ld, a_r0 + r1, a_c0, r2, c);
    } else {
        const int c1 = c / 2;
        const int c2 = c - c1;

        /* Left half cols */
        transpose_rec_impl(n, AT, at_ld, at_r0, at_c0,     A, a_ld, a_r0, a_c0,     r, c1);
        /* Right half cols */
        transpose_rec_impl(n, AT, at_ld, at_r0, at_c0 + c1, A, a_ld, a_r0, a_c0 + c1, r, c2);
    }
}

void transpose_recursive(const int n, double *AT, double *A) {
     transpose_rec_impl(n, AT, n, 0, 0, A, n, 0, 0, n, n);
}

double time_transpose(transpose_fn f, const int n, double *AT, double *A, int trials){
    if (trials < 25) {
        trials = 25;   //SET THE NUMBER OF TRIALS YOU WISH
    }

    double best = 1e300;
    for (int i =0; i<trials ;i++){
        const double t0 = now_seconds();
        f(n, AT, A);
        const double t1 = now_seconds();
        //printf("DEBUG dt = %.17e\n", t1 - t0);

        const double dT = t1 -t0;
        if (dT < best) best = dT;
    }
    return best;
}