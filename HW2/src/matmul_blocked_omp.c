#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

static int is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

static int imin(int a, int b) {
    return (a < b) ? a : b;
}

static void fill_matrix(int n, double *A, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; ++i) {
        A[i] = 2.0 * ((double) rand() / (double) RAND_MAX) - 1.0;
    }
}

static void zero_matrix(int n, double *A) {
    for (int i = 0; i < n * n; ++i) {
        A[i] = 0.0;
    }
}

static void matmul_reference(int n, const double *A, const double *B, double *Cref) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            Cref[i * n + j] = sum;
        }
    }
}

static double max_rel_err(int n, const double *C, const double *Cref) {
    double m = 0.0;
    for (int i = 0; i < n * n; ++i) {
        double denom = fabs(Cref[i]);
        if (denom < 1.0) denom = 1.0;
        double rel = fabs(C[i] - Cref[i]) / denom;
        if (rel > m) m = rel;
    }
    return m;
}

static void matmul_blocked_serial(int n, int BS, double *C, const double *A, const double *B) {
    for (int ii = 0; ii < n; ii += BS) {
        int iimax = imin(ii + BS, n);
        for (int jj = 0; jj < n; jj += BS) {
            int jjmax = imin(jj + BS, n);
            for (int kk = 0; kk < n; kk += BS) {
                int kkmax = imin(kk + BS, n);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i * n + k];
                        const double *bptr = &B[k * n + jj];
                        double *cptr = &C[i * n + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            cptr[j - jj] += aik * bptr[j - jj];
                        }
                    }
                }
            }
        }
    }
}

static void matmul_blocked_omp_for(int n, int BS, double *C, const double *A, const double *B) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            int iimax = imin(ii + BS, n);
            int jjmax = imin(jj + BS, n);
            for (int kk = 0; kk < n; kk += BS) {
                int kkmax = imin(kk + BS, n);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i * n + k];
                        const double *bptr = &B[k * n + jj];
                        double *cptr = &C[i * n + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            cptr[j - jj] += aik * bptr[j - jj];
                        }
                    }
                }
            }
        }
    }
}

static void matmul_blocked_omp_collapse2(int n, int BS, double *C, const double *A, const double *B) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            int iimax = imin(ii + BS, n);
            int jjmax = imin(jj + BS, n);
            for (int kk = 0; kk < n; kk += BS) {
                int kkmax = imin(kk + BS, n);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i * n + k];
                        const double *bptr = &B[k * n + jj];
                        double *cptr = &C[i * n + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            cptr[j - jj] += aik * bptr[j - jj];
                        }
                    }
                }
            }
        }
    }
}

static void matmul_blocked_omp_collapse3(int n, int BS, double *C, const double *A, const double *B) {
    int nb = (n + BS - 1) / BS;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int bi = 0; bi < nb; ++bi) {
        for (int bj = 0; bj < nb; ++bj) {
            for (int li = 0; li < BS; ++li) {
                int ii = bi * BS;
                int jj = bj * BS;
                int i = ii + li;
                int jjmax;
                if (i >= n) continue;
                jjmax = imin(jj + BS, n);
                for (int kk = 0; kk < n; kk += BS) {
                    int kkmax = imin(kk + BS, n);
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[i * n + k];
                        const double *bptr = &B[k * n + jj];
                        double *cptr = &C[i * n + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            cptr[j - jj] += aik * bptr[j - jj];
                        }
                    }
                }
            }
        }
    }
}

typedef void (*matmul_fn)(int, int, double *, const double *, const double *);

static double best_time_seconds(matmul_fn fn, int n, int BS, double *C,
                                const double *A, const double *B,
                                int trials, int threads) {
    double best = DBL_MAX;
    omp_set_num_threads(threads);

    for (int t = 0; t < trials; ++t) {
        double t0, t1, dt;
        zero_matrix(n, C);
        t0 = omp_get_wtime();
        fn(n, BS, C, A, B);
        t1 = omp_get_wtime();
        dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static const char *collapse_name(int c) {
    if (c == 1) return "omp_for";
    if (c == 2) return "collapse2";
    if (c == 3) return "collapse3";
    return "unknown";
}

static matmul_fn fn_from_collapse(int c) {
    if (c == 1) return matmul_blocked_omp_for;
    if (c == 2) return matmul_blocked_omp_collapse2;
    if (c == 3) return matmul_blocked_omp_collapse3;
    return NULL;
}

static int choose_best_collapse(int n, int BS, int threads, double *C,
                                const double *A, const double *B, int trials) {
    int candidates[3] = {1, 2, 3};
    double best_t = DBL_MAX;
    int best_c = -1;

    for (int idx = 0; idx < 3; ++idx) {
        int c = candidates[idx];
        double t = best_time_seconds(fn_from_collapse(c), n, BS, C, A, B, trials, threads);
        printf("collapse candidate %s, threads=%d, time=%.6f s\n", collapse_name(c), threads, t);
        if (t < best_t) {
            best_t = t;
            best_c = c;
        }
    }
    return best_c;
}

static void print_scaling_header(const char *label) {
    printf("\n%s\n", label);
    printf("threads,time_s,speedup,efficiency,gflops\n");
}

static void strong_scaling_study(int n, int BS, int best_collapse, double *C,
                                 const double *A, const double *B, int trials) {
    int threads_list[5] = {1, 2, 4, 8, 16};
    matmul_fn fn = fn_from_collapse(best_collapse);
    double t1 = best_time_seconds(fn, n, BS, C, A, B, trials, 1);

    print_scaling_header("STRONG SCALING (fixed problem size)");
    for (int idx = 0; idx < 5; ++idx) {
        int th = threads_list[idx];
        double t = best_time_seconds(fn, n, BS, C, A, B, trials, th);
        double speedup = t1 / t;
        double eff = speedup / (double) th;
        double gflops = (2.0 * n * n * n) / t / 1e9;
        printf("%d,%.6f,%.6f,%.6f,%.6f\n", th, t, speedup, eff, gflops);
    }
}

static void weak_scaling_study(int base_n, int BS, int best_collapse, int trials) {
    int threads_list[5] = {1, 2, 4, 8, 16};
    matmul_fn fn = fn_from_collapse(best_collapse);
    double t_ref = -1.0;

    printf("\nWEAK SCALING (n ~ base_n * threads^(1/3))\n");
    printf("threads,n,time_s,relative_to_1thread\n");

    for (int idx = 0; idx < 5; ++idx) {
        int th = threads_list[idx];
        int n = (int) llround(base_n * cbrt((double) th));
        double *A, *B, *C;
        double t;

        if (n % BS != 0) n += BS - (n % BS);

        A = (double *) malloc((size_t) n * n * sizeof(double));
        B = (double *) malloc((size_t) n * n * sizeof(double));
        C = (double *) malloc((size_t) n * n * sizeof(double));
        if (A == NULL || B == NULL || C == NULL) {
            fprintf(stderr, "Memory allocation failed in weak scaling study.\n");
            free(A);
            free(B);
            free(C);
            return;
        }

        fill_matrix(n, A, 1234u + (unsigned int) th);
        fill_matrix(n, B, 5678u + (unsigned int) th);
        t = best_time_seconds(fn, n, BS, C, A, B, trials, th);
        if (th == 1) t_ref = t;
        printf("%d,%d,%.6f,%.6f\n", th, n, t, t / t_ref);

        free(A);
        free(B);
        free(C);
    }
}

int main(int argc, char **argv) {
    int n = (argc >= 2) ? atoi(argv[1]) : 512;
    int BS = (argc >= 3) ? atoi(argv[2]) : 32;
    int trials = (argc >= 4) ? atoi(argv[3]) : 3;
    double *A, *B, *C, *Cref;
    int best_c;
    double err_serial, err_best;

    if (!is_power_of_two(n)) {
        fprintf(stderr, "Error: n must be a power of 2 for the requested test setup.\n");
        return 1;
    }

    printf("n=%d, block_size=%d, trials=%d, max_threads_available=%d\n",
           n, BS, trials, omp_get_max_threads());

    A = (double *) malloc((size_t) n * n * sizeof(double));
    B = (double *) malloc((size_t) n * n * sizeof(double));
    C = (double *) malloc((size_t) n * n * sizeof(double));
    Cref = (double *) malloc((size_t) n * n * sizeof(double));

    if (A == NULL || B == NULL || C == NULL || Cref == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 1;
    }

    fill_matrix(n, A, 1234u);
    fill_matrix(n, B, 5678u);
    zero_matrix(n, C);
    zero_matrix(n, Cref);

    matmul_reference(n, A, B, Cref);
    matmul_blocked_serial(n, BS, C, A, B);
    err_serial = max_rel_err(n, C, Cref);
    printf("serial blocked max relative error = %.6e\n", err_serial);
    if (err_serial > 1e-12) {
        fprintf(stderr, "Serial blocked implementation failed correctness check.\n");
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 2;
    }

    best_c = choose_best_collapse(n, BS, 16, C, A, B, trials);
    printf("\nSelected best parallelization variant at 16 threads: %s\n", collapse_name(best_c));

    zero_matrix(n, C);
    fn_from_collapse(best_c)(n, BS, C, A, B);
    err_best = max_rel_err(n, C, Cref);
    printf("best variant max relative error = %.6e\n", err_best);
    if (err_best > 1e-12) {
        fprintf(stderr, "Chosen OpenMP implementation failed correctness check.\n");
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 3;
    }

    strong_scaling_study(n, BS, best_c, C, A, B, trials);
    weak_scaling_study(384, BS, best_c, trials);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
