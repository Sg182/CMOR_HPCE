#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_BASE_CASE 32
#define DEFAULT_TASK_CUTOFF 256

static int imin(int a, int b) { return a < b ? a : b; }

static int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static void fill_random_matrix(int n, double *A, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; ++i) {
        A[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
    }
}

static void zero_matrix(int n, double *A) {
    memset(A, 0, (size_t)n * (size_t)n * sizeof(double));
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

static void matmul_recursive_serial_accum(const int N, const int n,
                                          double *C, const double *A, const double *B,
                                          int base_case) {
    if (n <= base_case) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double cij = C[j + i * N];
                for (int k = 0; k < n; ++k) {
                    cij += A[k + i * N] * B[j + k * N];
                }
                C[j + i * N] = cij;
            }
        }
        return;
    }

    {
        const int h = n / 2;
        const double *A11 = A;
        const double *A12 = A + h;
        const double *A21 = A + h * N;
        const double *A22 = A + h * N + h;

        const double *B11 = B;
        const double *B12 = B + h;
        const double *B21 = B + h * N;
        const double *B22 = B + h * N + h;

        double *C11 = C;
        double *C12 = C + h;
        double *C21 = C + h * N;
        double *C22 = C + h * N + h;

        matmul_recursive_serial_accum(N, h, C11, A11, B11, base_case);
        matmul_recursive_serial_accum(N, h, C11, A12, B21, base_case);

        matmul_recursive_serial_accum(N, h, C12, A11, B12, base_case);
        matmul_recursive_serial_accum(N, h, C12, A12, B22, base_case);

        matmul_recursive_serial_accum(N, h, C21, A21, B11, base_case);
        matmul_recursive_serial_accum(N, h, C21, A22, B21, base_case);

        matmul_recursive_serial_accum(N, h, C22, A21, B12, base_case);
        matmul_recursive_serial_accum(N, h, C22, A22, B22, base_case);
    }
}

static void matmul_recursive_task_accum(const int N, const int n,
                                        double *C, const double *A, const double *B,
                                        int base_case, int task_cutoff) {
    if (n <= base_case) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double cij = C[j + i * N];
                for (int k = 0; k < n; ++k) {
                    cij += A[k + i * N] * B[j + k * N];
                }
                C[j + i * N] = cij;
            }
        }
        return;
    }

    {
        const int h = n / 2;
        const double *A11 = A;
        const double *A12 = A + h;
        const double *A21 = A + h * N;
        const double *A22 = A + h * N + h;

        const double *B11 = B;
        const double *B12 = B + h;
        const double *B21 = B + h * N;
        const double *B22 = B + h * N + h;

        double *C11 = C;
        double *C12 = C + h;
        double *C21 = C + h * N;
        double *C22 = C + h * N + h;

        if (n <= task_cutoff) {
            matmul_recursive_serial_accum(N, h, C11, A11, B11, base_case);
            matmul_recursive_serial_accum(N, h, C11, A12, B21, base_case);

            matmul_recursive_serial_accum(N, h, C12, A11, B12, base_case);
            matmul_recursive_serial_accum(N, h, C12, A12, B22, base_case);

            matmul_recursive_serial_accum(N, h, C21, A21, B11, base_case);
            matmul_recursive_serial_accum(N, h, C21, A22, B21, base_case);

            matmul_recursive_serial_accum(N, h, C22, A21, B12, base_case);
            matmul_recursive_serial_accum(N, h, C22, A22, B22, base_case);
            return;
        }

        #pragma omp task default(none) firstprivate(N,h,base_case,task_cutoff) shared(C11,A11,B11,A12,B21)
        {
            matmul_recursive_task_accum(N, h, C11, A11, B11, base_case, task_cutoff);
            matmul_recursive_task_accum(N, h, C11, A12, B21, base_case, task_cutoff);
        }

        #pragma omp task default(none) firstprivate(N,h,base_case,task_cutoff) shared(C12,A11,B12,A12,B22)
        {
            matmul_recursive_task_accum(N, h, C12, A11, B12, base_case, task_cutoff);
            matmul_recursive_task_accum(N, h, C12, A12, B22, base_case, task_cutoff);
        }

        #pragma omp task default(none) firstprivate(N,h,base_case,task_cutoff) shared(C21,A21,B11,A22,B21)
        {
            matmul_recursive_task_accum(N, h, C21, A21, B11, base_case, task_cutoff);
            matmul_recursive_task_accum(N, h, C21, A22, B21, base_case, task_cutoff);
        }

        #pragma omp task default(none) firstprivate(N,h,base_case,task_cutoff) shared(C22,A21,B12,A22,B22)
        {
            matmul_recursive_task_accum(N, h, C22, A21, B12, base_case, task_cutoff);
            matmul_recursive_task_accum(N, h, C22, A22, B22, base_case, task_cutoff);
        }

        #pragma omp taskwait
    }
}

static void matmul_recursive_task_driver(int n, double *C, const double *A, const double *B,
                                         int base_case, int task_cutoff) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            matmul_recursive_task_accum(n, n, C, A, B, base_case, task_cutoff);
        }
    }
}

static double benchmark_task_version(int n, const double *A, const double *B, double *C,
                                     int threads, int trials, int base_case, int task_cutoff) {
    double best = 1.0e300;
    omp_set_num_threads(threads);

    for (int t = 0; t < trials; ++t) {
        zero_matrix(n, C);
        double start = omp_get_wtime();
        matmul_recursive_task_driver(n, C, A, B, base_case, task_cutoff);
        double end = omp_get_wtime();
        double dt = end - start;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    int n = (argc >= 2) ? atoi(argv[1]) : 1024;
    int trials = (argc >= 3) ? atoi(argv[2]) : 5;
    int base_case = (argc >= 4) ? atoi(argv[3]) : DEFAULT_BASE_CASE;
    int task_cutoff = (argc >= 5) ? atoi(argv[4]) : DEFAULT_TASK_CUTOFF;

    if (!is_power_of_two(n)) {
        fprintf(stderr, "Error: n must be a power of 2.\n");
        return 1;
    }
    if (!is_power_of_two(base_case) || !is_power_of_two(task_cutoff)) {
        fprintf(stderr, "Error: base_case and task_cutoff must be powers of 2.\n");
        return 1;
    }
    if (base_case > n || task_cutoff > n) {
        fprintf(stderr, "Error: base_case and task_cutoff must be <= n.\n");
        return 1;
    }

    double *A = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *B = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *C = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *Cref = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!A || !B || !C || !Cref) {
        fprintf(stderr, "Allocation failed.\n");
        free(A); free(B); free(C); free(Cref);
        return 1;
    }

    fill_random_matrix(n, A, 12345U);
    fill_random_matrix(n, B, 67890U);
    zero_matrix(n, C);
    zero_matrix(n, Cref);

    printf("n=%d, trials=%d, base_case=%d, task_cutoff=%d, max_threads_available=%d\n",
           n, trials, base_case, task_cutoff, omp_get_max_threads());

    matmul_reference(n, A, B, Cref);
    matmul_recursive_task_driver(n, C, A, B, base_case, task_cutoff);
    printf("task version max relative error = %.6e\n", max_rel_err(n, C, Cref));

    if (max_rel_err(n, C, Cref) > 1e-12) {
        fprintf(stderr, "FAILED correctness check.\n");
        free(A); free(B); free(C); free(Cref);
        return 2;
    }

    {
        int thread_list[5] = {1, 2, 4, 8, 16}; //initializing the number of threads to use
        double t1;
        printf("\nSTRONG SCALING (fixed problem size)\n");
        printf("threads,time_s,speedup,efficiency,gflops\n");
        t1 = benchmark_task_version(n, A, B, C, 1, trials, base_case, task_cutoff);
        for (int idx = 0; idx < 5; ++idx) {
            int th = thread_list[idx];
            double t = benchmark_task_version(n, A, B, C, th, trials, base_case, task_cutoff);
            double speedup = t1 / t;
            double efficiency = speedup / (double)th;
            double gflops = (2.0 * n * n * n) / t / 1.0e9;
            printf("%d,%.6f,%.6f,%.6f,%.6f\n", th, t, speedup, efficiency, gflops);
        }
    }

    {
        int thread_list[5] = {1, 2, 4, 8, 16};
        int base_n = 384;
        double t_base = -1.0;
        printf("\nWEAK SCALING (n ~ base_n * threads^(1/3))\n");
        printf("threads,n,time_s,relative_to_1thread\n");
        for (int idx = 0; idx < 5; ++idx) {
            int th = thread_list[idx];
            int nwe = (int)(base_n * cbrt((double)th) + 0.5);  //cbrt calculates cube root
             
            if (nwe < base_case) nwe = base_case;

            double *Aw = (double *)malloc((size_t)nwe * (size_t)nwe * sizeof(double));
            double *Bw = (double *)malloc((size_t)nwe * (size_t)nwe * sizeof(double));
            double *Cw = (double *)malloc((size_t)nwe * (size_t)nwe * sizeof(double));
            if (!Aw || !Bw || !Cw) {
                fprintf(stderr, "Weak scaling allocation failed for n=%d.\n", nwe);
                free(Aw); free(Bw); free(Cw);
                free(A); free(B); free(C); free(Cref);
                return 1;
            }

            fill_random_matrix(nwe, Aw, 1000U + (unsigned int)th);
            fill_random_matrix(nwe, Bw, 2000U + (unsigned int)th);
            double tw = benchmark_task_version(nwe, Aw, Bw, Cw, th, imin(trials, 3), base_case, imin(task_cutoff, nwe));
            if (t_base < 0.0) t_base = tw;
            printf("%d,%d,%.6f,%.6f\n", th, nwe, tw, tw / t_base);

            free(Aw); 
            free(Bw); 
            free(Cw);
        }
    }

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
