#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    double scalar = 3.0;
    int repeats = 10;

    double *a = malloc((size_t)N * sizeof(double));
    double *b = malloc((size_t)N * sizeof(double));
    double *c = malloc((size_t)N * sizeof(double));

    if (!a || !b || !c) {
        fprintf(stderr, "Allocation failed\n");
        free(a); 
        free(b); 
        free(c);
        return 1;
    }

    for (long long i = 0; i < N; i++) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    double best = 1e100;

    for (int r = 0; r < repeats; r++) {
        double t0 = omp_get_wtime();

        #pragma omp parallel for
        for (long long i = 0; i < N; i++) {
            a[i] = b[i] + scalar * c[i];
        }

        double t1 = omp_get_wtime();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    // Triad traffic: read b + read c + write a = 24 bytes/element
    double bandwidth_gbs = (24.0 * (double)N) / best / 1e9;

    printf("N = %lld\n", N);
    printf("Best time = %.6f s\n", best);
    printf("Bandwidth = %.3f GB/s\n", bandwidth_gbs);

    free(a);
    free(b);
    free(c);
    return 0;
}