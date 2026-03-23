#include <float.h>
#include <string.h>
#include <omp.h>
#include "integration.h"

double time_serial_best(long num_steps, int repeats, double *pi_out) {
    double best = DBL_MAX;
    double pi = 0.0;

    for (int r = 0; r < repeats; ++r) {
        const double t0 = omp_get_wtime();
        pi = integrate_serial(num_steps);
        const double t1 = omp_get_wtime();
        const double dt = t1 - t0;
        if (dt < best) {
            best = dt;
        }
    }

    if (pi_out != NULL) {
        *pi_out = pi;
    }
    return best;
}

double time_parallel_best(const char *impl, long num_steps, int num_threads, int repeats, double *pi_out) {
    double best = DBL_MAX;
    double pi = 0.0;

    for (int r = 0; r < repeats; ++r) {
        const double t0 = omp_get_wtime();

        if (strcmp(impl, "reduction") == 0) {
            pi = integrate_reduction(num_steps, num_threads);
        } else if (strcmp(impl, "atomic") == 0) {
            pi = integrate_atomic(num_steps, num_threads);
        } else {
            return -1.0;
        }

        const double t1 = omp_get_wtime();
        const double dt = t1 - t0;
        if (dt < best) {
            best = dt;
        }
    }

    if (pi_out != NULL) {
        *pi_out = pi;
    }
    return best;
}
