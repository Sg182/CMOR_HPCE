#include <omp.h>
#include "integration.h"

double integrate_reduction(long num_steps, int num_threads) {
    const double step = 1.0 / (double) num_steps;
    double sum = 0.0;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < num_steps; ++i) {
        const double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}
