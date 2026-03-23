#include "integration.h"

double integrate_serial(long num_steps) {
    const double step = 1.0 / (double) num_steps;
    double sum = 0.0;

    for (long i = 0; i < num_steps; ++i) {
        const double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}
