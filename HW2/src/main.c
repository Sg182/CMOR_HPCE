#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "integration.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s <serial|reduction|atomic|all> <num_steps> <max_threads> <repeats>\n"
            "Example: %s all 100000000 8 3\n",
            prog, prog);
}

static void run_serial(long num_steps, int repeats) {
    double pi = 0.0;
    const double best = time_serial_best(num_steps, repeats, &pi);
    printf("%-10s %8s %14s %14s %16s\n", "impl", "threads", "pi", "runtime(s)", "parallel_eff");
    printf("%-10s %8d %14.12f %14.6f %16.6f\n", "serial", 1, pi, best, 1.0);
}

static void run_strong_scaling(const char *impl, long num_steps, int max_threads, int repeats) {
    double pi1 = 0.0;
    const double t1 = time_parallel_best(impl, num_steps, 1, repeats, &pi1);

    printf("\nImplementation: %s\n", impl);
    printf("%-10s %8s %14s %14s %16s\n", "impl", "threads", "pi", "runtime(s)", "parallel_eff");

    for (int t = 1; t <= max_threads; t *= 2) {
        double pi = 0.0;
        double tp = 0.0;
        double efficiency = 0.0;

        if (t == 1) {
            tp = t1;
            pi = pi1;
            efficiency = 1.0;
        } else {
            tp = time_parallel_best(impl, num_steps, t, repeats, &pi);
            efficiency = t1 / (tp * (double) t);
        }
        printf("%-10s %8d %14.12f %14.6f %16.6f\n", impl, t, pi, tp, efficiency);
    }
}

int main(int argc, char **argv) {
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const long num_steps = atol(argv[2]);
    const int max_threads = atoi(argv[3]);
    const int repeats = atoi(argv[4]);

    if (num_steps <= 0 || max_threads <= 0 || repeats <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(mode, "serial") == 0) {
        run_serial(num_steps, repeats);
    } else if (strcmp(mode, "reduction") == 0) {
        run_strong_scaling("reduction", num_steps, max_threads, repeats);
    } else if (strcmp(mode, "atomic") == 0) {
        run_strong_scaling("atomic", num_steps, max_threads, repeats);
    } else if (strcmp(mode, "all") == 0) {
        run_serial(num_steps, repeats);
        run_strong_scaling("reduction", num_steps, max_threads, repeats);
        run_strong_scaling("atomic", num_steps, max_threads, repeats);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
