#ifndef INTEGRATION_H
#define INTEGRATION_H

    double integrate_serial(long num_steps);
    double integrate_reduction(long num_steps, int num_threads);
    double integrate_atomic(long num_steps, int num_threads);

    double get_time();
    double time_serial_best(long num_steps, int repeats, double *pi_out);
    double time_parallel_best(const char *impl, long num_steps, int num_threads, int repeats, double *pi_out);

#endif