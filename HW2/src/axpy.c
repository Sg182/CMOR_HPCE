#include <omp.h>
#include <axpy.h>

void axpy_version_1(int n, double *y, double alpha, double *x) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();

        for (int i = tid; i < n; i += nt) {
            y[i] += alpha * x[i];
        }
    }
}


void axpy_version_2(int n, double *y, double alpha, double *x){
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();

        int chunk = n/nt;
        int rem = n%nt;
        int start,end;

        if (tid < rem){
             start = tid*(chunk+1);
             end = (start + chunk) +1;
        }else{
            start = rem*(chunk+1) + (tid - rem)*chunk;
            end = start + chunk;

        }
        for (int i = start;i<end;i++){
            y[i] += x[i]*alpha;
        }


    }
}