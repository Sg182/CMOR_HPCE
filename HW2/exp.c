#include <stdio.h>
#include <omp.h>

int main(){
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        for (int i = tid; i < 6; i+=nt){
            printf("The number %d is run by thread %d\n",i,tid);
        }
    }

    #pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();
        printf("Hello from next program thread %d\n", tid);
    }

    return 0;
}