#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <axpy.h>




void print_vector(const char *name, int n, double *v) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.2f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

void init_array(int n, double *x, double *y){
    for (int i=0; i <n; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }
}

double benchmark(void(*axpy)(int,double*,double,double*),int n, double *x,double *y, 
        double alpha,int repeats){

        double best = 1e100;
        for (int i=0; i<repeats;++i){
            init_array(n, x, y);
            double t0 = omp_get_wtime();
            axpy(n, y, alpha, x);
            double t1 = omp_get_wtime();

            double dt = t1 - t0;
            if (dt < best){
                best = dt;
            }
        }
        return best;

        }

int main(int argc, char **argv) {
    
    if (argc != 2){
        printf("Using %s \n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    double alpha = 2.0;
    int repeats = 5;

    double *x = malloc((size_t)n*sizeof(double));
    double *y = malloc((size_t)n*sizeof(double));

    if (!x || !y ) {
    printf("Memory allocation failed\n");
    return 1;
}

    double t1 = benchmark(axpy_version_1, n,x,y,alpha,repeats);
    double t2 = benchmark(axpy_version_2, n,x,y,alpha,repeats);
    
    printf("n = %d\n", n);
    printf("axpy_version_1 time = %.6f s\n", t1);
    printf("axpy_version_2 time = %.6f s\n", t2);

    free(x);
    free(y);


    
    return 0;
}