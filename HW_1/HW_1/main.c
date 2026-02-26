#include "transpose.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// FUNCTION FOR FILLING A MATRIX (ROW MAJOR)
static void fill_matrix(int n, double *A) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i*n + j] = (double)(i + 1) + 1e-3 * (double)(j + 1);
        }
    }
}
//==============THIS FUNCTION RUNS THE TRANSPOSE FUNCTION AND RETURNS THE ERR AND TIME===============//
     
static void run_one(const char *name, transpose_fn f, int n,
                    double *A, double *AT, double *AT_ref) {

         
    transpose_reference(n, AT_ref, A);
    f(n, AT, A);

    const double err = max_error(n, AT, AT_ref);  //RETURNS MAX RELATIVE ERROR
    const double sec = time_transpose(f, n, AT, A, 30); //RETURNS MINIMUM TIME

    printf("%-10s  n=%d  TIME=%.12e s   MAX_REL_ERROR=%.3e\n",
           name, n, sec, err);
 
    if (err > 1e-14) {
        fprintf(stderr, "ERROR: %s failed correctness (err=%.3e)\n", name, err);
        exit(1);
    }
}

//=========================ALLOCATES # MATRIX===================================//
static void alloc_mats(int n, double **A, double **AT, double **AT_ref) {
    size_t N = (size_t)n * (size_t)n;
    *A      = (double*) aligned_alloc(64, N * sizeof(double));
    *AT     = (double*) aligned_alloc(64, N * sizeof(double));
    *AT_ref = (double*) aligned_alloc(64, N * sizeof(double));
    if (!(*A) || !(*AT) || !(*AT_ref)) {
            fprintf(stderr, "Allocation failed for n=%d\n", n);
            exit(1);
    }
    fill_matrix(n, *A);
    memset(*AT, 0, N * sizeof(double));
    memset(*AT_ref, 0, N * sizeof(double));
}

//================================PRINT CSV=============================================// 

static void print_csv(const char *impl, int n, int param, double sec, double err){
    printf("%s,%d,%d,%.10e,%.3e\n", impl, n, param, sec, err);
}

static void usage(const char *prog) {       //DEFINED USAGE  HELPER FUNCTION
    fprintf(stderr,
        "Usage:\n"
        "  %s [n]\n"
        "  %s --sweep-block n [trials]\n"
        "  %s --sweep-thresh n [trials]\n"
        "  %s --sizes Bopt Topt [trials]\n",
        prog, prog, prog, prog);
}
//================ WRAPPER TO GET OPTIMUM BLOCK SIZE===============================//
static void sweep_block(int n, int trials) {
    
    const int Bs[] = {4,8,12,13,14,15,16,24,32,48,64,96,128,192,256}; //BLOCK SIZES OF INTEREST
    const int nb = (int)(sizeof(Bs)/sizeof(Bs[0]));

    double *A, *AT, *AT_ref;
    alloc_mats(n, &A, &AT, &AT_ref);
 
    transpose_reference(n, AT_ref, A);

    printf("impl,n,param,time,err\n");
    double best_t = 1e300;
    int best_B = -1;

    for (int k = 0; k < nb; ++k) {
        Block_size = Bs[k];

        transpose_blocked(n, AT, A);
        double err = max_error(n, AT, AT_ref);
        if (err > 1e-14) {
            fprintf(stderr, "Blocked failed for B=%d (err=%.3e)\n", Bs[k], err);
            exit(1);
        }

         
        double t = time_transpose(transpose_blocked, n, AT, A, trials);  //CHECK OPTIMUM TIME
        print_csv("blocked", n, Bs[k], t, err);

        if (t < best_t) { 
            best_t = t; 
            best_B = Bs[k]; }   //SELECTS BEST BLOCK SIZE
    }

    fprintf(stderr, "Best block size for n=%d: Block=%d (time=%.6e s)\n", n, best_B, best_t);

    free(A); 
    free(AT); 
    free(AT_ref);
}
//================================================================================================//


//============DETERMIMNES OPTIMAL THERSHOLD SIZE FOR RECURSIVE TRANSPOSE==========================//

static void sweep_thresh(int n, int trials) {
    const int Ts[] = {8,16,24,32,48,64,96,128,192,256,384,512};
    const int nt = (int)(sizeof(Ts)/sizeof(Ts[0]));

    double *A, *AT, *AT_ref;
    alloc_mats(n, &A, &AT, &AT_ref);

    transpose_reference(n, AT_ref, A);

    printf("impl,n,param,time,err\n");
    double best_t = 1e300;
    int best_T = -1;

    for (int k = 0; k < nt; ++k) {
        Thresh = Ts[k];

        transpose_recursive(n, AT, A);
        double err = max_error(n, AT, AT_ref);
        if (err > 1e-14) {
            fprintf(stderr, "Recursive failed for T=%d (err=%.3e)\n", Ts[k], err);
            exit(1);
        }

        double t = time_transpose(transpose_recursive, n, AT, A, trials);
        print_csv("recursive", n, Ts[k], t, err);

        if (t < best_t) { 
            best_t = t; 
            best_T = Ts[k]; }
    }

    fprintf(stderr, "Best threshold for n=%d: Thresh=%d (time=%.6e s)\n", n, best_T, best_t);

    free(A); free(AT); free(AT_ref);
}

//================================================================================================//

//=====================GETS THE RUNTIME FOR EACH IMPLEMENTATION====================================//
 
/*static void run_sizes(int B_opt, int T_opt, int trials) {
    const int Ns[] = {32,64,128,256,512,1024,2048};
    const int nn = (int)(sizeof(Ns)/sizeof(Ns[0]));

    printf("impl,n,param,time,err\n");

    for (int idx = 0; idx < nn; ++idx) {
        int n = Ns[idx];
        double *A, *AT, *AT_ref;
        alloc_mats(n, &A, &AT, &AT_ref);

         
        transpose_reference(n, AT_ref, A);

        transpose_naive(n, AT, A);
        double errN = max_error(n, AT, AT_ref);
        double tN = time_transpose(transpose_naive, n, AT, A, trials);
        print_csv("naive", n, 0, tN, errN);

         
        Block_size = B_opt;
        transpose_blocked(n, AT, A);
        double errB = max_error(n, AT, AT_ref);
        double tB = time_transpose(transpose_blocked, n, AT, A, trials);
        print_csv("blocked", n, B_opt, tB, errB);

         
        Thresh = T_opt;
        transpose_recursive(n, AT, A);
        double errR = max_error(n, AT, AT_ref);
        double tR = time_transpose(transpose_recursive, n, AT, A, trials);
        print_csv("recursive", n, T_opt, tR, errR);

        free(A); free(AT); free(AT_ref);
    }
}*/

int main(int argc, char **argv) {

     
    if (argc >= 2 && argv[1][0] == '-') { //CHECK A SPECIAL MODE
    if (strcmp(argv[1], "--sweep-block") == 0) {
        if (argc < 3) { usage(argv[0]); return 1; }
        int n = atoi(argv[2]);
        int trials = (argc >= 4) ? atoi(argv[3]) : 30;
        if (n <= 0) { usage(argv[0]); return 1; }
        sweep_block(n, trials);
        return 0;
    }

    if (strcmp(argv[1], "--sweep-thresh") == 0) {
        if (argc < 3) { usage(argv[0]); return 1; }
        int n = atoi(argv[2]);
        int trials = (argc >= 4) ? atoi(argv[3]) : 30;
        if (n <= 0) { usage(argv[0]); return 1; }
        sweep_thresh(n, trials);
        return 0;
    }

    if (strcmp(argv[1], "--sizes") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        int Bopt = atoi(argv[2]);
        int Topt = atoi(argv[3]);
        int trials = (argc >= 5) ? atoi(argv[4]) : 30;
        if (Bopt <= 0 || Topt <= 0) { usage(argv[0]); return 1; }
        run_sizes(Bopt, Topt, trials);
        return 0;
    }

    usage(argv[0]);
    return 1;}

//===========================================================================//
    //THIS SECTION JUST TEST THE "time_transpose" FUNCTION ON DIFFERENT ALGORITHMS
    int n = 1024;  //CHANGE n TO WHATEVER YOU LIKE//
    if (argc >= 2) n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return 1;
    }

    const size_t N = (size_t)n * (size_t)n;
    double *A      = (double*) aligned_alloc(64, N * sizeof(double));
    double *AT     = (double*) aligned_alloc(64, N * sizeof(double));
    double *AT_ref = (double*) aligned_alloc(64, N * sizeof(double));

    if (!A || !AT || !AT_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    fill_matrix(n, A);
    memset(AT, 0, N * sizeof(double));
    memset(AT_ref, 0, N * sizeof(double));

    run_one("NAIVE",     transpose_naive,     n, A, AT, AT_ref); //RUNS NAIVE 
    run_one("BLOCKED",   transpose_blocked,   n, A, AT, AT_ref); //RUNS BLOCKED
    run_one("RECURSIVE", transpose_recursive, n, A, AT, AT_ref); //RUNS RECURSIVE

    free(A);
    free(AT);
    free(AT_ref);
//=======================================================================//
     
    return 0;
}