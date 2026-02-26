#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
 
using namespace std::chrono;
using namespace std;

// should be a power of 2
#define MIN_BLOCK_SIZE 8

// N = full matrix size
// n = block size
void matmul_recursive(const int N, const int n, double *C, double *A, double *B)
{
    if (n <= MIN_BLOCK_SIZE)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                double Cij = C[j + i * N];
                for (int k = 0; k < n; ++k)
                {
                    // C_ij <-- A_ik * B_kj
                    double Aik = A[k + i * N];
                    double Bkj = B[j + k * N];
                    Cij += Aik * Bkj;
                }
                C[j + i * N] = Cij;
            }
        }
    }
    else
    {
        // A = [A11 A12
        //      A21 A22]
        double * A11 = A;
        double * A12 = A + n / 2;
        double * A21 = A + (n / 2) * N;
        double * A22 = A + N * (n / 2) + n / 2;

        double * B11 = B;
        double * B12 = B + n / 2;
        double * B21 = B + (n / 2) * N;
        double * B22 = B + N * (n / 2) + n / 2;

        double * C11 = C;
        double * C12 = C + n / 2;
        double * C21 = C + (n / 2) * N;
        double * C22 = C + N * (n / 2) + n / 2;

        // top left block
        matmul_recursive(N, n / 2, C11, A11, B11);
        matmul_recursive(N, n / 2, C11, A12, B21); 

        // top right block
        matmul_recursive(N, n / 2, C12, A11, B12);
        matmul_recursive(N, n / 2, C12, A12, B22);

        // bottom left block
        matmul_recursive(N, n / 2, C21, A21, B11);
        matmul_recursive(N, n / 2, C21, A22, B21);

        // bottom right block
        matmul_recursive(N, n / 2, C22, A21, B12);
        matmul_recursive(N, n / 2, C22, A22, B22);

    }
}

void print_matrix(int n, double *A){
    printf("A = \n");
    for (int i = 0; i < n; ++i){        
        for (int j = 0; j < n; ++j){
            printf("%1.1f ", A[j + i * n]);
        }
        printf("\n");
    }
}

static void matmul_reference(int n, const double* A, const double* B, double* Cref){

    for (int i=0; i<n;++i){
        for (int j=0;j<n;++j){
            double sum = 0.0;
            for (int k=0; k<n; ++k){
                sum += A[i*n +k]*B[k*n +j];
            }
            Cref[i*n +j] = sum;

        }
    }
}

static double max_rel_err(int n, const double* C, const double* Cref){
    double m =0.0;

    for (int i =0;i<n*n; ++i){
        double denom = max(1.0, fabs(Cref[i]));
        double rel = fabs(C[i]- Cref[i])/denom;
        m = max(m, rel);
    }
    return m;
}

static bool is_power_of_two(int n){
    return n >0 && (n &(n-1)) ==0;
}


// note: this should only be run with n as some power of 2
int main(int argc, char * argv[]){

    int n = (argc >=2) ? atoi(argv[1]): 256;

    if (!is_power_of_two(n)){
        cerr << "Error: n must be a power of 2!\n";
        return 1;
    }
    cout << "Matrix size n = " << n << ", recursive threshhold = " << MIN_BLOCK_SIZE << endl;
    
    double * A = new double[n * n];
    double * B = new double[n * n];
    double * C = new double[n * n];
    double * Cref = new double[n*n];
  
    // A, B = identity matrices
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            A[j + i * n] = (i == j) ? 1.0 : 0.0;
            B[j + i * n] = (i == j) ? 1.0 : 0.0;
            C[j + i * n] = 0.0;
            Cref[i*n + j] = 0.0;
        }
    }
    
    matmul_reference(n, A,B,Cref);
    matmul_recursive(n, n, C, A, B); 
    // print_matrix(n, C);

    //==============CHECK RELATIVE ERROR================
    double err;

    err = max_rel_err(n, C, Cref);
    cout << "max relative error = "<< err<<"\n";

     
    if (err > 1e-14){
        cerr << "FAILED correctness check.\n";
        return 2;
    }

    cout << "PASSED correctness check.\n";
    //===================================================//

    int num_samples = 25;  //NUMBER OF TRIALS
    double min_time_recursive = 1e18;

    for (int t = 0; t < num_samples; ++t) {
        std::fill(C, C + n*n, 0.0);  

        auto start = high_resolution_clock::now();
        matmul_recursive(n, n, C, A, B);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start);
        min_time_recursive = std::min(min_time_recursive, (double)duration.count());
    }

    cout << "Elapsed time for recursive matmul in (secs): "
         << min_time_recursive*1e-6 << endl;

    delete[] A;
    delete[] B;
    delete[] C;  
    
    return 0;
  }