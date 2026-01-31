#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1<<16)

int main() {
    double *X = (double*)malloc(N * sizeof(double));
    double *Y = (double*)malloc(N * sizeof(double));
    double a = 2.5;

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    // DAXPY with varying thread counts
    for (int threads = 2; threads <= 8; threads *= 2) {
        omp_set_num_threads(threads);

        double start = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            X[i] = a * X[i] + Y[i];
        }

        double end = omp_get_wtime();

        printf("Threads: %d | Execution Time: %f seconds\n",
               threads, end - start);
    }

    free(X);
    free(Y);

    return 0;
}
