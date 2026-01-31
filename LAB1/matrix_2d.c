#include <stdio.h>
#include <omp.h>

#define N 1000

int main() {
    static double A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = 1.0;

    double start = omp_get_wtime();

    // 2D threading: collapse nested loops
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }

    double end = omp_get_wtime();
    printf("Matrix Multiplication (2D Threading) Time: %f seconds\n", end - start);

    return 0;
}
