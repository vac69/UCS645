#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;

int main() {
    double step = 1.0 / (double) num_steps;
    double sum = 0.0;

    double start = omp_get_wtime();

    // Parallel computation using reduction
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end = omp_get_wtime();

    printf("Calculated value of Pi = %.12f\n", pi);
    printf("Execution Time = %f seconds\n", end - start);

    return 0;
}
