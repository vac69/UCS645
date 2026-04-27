#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1 << 20)   // total elements

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const double scalar = 3.14;

    double *vecX = (double *)malloc(SIZE * sizeof(double));
    double *vecY = (double *)malloc(SIZE * sizeof(double));

    // Initialize only on root process
    if (pid == 0) {
        for (int i = 0; i < SIZE; ++i) {
            vecX[i] = i * 0.1;
            vecY[i] = i * 0.2;
        }
    }

    int chunk = SIZE / nprocs;

    double *subX = (double *)malloc(chunk * sizeof(double));
    double *subY = (double *)malloc(chunk * sizeof(double));

    // Distribute data
    MPI_Scatter(vecX, chunk, MPI_DOUBLE, subX, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vecY, chunk, MPI_DOUBLE, subY, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Perform DAXPY: X = aX + Y
    for (int j = 0; j < chunk; ++j) {
        subX[j] = scalar * subX[j] + subY[j];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // Collect results back to root
    MPI_Gather(subX, chunk, MPI_DOUBLE, vecX, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("DAXPY operation finished.\n");
        printf("Execution time using %d processes: %lf seconds\n", nprocs, t_end - t_start);
        printf("Compare runs with different process counts to evaluate speedup.\n");
    }

    free(subX);
    free(subY);
    free(vecX);
    free(vecY);

    MPI_Finalize();
    return 0;
}