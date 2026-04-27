#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 50000000   // 50 million elements

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int proc_id, proc_total;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_total);

    double scale = 2.5;

    if (proc_id == 0) {
        printf("Scaling factor (default = 2.5) is being used.\n");
    }

    // Broadcast scaling factor to all processes
    MPI_Bcast(&scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int chunk_size = TOTAL_SIZE / proc_total;
    double partial_sum = 0.0;

    double t_start = MPI_Wtime();

    // Each process computes its portion
    for (int idx = 0; idx < chunk_size; ++idx) {
        double val1 = 1.0;
        double val2 = 2.0 * scale;
        partial_sum += val1 * val2;
    }

    double final_sum = 0.0;

    // Combine results from all processes
    MPI_Reduce(&partial_sum, &final_sum, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (proc_id == 0) {
        printf("Computed Dot Product: %f\n", final_sum);
        printf("Execution Time (%d processes): %lf seconds\n",
               proc_total, t_end - t_start);
        printf("Try multiple process counts (1,2,4,8) to analyze performance.\n");
    }

    MPI_Finalize();
    return 0;
}