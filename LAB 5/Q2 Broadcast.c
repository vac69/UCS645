#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_COUNT 10000000   // 10 million elements

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int my_id, total_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    double *buffer = (double *)malloc(DATA_COUNT * sizeof(double));
    double t1, t2;

    if (my_id == 0) {
        printf("=== Execution with %d processes ===\n", total_procs);
    }

    /* ----------- Part A: Manual Broadcast (Linear Send) ----------- */
    if (my_id == 0) {
        for (int i = 0; i < DATA_COUNT; ++i) {
            buffer[i] = i * 0.1;
        }

        t1 = MPI_Wtime();
        for (int dest = 1; dest < total_procs; ++dest) {
            MPI_Send(buffer, DATA_COUNT, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(buffer, DATA_COUNT, MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == 0) {
        t2 = MPI_Wtime();
        printf("Custom Broadcast (Linear) Time: %lf seconds\n", t2 - t1);
    }

    /* ----------- Part B: Built-in MPI_Bcast ----------- */
    if (my_id == 0) {
        for (int i = 0; i < DATA_COUNT; ++i) {
            buffer[i] = i * 0.1;
        }
        t1 = MPI_Wtime();
    }

    MPI_Bcast(buffer, DATA_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == 0) {
        t2 = MPI_Wtime();
        printf("MPI Built-in Broadcast Time: %lf seconds\n", t2 - t1);
    }

    free(buffer);

    MPI_Finalize();
    return 0;
}