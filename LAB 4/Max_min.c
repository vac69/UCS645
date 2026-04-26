#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int proc_rank, total_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    // Seed random generator uniquely per process
    srand((unsigned int)(time(NULL) + proc_rank));

    int max_val = -1;
    int min_val = 1001;

    // Generate random numbers and track local min/max
    for (int count = 0; count < 10; count++) {
        int value = rand() % 1001;

        if (value > max_val) {
            max_val = value;
        }
        if (value < min_val) {
            min_val = value;
        }
    }

    int max_info[2] = {max_val, proc_rank};
    int min_info[2] = {min_val, proc_rank};

    int final_max[2];
    int final_min[2];

    MPI_Reduce(max_info, final_max, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(min_info, final_min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
        printf("Maximum value: %d found at process %d\n", final_max[0], final_max[1]);
        printf("Minimum value: %d found at process %d\n", final_min[0], final_min[1]);
    }

    MPI_Finalize();
    return 0;
}