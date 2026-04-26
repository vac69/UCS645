#include <mpi.h>
#include <stdio.h>

#define SIZE 8

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int elements_per_proc = SIZE / num_procs;

    int vec1[SIZE] = {1,2,3,4,5,6,7,8};
    int vec2[SIZE] = {8,7,6,5,4,3,2,1};

    int part1[elements_per_proc];
    int part2[elements_per_proc];

    MPI_Scatter(vec1, elements_per_proc, MPI_INT,
                part1, elements_per_proc, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vec2, elements_per_proc, MPI_INT,
                part2, elements_per_proc, MPI_INT,
                0, MPI_COMM_WORLD);

    double partial_result = 0.0;
    for (int idx = 0; idx < elements_per_proc; idx++) {
        partial_result += (double)part1[idx] * part2[idx];
    }

    double total_result = 0.0;
    MPI_Reduce(&partial_result, &total_result, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Dot Product Result: %.0f (expected 120)\n", total_result);
    }

    MPI_Finalize();
    return 0;
}