#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int proc_id, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    int total_elements = 100;
    int chunk_size = total_elements / proc_count;

    int *main_data = NULL;
    int *sub_data = (int *)malloc(sizeof(int) * chunk_size);

    if (proc_id == 0) {
        main_data = (int *)malloc(sizeof(int) * total_elements);
        for (int j = 0; j < total_elements; j++) {
            main_data[j] = j + 1;
        }
    }

    MPI_Scatter(main_data, chunk_size, MPI_INT,
                sub_data, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    int partial_sum = 0;
    for (int k = 0; k < chunk_size; k++) {
        partial_sum += sub_data[k];
    }

    int final_sum = 0;
    MPI_Reduce(&partial_sum, &final_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (proc_id == 0) {
        printf("Total sum: %d (expected 5050)\n", final_sum);
    }

    free(sub_data);
    if (proc_id == 0) {
        free(main_data);
    }

    MPI_Finalize();
    return 0;
}