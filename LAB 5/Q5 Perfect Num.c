#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define LIMIT 10000   // upper bound for search

// Function to check perfect number
int checkPerfect(int num) {
    if (num <= 1) return 0;

    int total = 1;
    for (int d = 2; d * d <= num; ++d) {
        if (num % d == 0) {
            total += d;
            if (d != num / d)
                total += num / d;
        }
    }
    return (total == num);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int id, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (id == 0) {
        // MASTER PROCESS
        int current_num = 2;
        int found_count = 0;
        int *results = (int *)malloc(100 * sizeof(int));  // buffer

        int workers_active = proc_count - 1;
        MPI_Status stat;
        int incoming_val;

        printf("Master: Searching perfect numbers up to %d using %d workers...\n",
               LIMIT, workers_active);

        while (workers_active > 0) {
            MPI_Recv(&incoming_val, 1, MPI_INT, MPI_ANY_SOURCE,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

            int sender = stat.MPI_SOURCE;

            if (incoming_val == 0) {
                // Worker requests new task
                if (current_num <= LIMIT) {
                    MPI_Send(&current_num, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
                    current_num++;
                } else {
                    int stop_flag = -1;
                    MPI_Send(&stop_flag, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
                    workers_active--;
                }
            } 
            else if (incoming_val > 0) {
                results[found_count++] = incoming_val;
            }
        }

        printf("\nPerfect numbers found (%d): ", found_count);
        for (int i = 0; i < found_count; ++i) {
            printf("%d ", results[i]);
        }
        printf("\n");

        free(results);
    }
    else {
        // WORKER PROCESS
        int req_signal = 0;

        // Initial request
        MPI_Send(&req_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            int value;

            MPI_Recv(&value, 1, MPI_INT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (value < 0)
                break;   // termination condition

            int output_val = checkPerfect(value) ? value : -value;

            MPI_Send(&output_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Ask for next number
            MPI_Send(&req_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}