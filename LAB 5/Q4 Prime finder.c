#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LIMIT 2000   // upper bound for prime search

// Function to check if a number is prime
int checkPrime(int value) {
    if (value <= 1) return 0;
    if (value <= 3) return 1;
    if (value % 2 == 0 || value % 3 == 0) return 0;

    for (int k = 5; k * k <= value; k += 6) {
        if (value % k == 0 || value % (k + 2) == 0)
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int id, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    if (id == 0) {
        // MASTER PROCESS
        int current = 2;
        int count = 0;
        int *primeList = (int *)malloc(LIMIT * sizeof(int));
        int workers = total - 1;

        MPI_Status stat;
        int incoming;

        printf("Master: Computing primes up to %d using %d workers...\n",
               LIMIT, workers);

        while (workers > 0) {
            MPI_Recv(&incoming, 1, MPI_INT, MPI_ANY_SOURCE,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

            int sender = stat.MPI_SOURCE;

            if (incoming == 0) {
                // Worker asking for new number
                if (current <= LIMIT) {
                    MPI_Send(&current, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
                    current++;
                } else {
                    int stop = -1;
                    MPI_Send(&stop, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
                    workers--;
                }
            } 
            else if (incoming > 0) {
                primeList[count++] = incoming;
            }
        }

        printf("\nTotal primes found: %d\n", count);
        for (int i = 0; i < count; i++) {
            printf("%d ", primeList[i]);
            if ((i + 1) % 10 == 0) printf("\n");
        }
        printf("\n");

        free(primeList);
    }
    else {
        // WORKER PROCESS
        int signal = 0;

        // Initial request for work
        MPI_Send(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            int receivedNum;

            MPI_Recv(&receivedNum, 1, MPI_INT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (receivedNum < 0)
                break;  // termination

            int output = checkPrime(receivedNum) ? receivedNum : -receivedNum;

            MPI_Send(&output, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Request next task
            MPI_Send(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}