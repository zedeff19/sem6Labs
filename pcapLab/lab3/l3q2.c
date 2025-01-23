#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N;
    int *A = NULL;  // Array to store input elements
    int *local_data = NULL;  // Array to store local elements for each process
    double local_avg, total_avg;
    double *all_averages = NULL;  // Array to store averages from all processes

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads M and N
        printf("Enter the number of processes N: ");
        fflush(stdout);
        scanf("%d", &N);  // N is the number of processes

        if (size != N) {
            printf("Error: The number of processes does not match N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Enter the number of elements M per process: ");
        fflush(stdout);
        scanf("%d", &M);  // M is the number of elements per process

        // Allocate memory for the array of size N*M
        A = (int*)malloc(N * M * sizeof(int));

        // Read N*M elements from the user
        printf("Enter %d elements:\n", N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &A[i]);
        }

        // Allocate memory to store the averages of each process
        all_averages = (double*)malloc(N * sizeof(double));
    }

    // Allocate memory for each process to store M elements
    local_data = (int*)malloc(M * sizeof(int));

    // Scatter the M elements to each process
    MPI_Scatter(A, M, MPI_INT, local_data, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the average of the received M elements
    int sum = 0;
    for (int i = 0; i < M; i++) {
        sum += local_data[i];
    }
    local_avg = (double)sum / M;

    // Gather the averages from all processes
    MPI_Gather(&local_avg, 1, MPI_DOUBLE, all_averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Root process computes the total average
        total_avg = 0.0;
        for (int i = 0; i < N; i++) {
            total_avg += all_averages[i];
        }
        total_avg /= N;

        // Print the result
        printf("Total average of all averages: %.2f\n", total_avg);

        // Clean up dynamically allocated memory
        free(A);
        free(all_averages);
    }

    // Clean up dynamically allocated memory for each process
    free(local_data);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
