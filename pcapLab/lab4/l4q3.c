#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define N 4  // Matrix size (4x4)

int main(int argc, char **argv) {
    int rank, size;
    int matrix[N][N];   // 4x4 Matrix
    int partial_sum[N]; // Array to hold the cumulative sum for each column
    int recv_data[N][N];  // Array to hold the final result of the progressive sum
    int i, j;

    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the number of processes

    // Ensure that we have 4 processes (one for each column of the matrix)
    if (size != N) {
        if (rank == 0) {
            printf("This program requires exactly 4 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Input the matrix on the root process (rank 0)
    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(matrix, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the progressive sum (cumulative sum) for each column
    for (i = 0; i < N; i++) {
        partial_sum[i] = matrix[i][rank];  // Start with the current element in the column
        // Compute the cumulative sum for each column
        for (j = i - 1; j >= 0; j--) {
            partial_sum[i] += matrix[j][rank];
        }
    }

    // Gather the results into recv_data from all processes
    MPI_Gather(partial_sum, N, MPI_INT, recv_data[rank], N, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process will print the result
    if (rank == 0) {
        printf("\nProgressive sum of each column (Cumulative sum):\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", recv_data[i][j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
