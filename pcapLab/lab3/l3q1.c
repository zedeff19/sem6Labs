#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

// Function to calculate factorial
long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    int *A = NULL;  
    long long *B = NULL;
    long long factorial_value, sum = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        
        printf("Enter the number of values N: ");
        fflush(stdout);
        scanf("%d", &N);
        
        if (size > N) {
            printf("Error: The number of processes cannot be greater than the number of values.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A = (int*)malloc(N * sizeof(int));
        B = (long long*)malloc(N * sizeof(long long));

        printf("Enter %d values: \n", N);
        fflush(stdout);
        for (int i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    int value;

    MPI_Scatter(A, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    factorial_value = factorial(value);

    MPI_Gather(&factorial_value, 1, MPI_LONG_LONG, B, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Root process computes the sum of all factorials
        for (int i = 0; i < N; i++) {
            sum += B[i];
        }

        // Print the result
        printf("Sum of all factorials: %lld\n", sum);

        // Clean up dynamically allocated memory
        free(A);
        free(B);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
